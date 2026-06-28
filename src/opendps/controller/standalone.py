"""Standalone (zero-k8s) control loop — the N1 demo binary.

Architecture
------------
    PromClient ──reads──► NodeSample          (--prom mode, default)
                               │
    SimBackend ──reads──►      │              (--sim mode: actuator IS the source)
                               │
                               ▼
                         DomainState (per domain)
                               │
                               ▼
                          DPMBrain.decide()
                               │
                               ▼
                     Actuator.set_power_cap()

Telemetry comes either from Prometheus (DCGM metrics via dcgm-exporter) or,
when --sim is used, directly from the SimBackend (closed-loop demo without
a running Prometheus).  Cap enforcement goes through the pluggable Actuator
(SimBackend in N1, real NVML/dcgmi agent from N3 onward).

CLI
---
    opendps-controller --prom http://localhost:9090 --config topology.json \\
                        --interval 5 [--dry-run]

    opendps-controller --config topology.json --sim [--interval 5] [--dry-run]

Decisions are printed to stdout as JSONL (one object per domain per tick).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from opendps.brain.dpm import BrainDecision, DPMBrain, DomainState
from opendps.brain.prs import PRSBrain
from opendps.pdn.model import PDNTopology, from_dict
from opendps.sim.protocol import Actuator
from opendps.telemetry.metrics import start_healthz_server, start_metrics_server, update_decision_metrics
from opendps.telemetry.prom_client import NodeSampleFromProm, PromClient

log = logging.getLogger(__name__)


@dataclass
class ControllerConfig:
    """Runtime configuration for StandaloneController."""

    topology: PDNTopology
    actuator: Actuator
    prom_url: str = "http://localhost:9090"
    interval_s: float = 5.0
    domain_names: list[str] = field(default_factory=list)  # empty = all domains
    dry_run: bool = False                                   # log-only mode
    sim_mode: bool = False                                  # if True, read telemetry from actuator
    brain_type: str = "prs"                                 # "dpm" | "prs" | "job-prs"
    metrics_port: int | None = None                         # Prometheus /metrics port (None = disabled)
    actuator_type: str = "sim"                              # "sim" | "nvml" | "agent"
    agent_host: str = "127.0.0.1"
    agent_port: int = 9500
    # N5 — failsafe hardening / transient smoothing knobs (PRS-family brains)
    cap_raise_rate_w_per_tick: float = 0.0                  # 0 = unlimited
    ewma_alpha: float = 0.3
    # N6 — sim/demo: GPUs to mark busy for job-prs without nvidia-smi
    busy_gpus: list[int] = field(default_factory=list)
    # "prom" (default) reads draws from Prometheus; "actuator" reads them
    # directly from the actuator (real NVML node without a telemetry plane).
    telemetry: str = "prom"


class StandaloneController:
    """
    Main control loop: read telemetry → run brain → push caps.

    On each tick (run_once):
      1. Obtain telemetry — either from Prometheus (default) or the SimBackend
         (when sim_mode=True).
      2. For each managed domain:
         a. Filter telemetry to GPUs that belong to the domain.
         b. Build a DomainState (including hardware-max caps for ratchet recovery).
         c. Call DPMBrain.decide().
         d. If not dry_run: push caps via actuator.
         e. Print the decision as JSONL for downstream consumption.
      3. If sim_mode: advance the sim by calling actuator.tick().
    """

    def __init__(self, config: ControllerConfig) -> None:
        self._config = config
        if config.brain_type == "prs":
            self._brain: DPMBrain | PRSBrain = PRSBrain(
                config.topology,
                ewma_alpha=config.ewma_alpha,
                cap_raise_rate_w_per_tick=config.cap_raise_rate_w_per_tick,
            )
        elif config.brain_type == "cvxpy":
            from opendps.brain.cvxpy_brain import CVXPYBrain
            from typing import Any
            self._brain: Any = CVXPYBrain(config.topology)
        elif config.brain_type == "job-prs":
            from opendps.agent.job_tracker import JobTracker
            from opendps.brain.job_aware_prs import JobAwarePRSBrain
            from typing import Any
            tracker = JobTracker()
            if config.busy_gpus:
                # Sim/demo: no nvidia-smi available — seed a fixed busy set
                # instead of starting the polling thread.
                tracker.set_busy_gpus(config.busy_gpus)
            else:
                tracker.start()
            self._brain: Any = JobAwarePRSBrain(
                config.topology,
                tracker,
                ewma_alpha=config.ewma_alpha,
                cap_raise_rate_w_per_tick=config.cap_raise_rate_w_per_tick,
            )
        elif config.brain_type == "quota-prs":
            from opendps.brain.quota_prs import QuotaAwarePRSBrain
            from opendps.pdn.quota import QuotaConfig, TenantQuota
            from typing import Any
            # Default: two tenants splitting GPUs 60/40 (override via config file)
            n = len(list(config.topology.domains.values())[0].gpu_indices)
            half = n // 2
            quota = QuotaConfig(
                domain_name=list(config.topology.domains.keys())[0],
                tenants=[
                    TenantQuota("teamA", list(config.topology.domains.keys())[0],
                                list(range(half)), max_watts_pct=0.6),
                    TenantQuota("teamB", list(config.topology.domains.keys())[0],
                                list(range(half, n)), max_watts_pct=0.4),
                ],
            )
            self._brain: Any = QuotaAwarePRSBrain(config.topology, quota)
        else:
            self._brain = DPMBrain(config.topology)
        # Only create a PromClient when draws actually come from Prometheus.
        _use_prom = not config.sim_mode and config.telemetry != "actuator"
        self._client: PromClient | None = (
            PromClient(config.prom_url) if _use_prom else None
        )
        self._managed_domains: list[str] = (
            list(config.domain_names) if config.domain_names else list(config.topology.domains)
        )
        if config.metrics_port is not None:
            start_metrics_server(config.metrics_port)
            log.info("Prometheus metrics server started on port %d", config.metrics_port)
            start_healthz_server(config.metrics_port + 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_once(self) -> list[BrainDecision]:
        """Run one control tick.

        Returns the list of BrainDecisions (one per managed domain).
        In prom mode, raises on Prometheus connectivity errors — callers decide
        whether to swallow or propagate.
        """
        # Read telemetry from the actuator directly (no Prometheus) in sim mode
        # or when --telemetry actuator is set — e.g. a real NVML node where the
        # agent/actuator can report live draws. Otherwise pull from Prometheus.
        read_actuator = self._config.sim_mode or self._config.telemetry == "actuator"
        if read_actuator:
            ts = time.time()
            gpu_by_index = None
        else:
            node_sample = NodeSampleFromProm(self._client)
            gpu_by_index = {g.index: g for g in node_sample.gpus}
            ts = node_sample.ts

        decisions: list[BrainDecision] = []

        for domain_name in self._managed_domains:
            domain = self._config.topology.domains[domain_name]
            gpu_draws: dict[int, float] = {}
            gpu_caps: dict[int, float] = {}
            gpu_max_caps: dict[int, float] = {}

            fallback_cap = domain.budget_w / max(len(domain.gpu_indices), 1)

            for idx in domain.gpu_indices:
                if read_actuator:
                    gpu_draws[idx] = self._config.actuator.get_power_draw(idx)
                    gpu_caps[idx] = self._config.actuator.get_power_cap(idx)
                    if hasattr(self._config.actuator, "get_max_cap_w"):
                        gpu_max_caps[idx] = self._config.actuator.get_max_cap_w(idx)
                    else:
                        gpu_max_caps[idx] = fallback_cap
                else:
                    assert gpu_by_index is not None
                    g = gpu_by_index.get(idx)
                    if g is None:
                        log.warning("GPU %d not in Prometheus sample, skipping", idx)
                        continue
                    gpu_draws[idx] = g.power_draw_w if g.power_draw_w is not None else 0.0
                    gpu_caps[idx] = (
                        g.power_limit_w if g.power_limit_w is not None else fallback_cap
                    )
                    gpu_max_caps[idx] = (
                        g.power_max_limit_w if g.power_max_limit_w is not None else fallback_cap
                    )

            state = DomainState(
                domain_name=domain_name,
                gpu_draws=gpu_draws,
                gpu_caps=gpu_caps,
                gpu_max_caps=gpu_max_caps,
                ts=ts,
            )
            decision = self._brain.decide(domain_name, state)
            decisions.append(decision)

            if not self._config.dry_run:
                # Check the class (not the instance) so MagicMock in tests doesn't
                # shadow the real method check via __getattr__.
                if callable(getattr(type(self._config.actuator), 'push_all_caps', None)):
                    self._config.actuator.push_all_caps(decision.caps)
                else:
                    for gpu_idx, cap_w in decision.caps.items():
                        self._config.actuator.set_power_cap(gpu_idx, cap_w)
            else:
                log.info("[dry-run] would push caps for domain=%s: %s", domain_name, decision.caps)

            # Update Prometheus metrics if a port was configured. Stranded watts
            # are computed the SAME way for every brain (idle headroom = cap −
            # draw on GPUs running below the hot threshold) so the DPM baseline
            # and PRS are directly comparable — DPM strands the idle headroom it
            # statically allocates; PRS reclaims it. Special-casing PRS here
            # (the previous behaviour) reported 0 stranded watts for DPM, which
            # made the headline before/after comparison impossible.
            if self._config.metrics_port is not None:
                stats = _domain_stats(gpu_draws, decision.caps)
                update_decision_metrics(
                    domain_name,
                    prs_active=self._config.brain_type in ("prs", "job-prs", "quota-prs"),
                    domain_draw_w=stats["draw_w"],
                    domain_cap_w=stats["cap_w"],
                    idle_stranded_w=stats["idle_stranded_w"],
                    hot_count=stats["hot_count"],
                    idle_count=stats["idle_count"],
                    gpu_draws=gpu_draws,
                    gpu_caps=decision.caps,
                )

            # Emit as JSONL (one line per domain per tick).
            print(json.dumps({
                "ts": decision.ts,
                "domain": decision.domain,
                "caps": {str(k): v for k, v in decision.caps.items()},
                "reason": decision.reason,
            }), flush=True)

        # Advance sim state at the end of each tick so the next read reflects
        # the effect of the caps that were just applied.
        if self._config.sim_mode and hasattr(self._config.actuator, "tick"):
            self._config.actuator.tick()

        return decisions

    def run(self) -> None:
        """Run forever at config.interval_s.

        Errors within a single tick are logged and swallowed so a transient
        Prometheus outage does not kill the process.
        """
        log.info(
            "StandaloneController started — interval=%.1fs dry_run=%s sim_mode=%s domains=%s",
            self._config.interval_s,
            self._config.dry_run,
            self._config.sim_mode,
            self._managed_domains,
        )
        while True:
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                log.error("tick error: %s", exc)
            time.sleep(self._config.interval_s)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``opendps-controller`` CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="opendps-controller",
        description="opendps standalone control loop (brain v1 / DPM)",
    )
    parser.add_argument(
        "--prom",
        default="http://localhost:9090",
        metavar="URL",
        help="Prometheus base URL (default: http://localhost:9090)",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="FILE",
        help="PDN topology JSON file",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Control loop interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log decisions without pushing power caps",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help=(
            "Use SimBackend as telemetry source and enforcement target. "
            "Closes the feedback loop without a running Prometheus or real GPUs."
        ),
    )
    parser.add_argument(
        "--brain",
        choices=["dpm", "prs", "cvxpy", "job-prs", "quota-prs"],
        default="prs",
        help="Brain algorithm: dpm = static proportional (v1), prs = EWMA reclaim (v2, default), cvxpy = LP solver (v3), quota-prs = per-tenant quota enforcement (N13)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        metavar="PORT",
        help="Start Prometheus /metrics HTTP server on this port (e.g. 9402)",
    )
    parser.add_argument(
        "--actuator",
        choices=["sim", "nvml", "agent"],
        default="sim",
        help=(
            "Actuator backend: sim = SimBackend (default, no GPU required), "
            "nvml = direct pynvml calls (requires NVIDIA GPU + pynvml), "
            "agent = TCP to opendps-agent Rust process at --agent-host:--agent-port"
        ),
    )
    parser.add_argument("--agent-host", default="127.0.0.1", help="opendps-agent host (--actuator agent)")
    parser.add_argument("--agent-port", type=int, default=9500, help="opendps-agent port (--actuator agent)")
    parser.add_argument(
        "--cap-raise-rate",
        type=float,
        default=0.0,
        metavar="WATTS",
        help="N5: max watts a per-GPU cap may rise per tick (transient smoothing; 0 = unlimited)",
    )
    parser.add_argument(
        "--ewma-alpha",
        type=float,
        default=0.3,
        metavar="ALPHA",
        help="N5: EWMA smoothing factor for PRS draw tracking, 0<a<=1 (default 0.3; lower = smoother)",
    )
    parser.add_argument(
        "--busy-gpus",
        default="",
        metavar="IDX,IDX",
        help="N6: comma-separated GPU indices to mark busy for --brain job-prs in sim (no nvidia-smi)",
    )
    parser.add_argument(
        "--telemetry",
        choices=["prom", "actuator"],
        default="prom",
        help="draw source: 'prom' (Prometheus) or 'actuator' (read directly from the actuator, e.g. real NVML node without Prometheus)",
    )
    args = parser.parse_args(argv)

    try:
        busy_gpus = [int(x) for x in args.busy_gpus.split(",") if x.strip()]
    except ValueError:
        parser.error("--busy-gpus must be a comma-separated list of GPU indices")
    if busy_gpus and not args.sim:
        parser.error("--busy-gpus is a sim/demo-only override; use it with --sim")

    with open(args.config) as fh:
        topology = from_dict(json.load(fh))

    # A PowerPolicy-derived params.json (written by the operator into the domain
    # ConfigMap) overrides CLI defaults when present, so a PowerPolicy CR change
    # propagates to the controller. CLI flags remain the source for compose/sim.
    # Read once at startup: a PowerPolicy CR change takes effect on the next
    # controller (re)start, not mid-run. The operator rewrites params.json
    # immediately; picking it up live would need a watch/reload in the loop.
    params = _load_brain_params(args.config)
    cap_raise_rate = params.get("cap_raise_rate_w_per_tick", args.cap_raise_rate)
    ewma_alpha = params.get("ewma_alpha", args.ewma_alpha)

    if args.actuator == "nvml":
        try:
            from opendps.agent.nvml_agent import NvmlActuator
            actuator = NvmlActuator()
            log.info("Using NvmlActuator (real GPU caps via pynvml)")
        except Exception as exc:
            log.error("NvmlActuator init failed: %s — falling back to sim", exc)
            from opendps.sim.presets import oversub_scenario
            actuator = oversub_scenario(n_gpus=topology.total_gpu_count())
    elif args.actuator == "agent":
        from opendps.controller.agent_bridge import AgentBridgeActuator
        actuator = AgentBridgeActuator(host=args.agent_host, port=args.agent_port)
        log.info("Using AgentBridgeActuator → opendps-agent at %s:%d", args.agent_host, args.agent_port)
    else:
        from opendps.sim.presets import oversub_scenario  # local import avoids circular refs
        actuator = oversub_scenario(n_gpus=topology.total_gpu_count())
        log.info("Using SimBackend (--actuator sim)")

    cfg = ControllerConfig(
        topology=topology,
        actuator=actuator,
        prom_url=args.prom,
        interval_s=args.interval,
        dry_run=args.dry_run,
        sim_mode=args.sim,
        brain_type=args.brain,
        metrics_port=args.metrics_port,
        actuator_type=args.actuator,
        agent_host=args.agent_host,
        agent_port=args.agent_port,
        cap_raise_rate_w_per_tick=cap_raise_rate,
        ewma_alpha=ewma_alpha,
        busy_gpus=busy_gpus,
        telemetry=args.telemetry,
    )
    StandaloneController(cfg).run()
    return 0


def _domain_stats(gpu_draws: dict[int, float], caps: dict[int, float],
                  hot_threshold: float = 0.6) -> dict:
    """Brain-agnostic per-domain stats for Prometheus.

    A GPU is "idle" when draw/cap < hot_threshold. Stranded watts are the unused
    headroom (cap − draw) on idle GPUs — power statically allocated to GPUs that
    aren't using it. Identical definition for DPM, PRS, and CVXPY so the demo's
    before/after comparison is apples-to-apples.
    """
    idle_stranded = 0.0
    hot = idle = 0
    for gpu, cap in caps.items():
        draw = gpu_draws.get(gpu, 0.0)
        ratio = (draw / cap) if cap > 0 else 0.0
        if ratio < hot_threshold:
            idle += 1
            idle_stranded += max(0.0, cap - draw)
        else:
            hot += 1
    return {
        "idle_stranded_w": idle_stranded,
        "hot_count": hot,
        "idle_count": idle,
        "draw_w": sum(gpu_draws.values()),
        "cap_w": sum(caps.values()),
    }


def _load_brain_params(config_path: str) -> dict:
    """Load PowerPolicy-derived brain params from a ``params.json`` sitting next
    to the topology config (operator writes both into the domain ConfigMap).

    Returns an empty dict if absent or unreadable — callers fall back to CLI
    defaults. This is the controller side of the N5 PowerPolicy → ConfigMap →
    controller propagation path.
    """
    params_path = Path(config_path).parent / "params.json"
    try:
        with open(params_path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


if __name__ == "__main__":
    import sys

    sys.exit(main())
