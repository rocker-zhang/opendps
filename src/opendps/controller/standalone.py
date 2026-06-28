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
            self._brain: DPMBrain | PRSBrain = PRSBrain(config.topology)
        elif config.brain_type == "cvxpy":
            from opendps.brain.cvxpy_brain import CVXPYBrain
            from typing import Any
            self._brain: Any = CVXPYBrain(config.topology)
        elif config.brain_type == "job-prs":
            from opendps.agent.job_tracker import JobTracker
            from opendps.brain.job_aware_prs import JobAwarePRSBrain
            from typing import Any
            tracker = JobTracker()
            tracker.start()
            self._brain: Any = JobAwarePRSBrain(config.topology, tracker)
        else:
            self._brain = DPMBrain(config.topology)
        # Only create a PromClient when we will actually use it.
        self._client: PromClient | None = (
            None if config.sim_mode else PromClient(config.prom_url)
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
        if self._config.sim_mode:
            # Closed-loop sim: read telemetry directly from the actuator.
            ts = time.time()
            gpu_by_index = None  # not used in sim mode
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
                if self._config.sim_mode:
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
                for gpu_idx, cap_w in decision.caps.items():
                    self._config.actuator.set_power_cap(gpu_idx, cap_w)
            else:
                log.info("[dry-run] would push caps for domain=%s: %s", domain_name, decision.caps)

            # Update Prometheus metrics if a port was configured.
            if self._config.metrics_port is not None:
                prs_m = (
                    self._brain.get_last_metrics(domain_name)
                    if isinstance(self._brain, PRSBrain)
                    else None
                )
                update_decision_metrics(
                    domain_name,
                    prs_active=isinstance(self._brain, PRSBrain),
                    domain_draw_w=prs_m.domain_draw_w if prs_m else sum(gpu_draws.values()),
                    domain_cap_w=prs_m.domain_cap_w if prs_m else sum(decision.caps.values()),
                    idle_stranded_w=prs_m.idle_stranded_w if prs_m else 0.0,
                    hot_count=len(prs_m.hot_gpus) if prs_m else 0,
                    idle_count=len(prs_m.idle_gpus) if prs_m else 0,
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
        choices=["dpm", "prs", "cvxpy", "job-prs"],
        default="prs",
        help="Brain algorithm: dpm = static proportional (v1), prs = EWMA reclaim (v2, default), cvxpy = LP solver (v3)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        metavar="PORT",
        help="Start Prometheus /metrics HTTP server on this port (e.g. 9402)",
    )
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        topology = from_dict(json.load(fh))

    # N1: SimBackend is the only available actuator.  N3 will add the real agent.
    from opendps.sim.presets import oversub_scenario  # local import avoids circular refs
    actuator = oversub_scenario(n_gpus=topology.total_gpu_count())

    cfg = ControllerConfig(
        topology=topology,
        actuator=actuator,
        prom_url=args.prom,
        interval_s=args.interval,
        dry_run=args.dry_run,
        sim_mode=args.sim,
        brain_type=args.brain,
        metrics_port=args.metrics_port,
    )
    StandaloneController(cfg).run()
    return 0
