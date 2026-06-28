"""opendps-agent CLI entry point.

Node-local GPU power enforcement agent:
  - Uses NvmlActuator for real GPU cap push (or SimBackend with --sim)
  - Starts FailsafeLoop in background for brain-independent overload protection
  - Runs StandaloneController for the brain control loop

Usage
-----
    opendps-agent --config topology.json [--brain prs|dpm] [--interval 5]
                  [--metrics-port 9403] [--failsafe-threshold 1050]
                  [--failsafe-emergency-cap 800] [--failsafe-poll 0.1]
                  [--sim] [--dry-run]

Real hardware mode (requires NVML on the node):
    opendps-agent --config /etc/opendps/topology.json --metrics-port 9403

Sim mode (no GPU needed — same as running opendps-controller --sim):
    opendps-agent --config topology.json --sim
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys

from opendps.agent.failsafe import FailsafeLoop
from opendps.controller.standalone import ControllerConfig, StandaloneController
from opendps.pdn.model import from_dict


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        prog="opendps-agent",
        description="opendps node agent — NVML cap enforcement + failsafe loop",
    )
    parser.add_argument("--config", required=True, metavar="FILE",
                        help="PDN topology JSON file")
    parser.add_argument("--brain", choices=["dpm", "prs"], default="prs",
                        help="Brain algorithm (default: prs)")
    parser.add_argument("--prom", default="http://localhost:9090", metavar="URL",
                        help="Prometheus base URL (prom mode only)")
    parser.add_argument("--interval", type=float, default=5.0, metavar="SECONDS",
                        help="Control loop interval in seconds (default: 5)")
    parser.add_argument("--metrics-port", type=int, default=None, metavar="PORT",
                        help="Prometheus /metrics port (e.g. 9403)")
    parser.add_argument("--failsafe-threshold", type=float, default=1050.0,
                        metavar="WATTS",
                        help="Emergency draw threshold per GPU (default: 1050 W)")
    parser.add_argument("--failsafe-emergency-cap", type=float, default=800.0,
                        metavar="WATTS",
                        help="Cap applied on failsafe trip (default: 800 W)")
    parser.add_argument("--failsafe-poll", type=float, default=0.1,
                        metavar="SECONDS",
                        help="Failsafe poll interval in seconds (default: 0.1)")
    parser.add_argument("--sim", action="store_true",
                        help="Use SimBackend instead of real NVML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log decisions without pushing caps")
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        topology = from_dict(json.load(fh))

    if args.sim:
        from opendps.sim.presets import oversub_scenario
        actuator = oversub_scenario(n_gpus=topology.total_gpu_count())
        sim_mode = True
        log.info("Agent running in sim mode (%d GPUs)", topology.total_gpu_count())
    else:
        try:
            from opendps.agent.nvml_agent import NvmlActuator
            actuator = NvmlActuator()
            sim_mode = False
            log.info(
                "Agent running with NVML (%d real GPUs)",
                actuator.gpu_count(),
            )
        except Exception as exc:
            log.error(
                "NVML init failed (%s). Use --sim for simulation mode.", exc
            )
            return 1

    # Failsafe: cap-lower-only background thread
    failsafe = FailsafeLoop(
        actuator=actuator,
        emergency_threshold_w=args.failsafe_threshold,
        emergency_cap_w=args.failsafe_emergency_cap,
        poll_interval_s=args.failsafe_poll,
    )
    failsafe.start()
    log.info(
        "Failsafe started: threshold=%.0fW emergency_cap=%.0fW poll=%.2fs",
        args.failsafe_threshold, args.failsafe_emergency_cap, args.failsafe_poll,
    )

    cfg = ControllerConfig(
        topology=topology,
        actuator=actuator,
        prom_url=args.prom,
        interval_s=args.interval,
        dry_run=args.dry_run,
        sim_mode=sim_mode,
        brain_type=args.brain,
        metrics_port=args.metrics_port,
    )

    def _shutdown(signum, frame):  # noqa: ARG001
        log.info("Received signal %d, shutting down", signum)
        failsafe.stop()
        if not args.sim:
            try:
                actuator.shutdown()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        StandaloneController(cfg).run()
    finally:
        failsafe.stop()
        log.info("Failsafe trips: %d", failsafe.trip_count)
        if not args.sim:
            try:
                actuator.shutdown()
            except Exception:
                pass

    return 0
