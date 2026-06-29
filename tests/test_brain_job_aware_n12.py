"""N12 — config-driven job-aware priority boost.

The boost magnitude is now configurable (CLI --priority-boost / params.json /
ControllerConfig) instead of a hardcoded 0.15. These tests lock the configured
value into the brain, the zero-boost no-op, range validation, and the end-to-end
controller path (a busy GPU outdraws an equally-loaded idle-job GPU)."""
from __future__ import annotations

import time

import pytest

from opendps.agent.job_tracker import JobTracker
from opendps.brain.dpm import DomainState
from opendps.brain.job_aware_prs import JobAwarePRSBrain
from opendps.pdn.presets import demo_single_domain


def _busy_state():
    # Symmetric load so PRS alone caps both GPUs identically; tight budget so the
    # caps sit below hw max and the boost is observable.
    return DomainState(
        domain_name="domain-0",
        gpu_draws={0: 500.0, 1: 500.0},
        gpu_caps={0: 600.0, 1: 600.0},
        gpu_max_caps={0: 1000.0, 1: 1000.0},
        ts=time.time(),
    )


def _brain(boost: float) -> JobAwarePRSBrain:
    topo = demo_single_domain(n_gpus=2, budget_w=1200.0)
    tracker = JobTracker()
    tracker.set_busy_gpus([0])
    return JobAwarePRSBrain(topo, tracker, priority_boost=boost)


def test_configured_boost_scales_first_tick_exactly():
    """At the first tick the busy GPU's cap is the PRS cap times (1 + boost),
    before renormalisation pulls everything back under budget — a larger boost
    yields a larger busy/idle ratio."""
    small = _brain(0.10).decide("domain-0", _busy_state())
    large = _brain(0.50).decide("domain-0", _busy_state())
    r_small = small.caps[0] / small.caps[1]
    r_large = large.caps[0] / large.caps[1]
    assert r_large > r_small > 1.0, f"boost should scale the ratio: {r_small=} {r_large=}"


def test_zero_boost_is_noop():
    """priority_boost=0 must leave the busy GPU no higher than the idle one
    (identical load -> identical caps)."""
    d = _brain(0.0).decide("domain-0", _busy_state())
    assert d.caps[0] == pytest.approx(d.caps[1], rel=1e-6)


def test_negative_boost_rejected():
    topo = demo_single_domain(n_gpus=2, budget_w=1200.0)
    with pytest.raises(ValueError, match="priority_boost must be >= 0"):
        JobAwarePRSBrain(topo, JobTracker(), priority_boost=-0.1)


def test_controller_priority_boost_flows_to_brain():
    """ControllerConfig.priority_boost reaches the constructed JobAwarePRSBrain,
    and a busy GPU ends up capped above an equally-loaded no-job GPU."""
    import json
    from pathlib import Path

    from opendps.controller.standalone import ControllerConfig, StandaloneController
    from opendps.pdn.model import from_dict
    from opendps.sim.presets import oversub_scenario

    topo_path = Path(__file__).parents[1] / "deploy" / "topology-jobdemo.json"
    with open(topo_path) as fh:
        topo = from_dict(json.load(fh))
    cfg = ControllerConfig(
        topology=topo,
        actuator=oversub_scenario(n_gpus=10),
        sim_mode=True,
        brain_type="job-prs",
        metrics_port=None,
        actuator_type="sim",
        busy_gpus=[0, 1],
        priority_boost=0.30,
    )
    ctl = StandaloneController(cfg)
    assert ctl._brain._boost == 0.30
    last = None
    for _ in range(5):
        last = ctl.run_once()
    caps = last[0].caps
    boosted = (caps[0] + caps[1]) / 2
    plain = (caps[2] + caps[3] + caps[4] + caps[5]) / 4
    assert boosted > plain * 1.1, f"busy GPUs should outdraw equally-loaded ones: {boosted=} {plain=}"


def test_cli_rejects_negative_priority_boost(tmp_path):
    import json

    from opendps.controller.standalone import main

    topo = tmp_path / "topo.json"
    topo.write_text(json.dumps({
        "pdus": {"p": {"name": "p", "capacity_w": 9000.0, "derating": 0.9}},
        "domains": {"domain0": {"name": "domain0", "budget_w": 8000.0,
                                "gpu_indices": [0, 1], "pdu_name": "p", "priority": 0}},
    }))
    with pytest.raises(SystemExit):
        main(["--sim", "--brain", "job-prs", "--config", str(topo), "--priority-boost", "-0.2"])


def test_params_json_negative_priority_boost_rejected(tmp_path):
    """The operator production path writes priority_boost into params.json; a
    negative value there must also be rejected (same guard as the CLI flag)."""
    import json

    from opendps.controller.standalone import main

    topo = tmp_path / "topo.json"
    topo.write_text(json.dumps({
        "pdus": {"p": {"name": "p", "capacity_w": 9000.0, "derating": 0.9}},
        "domains": {"domain0": {"name": "domain0", "budget_w": 8000.0,
                                "gpu_indices": [0, 1], "pdu_name": "p", "priority": 0}},
    }))
    (tmp_path / "params.json").write_text(json.dumps({"priority_boost": -0.5}))
    with pytest.raises(SystemExit):
        main(["--sim", "--brain", "job-prs", "--config", str(topo)])


def test_stale_params_priority_boost_ignored_for_non_job_brain(tmp_path, monkeypatch):
    """A leftover (even negative) priority_boost in params.json must NOT break a
    non-job-prs brain that never reads it. The run is stopped after one tick."""
    import json

    from opendps.controller.standalone import StandaloneController, main

    topo = tmp_path / "topo.json"
    topo.write_text(json.dumps({
        "pdus": {"p": {"name": "p", "capacity_w": 9000.0, "derating": 0.9}},
        "domains": {"domain0": {"name": "domain0", "budget_w": 8000.0,
                                "gpu_indices": [0, 1], "pdu_name": "p", "priority": 0}},
    }))
    (tmp_path / "params.json").write_text(json.dumps({"priority_boost": -0.5}))

    # Stop the controller after a single tick instead of looping forever.
    calls = {"n": 0}
    real_run_once = StandaloneController.run_once

    def _one_tick(self):
        calls["n"] += 1
        real_run_once(self)
        raise KeyboardInterrupt
    monkeypatch.setattr(StandaloneController, "run_once", _one_tick)

    try:
        main(["--sim", "--brain", "prs", "--config", str(topo), "--interval", "0.01"])
    except KeyboardInterrupt:
        pass
    assert calls["n"] == 1  # reached the loop without a parser.error on the stale value


def test_priority_boost_rejected_for_non_job_brain(tmp_path):
    """--priority-boost is only meaningful for job-prs; passing it with another
    brain is a clean CLI error, not a silently-ignored flag."""
    import json

    from opendps.controller.standalone import main

    topo = tmp_path / "topo.json"
    topo.write_text(json.dumps({
        "pdus": {"p": {"name": "p", "capacity_w": 9000.0, "derating": 0.9}},
        "domains": {"domain0": {"name": "domain0", "budget_w": 8000.0,
                                "gpu_indices": [0, 1], "pdu_name": "p", "priority": 0}},
    }))
    with pytest.raises(SystemExit):
        main(["--sim", "--brain", "prs", "--config", str(topo), "--priority-boost", "0.3"])
