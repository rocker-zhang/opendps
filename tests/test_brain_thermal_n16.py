"""N16 — thermal-aware control.

A thermal-throttled GPU is heat-limited; its cap is backed off (derated) and the
freed watts handed to GPUs that can use them, without exceeding the budget."""
from __future__ import annotations

import time

import pytest

from opendps.brain.dpm import DomainState
from opendps.brain.thermal_prs import ThermalAwarePRSBrain
from opendps.pdn.presets import demo_single_domain

DOMAIN = "domain-0"


def _state(throttled, draws=None, cap=800.0, n=2, hw=1000.0):
    draws = draws or {i: 700.0 for i in range(n)}
    return DomainState(
        domain_name=DOMAIN,
        gpu_draws=draws,
        gpu_caps={i: cap for i in range(n)},
        gpu_max_caps={i: hw for i in range(n)},
        ts=time.time(),
        gpu_thermal_throttled={g: True for g in throttled},
    )


def _brain(budget=1600.0, n=2, **kw):
    return ThermalAwarePRSBrain(demo_single_domain(n_gpus=n, budget_w=budget), **kw)


def test_thermal_gpu_derated_below_peer():
    """An equally-loaded thermal GPU is capped below its non-throttled peer."""
    d = _brain().decide(DOMAIN, _state(throttled=[0]))
    assert d.caps[0] < d.caps[1], f"thermal GPU not derated: {d.caps}"
    assert d.caps[0] >= 200.0 - 1e-6


def test_freed_watts_go_to_non_throttled():
    """The watts freed from the thermal GPU raise a peer that has headroom."""
    base = _brain().decide(DOMAIN, _state(throttled=[]))      # no throttle -> baseline
    hot = _brain().decide(DOMAIN, _state(throttled=[0]))      # GPU0 throttled
    assert hot.caps[0] < base.caps[0]                          # GPU0 backed off
    assert hot.caps[1] > base.caps[1] - 1e-6                   # GPU1 got the freed watts


def test_no_throttle_is_passthrough():
    """With no thermal GPUs the result is exactly the PRS allocation."""
    from opendps.brain.prs import PRSBrain
    topo = demo_single_domain(n_gpus=2, budget_w=1600.0)
    st = _state(throttled=[])
    prs = PRSBrain(topo).decide(DOMAIN, st)
    thermal = ThermalAwarePRSBrain(topo).decide(DOMAIN, st)
    assert thermal.caps == prs.caps


def test_thermal_respects_floor():
    """Derate never pushes a GPU below the floor."""
    d = _brain(thermal_derate=0.9).decide(DOMAIN, _state(throttled=[0], cap=250.0))
    assert d.caps[0] >= 200.0 - 1e-6


def test_invalid_derate_rejected():
    with pytest.raises(ValueError, match="thermal_derate"):
        _brain(thermal_derate=1.0)
    with pytest.raises(ValueError, match="thermal_derate"):
        _brain(thermal_derate=-0.1)


def test_sum_within_budget():
    d = _brain().decide(DOMAIN, _state(throttled=[0]))
    assert sum(d.caps.values()) <= 1600.0 + 1.0


def test_controller_thermal_prs_end_to_end():
    from opendps.controller.standalone import ControllerConfig, StandaloneController
    from opendps.sim.presets import oversub_scenario

    topo = demo_single_domain(n_gpus=10, budget_w=8000.0)
    cfg = ControllerConfig(
        topology=topo,
        actuator=oversub_scenario(n_gpus=10),
        sim_mode=True,
        brain_type="thermal-prs",
        metrics_port=None,
        actuator_type="sim",
        thermal_throttled_gpus=[0],
    )
    ctl = StandaloneController(cfg)
    last = None
    for _ in range(5):
        last = ctl.run_once()
    caps = last[0].caps
    # GPU0 (thermal-throttled) capped below an equally-hot non-throttled peer.
    assert caps[0] < caps[1], f"thermal GPU should be derated: {caps[0]} vs {caps[1]}"
    assert sum(caps.values()) <= 8000.0 + 1.0


def test_cli_hot_gpus_requires_sim(tmp_path):
    import json

    from opendps.controller.standalone import main

    topo = tmp_path / "topo.json"
    topo.write_text(json.dumps({
        "pdus": {"p": {"name": "p", "capacity_w": 9000.0, "derating": 0.9}},
        "domains": {"domain0": {"name": "domain0", "budget_w": 8000.0,
                                "gpu_indices": [0, 1], "pdu_name": "p", "priority": 0}},
    }))
    with pytest.raises(SystemExit):  # --hot-gpus without --sim
        main(["--brain", "thermal-prs", "--config", str(topo), "--hot-gpus", "0"])


def test_throttled_gpu_never_raised_above_current():
    """If PRS would raise a throttled GPU's cap, it is derated from the current
    cap (never above it) — heat-limited GPUs are only backed off."""
    # Idle GPU whose PRS floor proposes ABOVE its low current cap, but throttled.
    state = DomainState(
        domain_name=DOMAIN,
        gpu_draws={0: 600.0, 1: 100.0},
        gpu_caps={0: 300.0, 1: 800.0},      # GPU0 currently low-capped
        gpu_max_caps={0: 1000.0, 1: 1000.0},
        ts=time.time(),
        gpu_thermal_throttled={0: True},
    )
    d = _brain(budget=2000.0).decide(DOMAIN, state)
    assert d.caps[0] <= 300.0 + 1e-6, f"throttled GPU raised above its current cap: {d.caps[0]}"


def test_throttled_gpu_below_floor_not_raised():
    """A throttled GPU already below min_cap_w must not be raised to the floor."""
    state = DomainState(
        domain_name=DOMAIN,
        gpu_draws={0: 100.0, 1: 100.0},
        gpu_caps={0: 150.0, 1: 800.0},      # GPU0 below the 200 W floor
        gpu_max_caps={0: 1000.0, 1: 1000.0},
        ts=time.time(),
        gpu_thermal_throttled={0: True},
    )
    d = _brain(budget=2000.0).decide(DOMAIN, state)
    assert d.caps[0] <= 150.0 + 1e-6, f"throttled sub-floor GPU was raised: {d.caps[0]}"


def test_cli_rejects_bad_thermal_temp(tmp_path):
    import json

    from opendps.controller.standalone import main

    topo = tmp_path / "topo.json"
    topo.write_text(json.dumps({
        "pdus": {"p": {"name": "p", "capacity_w": 9000.0, "derating": 0.9}},
        "domains": {"domain0": {"name": "domain0", "budget_w": 8000.0,
                                "gpu_indices": [0, 1], "pdu_name": "p", "priority": 0}},
    }))
    for bad in ("nan", "inf", "0", "-5"):
        with pytest.raises(SystemExit):
            main(["--sim", "--brain", "thermal-prs", "--config", str(topo),
                  "--thermal-throttle-temp-c", bad])
