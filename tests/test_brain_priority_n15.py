"""N15 — SLA-tiered priority preemption.

Under power contention, higher-tier GPUs keep more cap than equally-loaded
lower-tier GPUs, no GPU is starved below the floor, and the domain budget is
never exceeded."""
from __future__ import annotations

import time

import pytest

from opendps.brain.dpm import DomainState
from opendps.brain.priority_prs import TIER_WEIGHTS, PriorityTieredPRSBrain
from opendps.pdn.presets import demo_single_domain

DOMAIN = "domain-0"


def _contended_state(n=4, draw=500.0, cap=600.0):
    return DomainState(
        domain_name=DOMAIN,
        gpu_draws={i: draw for i in range(n)},
        gpu_caps={i: cap for i in range(n)},     # draw/cap = 0.83 >= 0.6 -> contended
        gpu_max_caps={i: 1000.0 for i in range(n)},
        ts=time.time(),
    )


def _brain(tiers, budget=2000.0, n=4):
    return PriorityTieredPRSBrain(demo_single_domain(n_gpus=n, budget_w=budget), tiers)


def test_tiers_order_caps_under_contention():
    """Equal load, 4 contended GPUs: critical > normal > low, all >= floor, Σ<=budget."""
    brain = _brain({0: "critical", 1: "low", 2: "normal", 3: "normal"})
    d = brain.decide(DOMAIN, _contended_state())
    assert d.caps[0] > d.caps[2] > d.caps[1], f"tier order violated: {d.caps}"
    assert all(c >= 200.0 - 1e-6 for c in d.caps.values()), f"floor violated: {d.caps}"
    assert sum(d.caps.values()) <= 2000.0 + 1.0


def test_all_same_tier_is_balanced():
    """With one uniform tier, equally-loaded GPUs get equal caps (no bias)."""
    brain = _brain({i: "normal" for i in range(4)})
    d = brain.decide(DOMAIN, _contended_state())
    vals = list(d.caps.values())
    assert max(vals) - min(vals) < 1.0, f"should be balanced: {d.caps}"


def test_unmapped_gpu_defaults_to_normal():
    """An unmapped GPU is treated as 'normal'; a critical peer outranks it."""
    brain = _brain({0: "critical"})  # 1,2,3 unmapped -> normal
    d = brain.decide(DOMAIN, _contended_state())
    assert d.caps[0] > d.caps[1]
    assert abs(d.caps[1] - d.caps[2]) < 1.0  # the two unmapped are equal


def test_unknown_tier_rejected():
    with pytest.raises(ValueError, match="unknown priority tier"):
        _brain({0: "platinum"})


def test_idle_high_tier_gpu_not_boosted():
    """A high-tier but idle GPU keeps its PRS floor — tier does not boost idle."""
    brain = _brain({0: "critical", 1: "critical", 2: "normal", 3: "normal"})
    state = DomainState(
        domain_name=DOMAIN,
        gpu_draws={0: 500.0, 1: 500.0, 2: 50.0, 3: 50.0},  # 2,3 idle (low ratio)
        gpu_caps={i: 600.0 for i in range(4)},
        gpu_max_caps={i: 1000.0 for i in range(4)},
        ts=time.time(),
    )
    d = brain.decide(DOMAIN, state)
    # idle normal GPUs stay near the floor, well below the busy critical ones
    assert d.caps[2] < d.caps[0] and d.caps[3] < d.caps[1]
    assert sum(d.caps.values()) <= 2000.0 + 1.0


def test_tier_weights_are_monotonic():
    assert TIER_WEIGHTS["low"] < TIER_WEIGHTS["normal"] < TIER_WEIGHTS["high"] < TIER_WEIGHTS["critical"]


# --- controller / CLI ---


def test_controller_priority_prs_end_to_end():
    from opendps.controller.standalone import ControllerConfig, StandaloneController
    from opendps.sim.presets import oversub_scenario

    topo = demo_single_domain(n_gpus=10, budget_w=3600.0)
    cfg = ControllerConfig(
        topology=topo,
        actuator=oversub_scenario(n_gpus=10),
        sim_mode=True,
        brain_type="priority-prs",
        metrics_port=None,
        actuator_type="sim",
        gpu_priority_tiers={0: "critical", 1: "low"},
    )
    ctl = StandaloneController(cfg)
    last = None
    for _ in range(5):
        last = ctl.run_once()
    caps = last[0].caps
    assert caps[0] > caps[1], f"critical GPU should outrank low: {caps[0]} vs {caps[1]}"
    assert sum(caps.values()) <= 3600.0 + 1.0


def test_cli_priority_prs_requires_tiers(tmp_path):
    import json

    from opendps.controller.standalone import main

    topo = tmp_path / "topo.json"
    topo.write_text(json.dumps({
        "pdus": {"p": {"name": "p", "capacity_w": 9000.0, "derating": 0.9}},
        "domains": {"domain0": {"name": "domain0", "budget_w": 8000.0,
                                "gpu_indices": [0, 1], "pdu_name": "p", "priority": 0}},
    }))
    with pytest.raises(SystemExit):
        main(["--sim", "--brain", "priority-prs", "--config", str(topo)])
    # tiers with a non-priority brain is also a clean error
    with pytest.raises(SystemExit):
        main(["--sim", "--brain", "prs", "--config", str(topo),
              "--gpu-priority-tiers", '{"0":"high"}'])


def test_never_oversubscribes_when_floors_infeasible():
    """Budget below the sum of floors (misconfig): caps scale down so Σ<=budget."""
    # 4 contended GPUs, floor 200 -> 800 of floor, but budget only 500.
    brain = _brain({0: "critical", 1: "low", 2: "normal", 3: "normal"}, budget=500.0)
    d = brain.decide(DOMAIN, _contended_state())
    assert sum(d.caps.values()) <= 500.0 + 1e-6, f"oversubscribed: {d.caps}"
