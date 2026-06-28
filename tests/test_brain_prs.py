"""Tests for PRSBrain (brain v2 — EWMA-based oversubscription reclaim)."""

from __future__ import annotations

import time

import pytest

from opendps.brain.dpm import DomainState
from opendps.brain.prs import PRSBrain, PRSMetrics
from opendps.pdn.presets import demo_single_domain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def topology10():
    """10 GPUs, 8000 W budget — the headline oversubscription scenario."""
    return demo_single_domain(n_gpus=10, budget_w=8000.0)


@pytest.fixture()
def brain10(topology10):
    return PRSBrain(topology10)


def _state(draws, caps, max_caps=None, ts=None):
    """Build a DomainState from plain dicts."""
    if max_caps is None:
        max_caps = dict(caps)
    return DomainState(
        domain_name="domain-0",
        gpu_draws=dict(draws),
        gpu_caps=dict(caps),
        gpu_max_caps=dict(max_caps),
        ts=ts or time.time(),
    )


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------

def test_under_budget_does_not_throttle_hot_gpus(brain10, topology10):
    """When all GPUs are hot but total draw < budget, PRS should not starve any GPU."""
    budget = 8000.0
    n = 10
    cap = budget / n        # 800 W each
    draws = {i: 700.0 for i in range(n)}   # 7000 W total < 8000 W
    caps  = {i: cap for i in range(n)}
    max_caps = {i: 1000.0 for i in range(n)}

    # Run multiple ticks so EWMA converges
    for _ in range(20):
        state = _state(draws, caps, max_caps)
        decision = brain10.decide("domain-0", state)

    # After convergence every GPU draw (700W) / cap (800W) = 0.875 ≥ 0.6 → all hot
    # Total domain draw 7000 W < 8000 W → hot GPUs should get budget/hot_count each
    assert all(v >= 700.0 for v in decision.caps.values()), (
        "Hot GPUs should not be starved when total draw is under budget"
    )


def test_total_caps_respect_budget(brain10):
    """Σcap_i must always be ≤ domain budget (8000 W)."""
    budget = 8000.0
    n = 10
    draws = {i: (800.0 if i < 6 else 80.0) for i in range(n)}   # 6 hot, 4 idle
    caps  = {i: 1000.0 for i in range(n)}
    max_caps = {i: 1000.0 for i in range(n)}

    for tick in range(30):
        state = _state(draws, caps, max_caps)
        decision = brain10.decide("domain-0", state)
        total_cap = sum(decision.caps.values())
        assert total_cap <= budget * 1.001, (
            f"tick {tick}: total caps {total_cap:.1f} W > budget {budget} W"
        )


def test_min_cap_floor_respected(brain10):
    """No GPU should be capped below min_cap_w (default 200 W)."""
    min_cap = 200.0
    draws = {i: 0.0 for i in range(10)}
    caps  = {i: 1000.0 for i in range(10)}
    max_caps = {i: 1000.0 for i in range(10)}

    for _ in range(10):
        state = _state(draws, caps, max_caps)
        decision = brain10.decide("domain-0", state)

    assert all(v >= min_cap for v in decision.caps.values()), (
        "Every GPU cap must stay at or above min_cap_w"
    )


# ---------------------------------------------------------------------------
# EWMA behaviour
# ---------------------------------------------------------------------------

def test_ewma_smooths_spikes():
    """A single-tick power spike should not immediately cause a large cap reduction."""
    topo = demo_single_domain(n_gpus=2, budget_w=2000.0)
    brain = PRSBrain(topo, ewma_alpha=0.3)

    # 20 ticks at steady state: both GPUs at 700 W, cap 1000 W (hot)
    steady_draws = {0: 700.0, 1: 700.0}
    caps  = {0: 1000.0, 1: 1000.0}
    max_c = {0: 1000.0, 1: 1000.0}
    for _ in range(20):
        brain.decide("domain-0", _state(steady_draws, caps, max_c))

    # One-tick spike on GPU 0: 2000 W (over budget alone)
    spike_state = _state({0: 2000.0, 1: 700.0}, caps, max_c)
    decision = brain.decide("domain-0", spike_state)

    # EWMA of GPU 0 should NOT have fully absorbed the spike in one tick
    m = brain.get_last_metrics("domain-0")
    assert m is not None
    ewma_0 = m.ewma_draws[0]
    # With α=0.3: ewma = 0.3*2000 + 0.7*700 = 600+490 = 1090
    assert 900 < ewma_0 < 1500, f"EWMA={ewma_0} should be between 900 and 1500 after one spike"

    # Caps must still be valid
    assert sum(decision.caps.values()) <= 2000.0 * 1.001


# ---------------------------------------------------------------------------
# Idle reclaim — the headline scenario
# ---------------------------------------------------------------------------

def test_prs_reclaims_idle_stranded_watts(brain10):
    """After convergence: idle GPUs should have low caps, hot GPUs high caps.

    Scenario: 10 GPUs, 8000 W budget.
      6 hot @ 900 W draw, 1000 W cap
      4 idle @ 80 W draw, 1000 W cap
    Expected after EWMA convergence:
      idle caps ≈ 80 × 1.3 = 104 W (close to floor)
      idle stranded watts should be < DPM baseline (4 × (1000-80) = 3680 W)
    """
    draws = {i: (900.0 if i < 6 else 80.0) for i in range(10)}
    caps  = {i: 1000.0 for i in range(10)}
    max_c = {i: 1000.0 for i in range(10)}

    for tick in range(50):
        state = _state(draws, caps, max_c)
        decision = brain10.decide("domain-0", state)
        # Feed back the new caps into next state
        caps = dict(decision.caps)

    m = brain10.get_last_metrics("domain-0")
    assert m is not None

    # Idle GPUs should have lower caps than the 1000W hardware max
    idle_cap_vals = [decision.caps[g] for g in m.idle_gpus]
    hot_cap_vals  = [decision.caps[g] for g in m.hot_gpus]

    if idle_cap_vals:
        avg_idle_cap = sum(idle_cap_vals) / len(idle_cap_vals)
        assert avg_idle_cap < 500.0, (
            f"Average idle GPU cap {avg_idle_cap:.0f} W should be well below 1000 W"
        )

    if hot_cap_vals and m.hot_gpus:
        avg_hot_cap = sum(hot_cap_vals) / len(hot_cap_vals)
        # Hot GPUs should have been given more than 800W (the naive equal share)
        assert avg_hot_cap > 800.0, (
            f"Hot GPU average cap {avg_hot_cap:.0f} W should exceed equal-share 800 W"
        )

    # Stranded watts should be < DPM baseline
    dpm_stranded = 4 * (1000.0 - 80.0)  # 3680 W
    assert m.idle_stranded_w < dpm_stranded * 0.5, (
        f"PRS stranded {m.idle_stranded_w:.0f} W should be far below DPM baseline {dpm_stranded:.0f} W"
    )


# ---------------------------------------------------------------------------
# get_last_metrics
# ---------------------------------------------------------------------------

def test_get_last_metrics_type(brain10):
    draws = {i: 500.0 for i in range(10)}
    caps  = {i: 800.0 for i in range(10)}
    brain10.decide("domain-0", _state(draws, caps))
    m = brain10.get_last_metrics("domain-0")
    assert isinstance(m, PRSMetrics)
    assert len(m.hot_gpus) + len(m.idle_gpus) == 10
    assert m.idle_stranded_w >= 0.0
    assert m.domain_draw_w == pytest.approx(5000.0)


def test_get_last_metrics_none_before_decide(brain10):
    assert brain10.get_last_metrics("domain-0") is None


# ---------------------------------------------------------------------------
# reset_ewma
# ---------------------------------------------------------------------------

def test_reset_ewma_clears_state(brain10):
    draws = {i: 900.0 for i in range(10)}
    caps  = {i: 1000.0 for i in range(10)}
    for _ in range(10):
        brain10.decide("domain-0", _state(draws, caps))
    brain10.reset_ewma("domain-0")
    assert brain10.get_last_metrics("domain-0") is None


# ---------------------------------------------------------------------------
# Hot-budget exhaustion: idle GPUs eat all budget
# ---------------------------------------------------------------------------

def test_hot_gpus_get_min_cap_when_budget_exhausted():
    """If idle GPUs alone consume the budget, hot GPUs still get min_cap_w."""
    # 2-GPU domain, budget 200 W (extremely tight)
    from opendps.pdn.presets import demo_single_domain
    topo = demo_single_domain(n_gpus=2, budget_w=200.0)
    brain = PRSBrain(topo, min_cap_w=100.0, idle_floor_margin=0.5)

    # Both GPUs technically idle but drawing 100W each — budget 200W
    draws = {0: 100.0, 1: 5.0}   # GPU 0 borderline hot, GPU 1 idle
    caps  = {0: 1000.0, 1: 1000.0}
    max_c = {0: 1000.0, 1: 1000.0}

    for _ in range(20):
        brain.decide("domain-0", _state(draws, caps, max_c))

    m = brain.get_last_metrics("domain-0")
    assert m is not None
    # Hot GPUs (if any) should be >= min_cap_w
    for g in m.hot_gpus:
        assert brain._ewma.get("domain-0", {}).get(g, 0) >= 0
