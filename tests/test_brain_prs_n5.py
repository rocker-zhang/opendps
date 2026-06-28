"""N5 — failsafe hardening / transient smoothing tests for PRSBrain.

Covers the cap-raise rate limiter and the controller-side params loader that
together implement the PowerPolicy → ConfigMap → controller propagation path.
"""
from __future__ import annotations

import json
import time

import pytest

from opendps.brain.dpm import DomainState
from opendps.brain.prs import PRSBrain
from opendps.pdn.presets import demo_single_domain


def _state(draws, caps, max_caps=None, ts=None):
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
# Cap-raise rate limiter
# ---------------------------------------------------------------------------

def test_cap_raise_is_rate_limited():
    """A cap that wants to jump should rise by at most cap_raise_rate per tick."""
    topo = demo_single_domain(n_gpus=2, budget_w=2000.0)
    brain = PRSBrain(topo, cap_raise_rate_w_per_tick=50.0)
    max_c = {0: 1000.0, 1: 1000.0}

    # Tick 1: both idle and low → caps settle low.
    brain.decide("domain-0", _state({0: 100.0, 1: 100.0}, {0: 1000.0, 1: 1000.0}, max_c))
    cap0_after_first = brain._last_caps["domain-0"][0]

    # Tick 2: GPU 0 suddenly very hot — target cap wants to jump toward hw max.
    decision = brain.decide("domain-0", _state({0: 950.0, 1: 100.0}, {0: cap0_after_first, 1: 1000.0}, max_c))

    rise = decision.caps[0] - cap0_after_first
    assert rise <= 50.0 + 1e-6, f"cap rose {rise:.1f} W in one tick, limit is 50 W"
    assert rise > 0.0, "a hot GPU's cap should still rise (just slowly)"


def test_cap_lowering_is_not_rate_limited():
    """Lowering a cap (shedding power) must be immediate, regardless of limiter."""
    topo = demo_single_domain(n_gpus=2, budget_w=2000.0)
    brain = PRSBrain(topo, cap_raise_rate_w_per_tick=50.0)
    max_c = {0: 1000.0, 1: 1000.0}

    # Establish a high cap on GPU 0 (hot), then make it idle so the cap drops.
    for _ in range(5):
        brain.decide("domain-0", _state({0: 950.0, 1: 950.0}, {0: 1000.0, 1: 1000.0}, max_c))
    high_cap = brain._last_caps["domain-0"][0]
    assert high_cap > 400.0

    decision = brain.decide("domain-0", _state({0: 20.0, 1: 950.0}, {0: high_cap, 1: 1000.0}, max_c))
    # Idle GPU 0 should drop far in a single tick — not limited to 50 W/tick.
    assert decision.caps[0] < high_cap - 50.0, "cap lowering must not be rate-limited"


def test_rate_limiter_disabled_by_default():
    """With the default rate (0.0) a cap may jump freely in one tick."""
    topo = demo_single_domain(n_gpus=2, budget_w=2000.0)
    brain = PRSBrain(topo)  # cap_raise_rate_w_per_tick defaults to 0.0
    max_c = {0: 1000.0, 1: 1000.0}
    brain.decide("domain-0", _state({0: 100.0, 1: 100.0}, {0: 1000.0, 1: 1000.0}, max_c))
    prev = brain._last_caps["domain-0"][0]
    decision = brain.decide("domain-0", _state({0: 950.0, 1: 100.0}, {0: prev, 1: 1000.0}, max_c))
    assert decision.caps[0] - prev > 50.0, "no limiter → cap free to jump more than 50 W"


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_invalid_ewma_alpha_rejected(alpha):
    topo = demo_single_domain(n_gpus=2, budget_w=2000.0)
    with pytest.raises(ValueError):
        PRSBrain(topo, ewma_alpha=alpha)


def test_negative_rate_rejected():
    topo = demo_single_domain(n_gpus=2, budget_w=2000.0)
    with pytest.raises(ValueError):
        PRSBrain(topo, cap_raise_rate_w_per_tick=-1.0)


# ---------------------------------------------------------------------------
# Controller-side params loader (PowerPolicy → ConfigMap → controller)
# ---------------------------------------------------------------------------

def test_load_brain_params_reads_sibling_params_json(tmp_path):
    from opendps.controller.standalone import _load_brain_params

    (tmp_path / "topology.json").write_text("{}")
    (tmp_path / "params.json").write_text(json.dumps({
        "cap_raise_rate_w_per_tick": 75.0,
        "ewma_alpha": 0.5,
    }))
    params = _load_brain_params(str(tmp_path / "topology.json"))
    assert params["cap_raise_rate_w_per_tick"] == 75.0
    assert params["ewma_alpha"] == 0.5


def test_load_brain_params_absent_returns_empty(tmp_path):
    from opendps.controller.standalone import _load_brain_params

    (tmp_path / "topology.json").write_text("{}")
    assert _load_brain_params(str(tmp_path / "topology.json")) == {}


def test_load_brain_params_malformed_returns_empty(tmp_path):
    from opendps.controller.standalone import _load_brain_params

    (tmp_path / "topology.json").write_text("{}")
    (tmp_path / "params.json").write_text("{ not valid json")
    assert _load_brain_params(str(tmp_path / "topology.json")) == {}
