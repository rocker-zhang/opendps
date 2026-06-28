from __future__ import annotations
import time
from opendps.brain.cvxpy_brain import CVXPYBrain
from opendps.brain.dpm import DomainState
from opendps.pdn.presets import demo_single_domain


DOMAIN = "domain-0"


def _state(draws: dict[int, float], caps: dict[int, float] | None = None) -> DomainState:
    n = len(draws)
    max_caps = {i: 1000.0 for i in range(n)}
    return DomainState(
        domain_name=DOMAIN,
        gpu_draws=draws,
        gpu_caps=caps or {i: 1000.0 for i in range(n)},
        gpu_max_caps=max_caps,
        ts=time.time(),
    )


def test_cvxpy_respects_budget():
    topo = demo_single_domain(n_gpus=10, budget_w=8000.0)
    brain = CVXPYBrain(topo)
    draws = {i: 700.0 if i < 6 else 100.0 for i in range(10)}
    decision = brain.decide(DOMAIN, _state(draws))
    assert sum(decision.caps.values()) <= 8000.0 + 0.5  # allow tiny solver tolerance


def test_cvxpy_min_cap_floor():
    topo = demo_single_domain(n_gpus=10, budget_w=8000.0)
    brain = CVXPYBrain(topo, min_cap_w=200.0)
    draws = {i: 50.0 for i in range(10)}
    decision = brain.decide(DOMAIN, _state(draws))
    assert all(c >= 199.9 for c in decision.caps.values())


def test_cvxpy_fallback_when_cvxpy_missing(monkeypatch):
    """When cvxpy is unavailable, should silently fall back to PRS."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "cvxpy":
            raise ImportError("mocked missing cvxpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    topo = demo_single_domain(n_gpus=10, budget_w=8000.0)
    brain = CVXPYBrain(topo)
    draws = {i: 700.0 for i in range(10)}
    decision = brain.decide(DOMAIN, _state(draws))
    # Should return a valid decision (PRS fallback)
    assert len(decision.caps) == 10
    assert all(c > 0 for c in decision.caps.values())
