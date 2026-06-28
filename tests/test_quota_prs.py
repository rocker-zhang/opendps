from __future__ import annotations
import time
import pytest
from opendps.brain.dpm import DomainState
from opendps.brain.quota_prs import QuotaAwarePRSBrain
from opendps.pdn.model import PDNTopology, PowerDomain
from opendps.pdn.quota import QuotaConfig, TenantQuota

DOMAIN = "dom0"


def _topo(budget: float = 8000.0, n: int = 10) -> PDNTopology:
    """Build a minimal single-domain topology with n GPUs."""
    domain = PowerDomain(
        name=DOMAIN,
        budget_w=budget,
        gpu_indices=list(range(n)),
        pdu_name="_pdu",
    )
    return PDNTopology(pdus={}, domains={DOMAIN: domain})


def _state(draws: dict[int, float]) -> DomainState:
    n = len(draws)
    return DomainState(
        domain_name=DOMAIN,
        gpu_draws=draws,
        gpu_caps={i: 1000.0 for i in range(n)},
        gpu_max_caps={i: 1000.0 for i in range(n)},
        ts=time.time(),
    )


def test_two_tenants_get_proportional_budgets():
    """60% tenant caps <= 0.6 * budget; 40% tenant caps <= 0.4 * budget."""
    topo = _topo(budget=8000.0, n=10)
    quota = QuotaConfig(domain_name=DOMAIN, tenants=[
        TenantQuota("teamA", DOMAIN, list(range(6)), max_watts_pct=0.6),
        TenantQuota("teamB", DOMAIN, list(range(6, 10)), max_watts_pct=0.4),
    ])
    brain = QuotaAwarePRSBrain(topo, quota)
    draws = {i: 700.0 for i in range(10)}
    decision = brain.decide(DOMAIN, _state(draws))

    capA = sum(decision.caps[i] for i in range(6))
    capB = sum(decision.caps[i] for i in range(6, 10))
    assert capA <= 8000.0 * 0.6 + 0.1
    assert capB <= 8000.0 * 0.4 + 0.1


def test_total_caps_within_domain_budget():
    """Sum of all caps must not exceed domain budget."""
    topo = _topo(budget=8000.0, n=10)
    quota = QuotaConfig(domain_name=DOMAIN, tenants=[
        TenantQuota("t1", DOMAIN, list(range(5)), max_watts_pct=0.5),
        TenantQuota("t2", DOMAIN, list(range(5, 10)), max_watts_pct=0.5),
    ])
    brain = QuotaAwarePRSBrain(topo, quota)
    draws = {i: 900.0 if i < 5 else 100.0 for i in range(10)}
    decision = brain.decide(DOMAIN, _state(draws))
    assert sum(decision.caps.values()) <= 8000.0 + 1.0


def test_unassigned_gpus_get_remaining_budget():
    """GPUs not in any tenant get equal share of remaining budget."""
    topo = _topo(budget=8000.0, n=10)
    quota = QuotaConfig(domain_name=DOMAIN, tenants=[
        TenantQuota("t1", DOMAIN, [0, 1, 2], max_watts_pct=0.3),
    ])
    brain = QuotaAwarePRSBrain(topo, quota)
    draws = {i: 200.0 for i in range(10)}
    decision = brain.decide(DOMAIN, _state(draws))
    # GPUs 3-9 (7 GPUs) should each get a share of remaining ~5600W budget
    unassigned_caps = [decision.caps[i] for i in range(3, 10)]
    # Each should be positive and <= 1000 (gpu max)
    assert all(c > 0 for c in unassigned_caps)
    assert all(c <= 1000.0 + 1.0 for c in unassigned_caps)


def test_quota_validation_rejects_over_100pct():
    with pytest.raises(ValueError, match="Total quota"):
        q = QuotaConfig(domain_name=DOMAIN, tenants=[
            TenantQuota("t1", DOMAIN, [0, 1], max_watts_pct=0.7),
            TenantQuota("t2", DOMAIN, [2, 3], max_watts_pct=0.7),
        ])
        q.validate()


def test_quota_validation_rejects_overlapping_gpus():
    with pytest.raises(ValueError, match="assigned to multiple"):
        q = QuotaConfig(domain_name=DOMAIN, tenants=[
            TenantQuota("t1", DOMAIN, [0, 1, 2], max_watts_pct=0.5),
            TenantQuota("t2", DOMAIN, [2, 3, 4], max_watts_pct=0.4),
        ])
        q.validate()
