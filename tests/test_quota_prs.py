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
    return DomainState(
        domain_name=DOMAIN,
        gpu_draws=draws,
        gpu_caps={g: 1000.0 for g in draws},
        gpu_max_caps={g: 1000.0 for g in draws},
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


# --- N13 hardening: from_dict loader, config validation, edge cases ---


def test_from_dict_parses_and_defaults_tenant_domain():
    """from_dict builds a valid config and defaults each tenant's domain_name
    to the top-level domain_name when omitted."""
    cfg = QuotaConfig.from_dict({
        "domain_name": DOMAIN,
        "tenants": [
            {"tenant_id": "a", "gpu_indices": [0, 1], "max_watts_pct": 0.5},
            {"tenant_id": "b", "gpu_indices": [2, 3], "max_watts_pct": 0.5},
        ],
    })
    assert cfg.domain_name == DOMAIN
    assert [t.tenant_id for t in cfg.tenants] == ["a", "b"]
    assert all(t.domain_name == DOMAIN for t in cfg.tenants)
    assert cfg.tenants[0].gpu_indices == [0, 1]


def test_from_dict_rejects_malformed():
    """A missing required field is a loud ValueError, not a KeyError."""
    with pytest.raises(ValueError, match="malformed quota config"):
        QuotaConfig.from_dict({
            "domain_name": DOMAIN,
            "tenants": [{"tenant_id": "a", "gpu_indices": [0]}],  # no max_watts_pct
        })


def test_from_dict_enforces_validate():
    """from_dict runs validate(): over-budget input is rejected at load time."""
    with pytest.raises(ValueError, match="Total quota"):
        QuotaConfig.from_dict({
            "domain_name": DOMAIN,
            "tenants": [
                {"tenant_id": "a", "gpu_indices": [0], "max_watts_pct": 0.7},
                {"tenant_id": "b", "gpu_indices": [1], "max_watts_pct": 0.7},
            ],
        })


def test_brain_validates_quota_at_construction():
    """An over-100% quota must fail when the brain is built, not silently later."""
    topo = _topo()
    bad = QuotaConfig(domain_name=DOMAIN, tenants=[
        TenantQuota("a", DOMAIN, [0], max_watts_pct=0.8),
        TenantQuota("b", DOMAIN, [1], max_watts_pct=0.8),
    ])
    with pytest.raises(ValueError, match="Total quota"):
        QuotaAwarePRSBrain(topo, bad)


def test_brain_skips_tenant_gpu_missing_from_state():
    """A tenant GPU absent from this tick's telemetry is skipped, not crashed on."""
    topo = _topo(budget=8000.0, n=10)
    quota = QuotaConfig(domain_name=DOMAIN, tenants=[
        TenantQuota("a", DOMAIN, [0, 1, 99], max_watts_pct=0.5),  # 99 not in state
    ])
    brain = QuotaAwarePRSBrain(topo, quota)
    decision = brain.decide(DOMAIN, _state({i: 300.0 for i in range(10)}))
    assert 99 not in decision.caps
    assert 0 in decision.caps and 1 in decision.caps


def test_brain_tolerates_missing_gpu_cap_in_state():
    """gpu_caps may lag gpu_max_caps; the brain falls back to max rather than KeyError."""
    topo = _topo(budget=8000.0, n=4)
    quota = QuotaConfig(domain_name=DOMAIN, tenants=[
        TenantQuota("a", DOMAIN, [0, 1], max_watts_pct=0.5),
    ])
    brain = QuotaAwarePRSBrain(topo, quota)
    state = DomainState(
        domain_name=DOMAIN,
        gpu_draws={i: 200.0 for i in range(4)},
        gpu_caps={},  # empty — no current caps known yet
        gpu_max_caps={i: 1000.0 for i in range(4)},
        ts=time.time(),
    )
    decision = brain.decide(DOMAIN, state)  # must not raise
    assert set(decision.caps) == {0, 1, 2, 3}


# --- N13: controller loader + topology validation + end-to-end ---


def test_load_quota_config_explicit_sibling_and_absent(tmp_path):
    from opendps.controller.standalone import _load_quota_config

    payload = {
        "domain_name": DOMAIN,
        "tenants": [{"tenant_id": "a", "gpu_indices": [0, 1], "max_watts_pct": 1.0}],
    }
    import json

    # explicit path
    explicit = tmp_path / "myquota.json"
    explicit.write_text(json.dumps(payload))
    cfg = _load_quota_config(str(tmp_path / "topo.json"), str(explicit))
    assert cfg is not None and cfg.tenants[0].tenant_id == "a"

    # sibling quota.json next to the topology config
    (tmp_path / "quota.json").write_text(json.dumps(payload))
    cfg2 = _load_quota_config(str(tmp_path / "topo.json"), None)
    assert cfg2 is not None and cfg2.domain_name == DOMAIN

    # absent → None (no sibling, no explicit)
    assert _load_quota_config(str(tmp_path / "none" / "topo.json"), None) is None

    # explicit-but-missing → loud error
    with pytest.raises(ValueError, match="not found"):
        _load_quota_config(str(tmp_path / "topo.json"), str(tmp_path / "nope.json"))


def test_validate_quota_against_topology():
    from opendps.controller.standalone import _validate_quota_against_topology

    topo = _topo(n=4)
    # stray GPU not in the domain
    stray = QuotaConfig(domain_name=DOMAIN, tenants=[
        TenantQuota("a", DOMAIN, [0, 9], max_watts_pct=1.0),
    ])
    with pytest.raises(ValueError, match="not in domain"):
        _validate_quota_against_topology(stray, topo)

    # unknown domain
    wrong = QuotaConfig(domain_name="ghost", tenants=[])
    with pytest.raises(ValueError, match="not in topology"):
        _validate_quota_against_topology(wrong, topo)

    # all good
    ok = QuotaConfig(domain_name=DOMAIN, tenants=[
        TenantQuota("a", DOMAIN, [0, 1], max_watts_pct=1.0),
    ])
    _validate_quota_against_topology(ok, topo)  # no raise


def test_controller_quota_prs_end_to_end_respects_tenant_budgets():
    """Full controller tick with --brain quota-prs: an oversubscribed-but-idle
    tenant's caps stay within its slice of the domain budget."""
    from opendps.controller.standalone import ControllerConfig, StandaloneController
    from opendps.sim.presets import oversub_scenario

    topo = _topo(budget=8000.0, n=10)
    quota = QuotaConfig.from_dict({
        "domain_name": DOMAIN,
        "tenants": [
            {"tenant_id": "teamA", "gpu_indices": list(range(6)), "max_watts_pct": 0.6},
            {"tenant_id": "teamB", "gpu_indices": list(range(6, 10)), "max_watts_pct": 0.4},
        ],
    })
    cfg = ControllerConfig(
        topology=topo,
        actuator=oversub_scenario(n_gpus=10),
        sim_mode=True,
        brain_type="quota-prs",
        metrics_port=None,
        actuator_type="sim",
        quota_config=quota,
    )
    ctl = StandaloneController(cfg)
    last = None
    for _ in range(6):  # let per-tenant EWMA converge
        last = ctl.run_once()
    caps = last[0].caps
    capA = sum(caps[i] for i in range(6))
    capB = sum(caps[i] for i in range(6, 10))
    assert capA <= 8000.0 * 0.6 + 1.0, f"teamA {capA} exceeds 60% slice"
    assert capB <= 8000.0 * 0.4 + 1.0, f"teamB {capB} exceeds 40% slice"
    assert sum(caps.values()) <= 8000.0 + 1.0


def test_validate_rejects_tenant_domain_mismatch():
    """A tenant naming a different domain would be silently skipped by the
    brain, so validate() must reject it up front."""
    with pytest.raises(ValueError, match="domain_name !="):
        QuotaConfig(domain_name=DOMAIN, tenants=[
            TenantQuota("a", "other-domain", [0, 1], max_watts_pct=0.5),
        ]).validate()


def test_brain_clamps_tenant_to_budget_when_floor_would_exceed():
    """PRS's per-GPU idle floor can sum above a small slice; the brain must
    renormalise the tenant down to its budget."""
    # 10% of 8000 W = 800 W slice, but 6 idle GPUs each floor at min_cap_w
    # (200 W) would want 1200 W. The clamp must hold the tenant at <=800 W.
    topo = _topo(budget=8000.0, n=10)
    quota = QuotaConfig(domain_name=DOMAIN, tenants=[
        TenantQuota("small", DOMAIN, list(range(6)), max_watts_pct=0.1),
    ])
    brain = QuotaAwarePRSBrain(topo, quota)
    decision = brain.decide(DOMAIN, _state({i: 50.0 for i in range(10)}))  # all idle
    tenant_caps = sum(decision.caps[i] for i in range(6))
    assert tenant_caps <= 800.0 + 1.0, f"tenant exceeded its 10% slice: {tenant_caps}"
