"""Cumulative energy accounting + per-tenant showback."""
from __future__ import annotations

import pytest

from opendps.telemetry.energy import EnergyAccountant


def test_add_tick_integrates_draw_times_dt():
    acc = EnergyAccountant()
    acc.add_tick({0: 100.0}, dt_s=5.0)   # 500 J
    acc.add_tick({0: 100.0}, dt_s=5.0)   # +500 J = 1000 J
    assert acc.energy_j[0] == pytest.approx(1000.0)
    assert acc.gpu_energy_wh(0) == pytest.approx(1000.0 / 3600.0)


def test_tenant_energy_aggregates_gpus():
    acc = EnergyAccountant()
    acc.add_tick({0: 50.0, 1: 50.0, 2: 100.0, 3: 100.0}, dt_s=10.0)
    # tenant-a (0,1): 100 W × 10 s = 1000 J; tenant-b (2,3): 200 W × 10 s = 2000 J
    assert acc.tenant_energy_wh([0, 1]) == pytest.approx(1000.0 / 3600.0)
    assert acc.tenant_energy_wh([2, 3]) == pytest.approx(2000.0 / 3600.0)
    assert acc.total_wh() == pytest.approx(3000.0 / 3600.0)


def test_non_positive_dt_ignored():
    acc = EnergyAccountant()
    acc.add_tick({0: 100.0}, dt_s=0.0)    # zero dt — no energy
    acc.add_tick({0: 100.0}, dt_s=-5.0)   # clock skew — ignored
    assert acc.energy_j.get(0, 0.0) == 0.0


def test_missing_gpu_contributes_zero():
    acc = EnergyAccountant()
    acc.add_tick({0: 100.0}, dt_s=5.0)
    assert acc.tenant_energy_wh([5]) == 0.0  # GPU 5 never drew


def test_none_draw_skipped():
    acc = EnergyAccountant()
    acc.add_tick({0: None, 1: 100.0}, dt_s=5.0)  # type: ignore[dict-item]
    assert 0 not in acc.energy_j
    assert acc.energy_j[1] == pytest.approx(500.0)


# --- controller showback ---


def test_controller_showback_attributes_energy_per_tenant():
    from opendps.controller.standalone import ControllerConfig, StandaloneController
    from opendps.pdn.presets import demo_single_domain
    from opendps.pdn.quota import QuotaConfig, TenantQuota
    from opendps.sim.presets import oversub_scenario

    q = QuotaConfig(domain_name="domain-0", tenants=[
        TenantQuota("tenant-a", "domain-0", [0, 1], 0.5),   # busy GPUs
        TenantQuota("tenant-b", "domain-0", [6, 7], 0.5),   # idle GPUs
    ])
    cfg = ControllerConfig(
        topology=demo_single_domain(n_gpus=10, budget_w=8000.0),
        actuator=oversub_scenario(n_gpus=10),
        sim_mode=True, brain_type="quota-prs", metrics_port=None,
        actuator_type="sim", quota_config=q, interval_s=5.0,
    )
    ctl = StandaloneController(cfg)
    for _ in range(10):
        ctl.run_once()
    sb = ctl.energy_showback()
    assert set(sb) == {"tenant-a", "tenant-b"}
    assert sb["tenant-a"] > 0.0
    # the busy tenant accrued more energy than the idle one
    assert sb["tenant-a"] > sb["tenant-b"]


def test_controller_showback_empty_without_quota():
    from opendps.controller.standalone import ControllerConfig, StandaloneController
    from opendps.pdn.presets import demo_single_domain
    from opendps.sim.presets import oversub_scenario

    cfg = ControllerConfig(
        topology=demo_single_domain(n_gpus=4, budget_w=4000.0),
        actuator=oversub_scenario(n_gpus=4),
        sim_mode=True, brain_type="prs", metrics_port=None, actuator_type="sim",
    )
    ctl = StandaloneController(cfg)
    ctl.run_once()
    assert ctl.energy_showback() == {}
