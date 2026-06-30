"""Rack-level power budget cascade.

A rack budget below the sum of its domains' budgets scales each domain's
effective budget to its proportional share, so the sum of GPU budgets across the
rack never exceeds the rack budget. Domains without a rack are unaffected."""
from __future__ import annotations

from opendps.pdn.model import PDU, PDNTopology, PowerDomain, Rack, from_dict


def _rack_topo(rack_budget, da=5000.0, db=5000.0):
    return PDNTopology(
        pdus={"pdu0": PDU("pdu0", 100000.0, 1.0)},  # ample PDU so the rack binds
        domains={
            "a": PowerDomain("a", da, [0, 1, 2, 3, 4], "pdu0", rack_name="r0"),
            "b": PowerDomain("b", db, [5, 6, 7, 8, 9], "pdu0", rack_name="r0"),
        },
        racks={"r0": Rack("r0", rack_budget)},
    )


def test_rack_scales_domain_budget_proportionally():
    # rack 6000 < 10000 sum -> each 5000 domain scaled to 5000*6000/10000 = 3000
    topo = _rack_topo(rack_budget=6000.0)
    assert topo.domain_budget_w("a") == 3000.0
    assert topo.domain_budget_w("b") == 3000.0
    # sum of GPU budgets across the rack equals the rack budget
    assert topo.domain_budget_w("a") + topo.domain_budget_w("b") == 6000.0


def test_rack_no_scale_when_budget_sufficient():
    # rack 12000 >= 10000 sum -> domains keep their full budget
    topo = _rack_topo(rack_budget=12000.0)
    assert topo.domain_budget_w("a") == 5000.0


def test_rack_scaling_unequal_domains():
    topo = _rack_topo(rack_budget=6000.0, da=8000.0, db=2000.0)  # sum 10000
    assert topo.domain_budget_w("a") == 8000.0 * 6000.0 / 10000.0  # 4800
    assert topo.domain_budget_w("b") == 2000.0 * 6000.0 / 10000.0  # 1200


def test_no_rack_membership_unaffected():
    topo = PDNTopology(
        pdus={"pdu0": PDU("pdu0", 100000.0, 1.0)},
        domains={"a": PowerDomain("a", 5000.0, [0, 1], "pdu0")},  # no rack
        racks={"r0": Rack("r0", 100.0)},  # tiny rack, but 'a' is not in it
    )
    assert topo.domain_budget_w("a") == 5000.0


def test_validate_allocation_respects_rack():
    topo = _rack_topo(rack_budget=6000.0)
    # domain 'a' proposing its scaled 3000 fits; peer 'b' budget 5000 -> rack
    # rollup = 3000 + 5000 = 8000 > 6000 -> rejected (conservative peer budget).
    assert topo.validate_allocation("a", {i: 600.0 for i in range(5)}) is False
    # A small allocation that fits the rack rollup passes.
    assert topo.validate_allocation("a", {i: 100.0 for i in range(5)}) is True


def test_from_dict_parses_racks_and_overhead():
    topo = from_dict({
        "pdus": {"pdu0": {"capacity_w": 100000.0, "derating": 1.0}},
        "racks": {"r0": {"budget_w": 6000.0}},
        "domains": {
            "a": {"budget_w": 5000.0, "gpu_indices": [0, 1], "pdu_name": "pdu0",
                  "rack_name": "r0", "node_overhead_w": 200.0},
        },
    })
    assert topo.racks["r0"].budget_w == 6000.0
    assert topo.domains["a"].rack_name == "r0"
    assert topo.domains["a"].node_overhead_w == 200.0


def test_from_dict_backward_compat_no_racks():
    topo = from_dict({
        "pdus": {"pdu0": {"capacity_w": 10000.0}},
        "domains": {"a": {"budget_w": 8000.0, "gpu_indices": [0, 1], "pdu_name": "pdu0"}},
    })
    assert topo.racks == {}
    assert topo.domains["a"].rack_name is None
    assert topo.domain_budget_w("a") == 8000.0


def test_controller_respects_rack_budget_end_to_end():
    """A rack-constrained topology: total caps across the rack's domains stay
    within the rack budget through the live control loop."""
    from opendps.controller.standalone import ControllerConfig, StandaloneController
    from opendps.sim.presets import oversub_scenario

    topo = _rack_topo(rack_budget=6000.0)
    cfg = ControllerConfig(
        topology=topo,
        actuator=oversub_scenario(n_gpus=10),
        sim_mode=True,
        brain_type="prs",
        metrics_port=None,
        actuator_type="sim",
    )
    ctl = StandaloneController(cfg)
    last = None
    for _ in range(6):
        last = ctl.run_once()
    total = sum(sum(d.caps.values()) for d in last)
    assert total <= 6000.0 + 1.0, f"rack budget exceeded: {total}"
