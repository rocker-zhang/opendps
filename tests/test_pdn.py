"""PDN topology model tests — no GPU required."""

from __future__ import annotations

import pytest

from opendps.pdn.model import PDU, PDNTopology, PowerDomain, from_dict
from opendps.pdn.presets import demo_multi_domain, demo_single_domain

EXAMPLE_CONFIG = {
    "pdus": {"pdu-A": {"capacity_w": 10000, "derating": 0.9}},
    "domains": {
        "domain-0": {
            "budget_w": 8000,
            "gpu_indices": [0, 1, 2, 3, 4, 5, 6, 7],
            "pdu_name": "pdu-A",
            "priority": 1,
        }
    },
}


def _simple_topology() -> PDNTopology:
    pdu = PDU(name="pdu-A", capacity_w=10_000.0, derating=0.9)
    domain = PowerDomain(
        name="domain-0", budget_w=8_000.0, gpu_indices=list(range(8)), pdu_name="pdu-A"
    )
    return PDNTopology(pdus={"pdu-A": pdu}, domains={"domain-0": domain})


def test_validate_allocation_over_budget_returns_false() -> None:
    topo = _simple_topology()
    # 8 GPUs × 1100 W = 8800 W > 8000 W budget
    caps = {i: 1100.0 for i in range(8)}
    assert topo.validate_allocation("domain-0", caps) is False


def test_validate_allocation_under_budget_returns_true() -> None:
    topo = _simple_topology()
    caps = {i: 900.0 for i in range(8)}  # 7200 W < 8000 W
    assert topo.validate_allocation("domain-0", caps) is True


def test_validate_allocation_respects_pdu_derating() -> None:
    """Domain budget fits but PDU effective capacity (9 000 W) is exceeded."""
    pdu = PDU(name="pdu-A", capacity_w=10_000.0, derating=0.9)  # effective = 9000
    # Two domains on the same PDU; each budget is 4800 W (total 9600 W > 9000 W effective).
    domain_a = PowerDomain(
        name="domain-a", budget_w=4_800.0, gpu_indices=list(range(4)), pdu_name="pdu-A"
    )
    domain_b = PowerDomain(
        name="domain-b", budget_w=4_800.0, gpu_indices=list(range(4, 8)), pdu_name="pdu-A"
    )
    topo = PDNTopology(
        pdus={"pdu-A": pdu}, domains={"domain-a": domain_a, "domain-b": domain_b}
    )
    # Propose caps that fit domain-a's budget but push the PDU over effective capacity.
    caps = {i: 1200.0 for i in range(4)}  # 4800 W for domain-a; +4800 other = 9600 > 9000
    assert topo.validate_allocation("domain-a", caps) is False


def test_oversubscription_ratio_over_budget() -> None:
    topo = _simple_topology()
    caps = {i: 1100.0 for i in range(8)}  # 8800 W / 8000 W = 1.1
    ratio = topo.oversubscription_ratio("domain-0", caps)
    assert ratio == pytest.approx(1.1, rel=1e-6)
    assert ratio > 1.0


def test_oversubscription_ratio_under_budget() -> None:
    topo = _simple_topology()
    caps = {i: 800.0 for i in range(8)}  # 6400 W / 8000 W = 0.8
    assert topo.oversubscription_ratio("domain-0", caps) == pytest.approx(0.8, rel=1e-6)


def test_from_dict_round_trips_example_config() -> None:
    topo = from_dict(EXAMPLE_CONFIG)
    assert "pdu-A" in topo.pdus
    assert topo.pdus["pdu-A"].capacity_w == 10_000.0
    assert topo.pdus["pdu-A"].derating == 0.9
    assert topo.pdus["pdu-A"].effective_capacity_w == pytest.approx(9_000.0)

    assert "domain-0" in topo.domains
    d = topo.domains["domain-0"]
    assert d.budget_w == 8_000.0
    assert d.gpu_indices == list(range(8))
    assert d.pdu_name == "pdu-A"
    assert d.priority == 1


def test_demo_single_domain_structure() -> None:
    topo = demo_single_domain(10)
    assert topo.total_gpu_count() == 10
    assert topo.domain_budget_w("domain-0") == 8_000.0
    # 10 GPUs × 1000 W = 10 000 W > 8000 W budget → oversubscribed
    caps = {i: 1000.0 for i in range(10)}
    assert topo.validate_allocation("domain-0", caps) is False
    ratio = topo.oversubscription_ratio("domain-0", caps)
    assert ratio == pytest.approx(10_000.0 / 8_000.0, rel=1e-6)
    assert ratio > 1.0


def test_demo_multi_domain_structure() -> None:
    topo = demo_multi_domain(gpus_per_domain=4, n_domains=2)
    assert topo.total_gpu_count() == 8
    assert len(topo.domains) == 2
    assert len(topo.pdus) == 2
    # Each domain on its own PDU — no cross-contamination.
    for domain_name in ("domain-0", "domain-1"):
        pdu = topo.pdu_for_domain(domain_name)
        assert pdu.effective_capacity_w == pytest.approx(3_000.0, rel=1e-4)
