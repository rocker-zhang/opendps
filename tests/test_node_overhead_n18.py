"""N18 — node overhead is subtracted from the GPU power budget.

`available_gpu_budget_w` existed in the model but every brain allocated against
the raw `budget_w` (the subtraction was dead code). These tests lock in that the
control path now sizes GPU caps against budget-minus-overhead, and that
validate_allocation and the PRS brain both respect it."""
from __future__ import annotations

import time

from opendps.brain.dpm import DomainState
from opendps.brain.prs import PRSBrain
from opendps.pdn.model import PDU, PDNTopology, PowerDomain

DOMAIN = "dom0"


def _topo(budget: float, overhead: float, n: int = 4) -> PDNTopology:
    domain = PowerDomain(
        name=DOMAIN,
        budget_w=budget,
        gpu_indices=list(range(n)),
        pdu_name="pdu0",
        node_overhead_w=overhead,
    )
    pdu = PDU(name="pdu0", capacity_w=budget * 4, derating=1.0)
    return PDNTopology(pdus={"pdu0": pdu}, domains={DOMAIN: domain})


def test_domain_budget_w_subtracts_overhead():
    topo = _topo(budget=4000.0, overhead=600.0)
    assert topo.domain_budget_w(DOMAIN) == 3400.0  # 4000 - 600
    # zero-overhead domain is unchanged
    assert _topo(budget=4000.0, overhead=0.0).domain_budget_w(DOMAIN) == 4000.0


def test_available_budget_floors_at_zero():
    # overhead larger than budget must not produce a negative available budget
    assert _topo(budget=1000.0, overhead=1500.0).domain_budget_w(DOMAIN) == 0.0


def test_prs_caps_respect_available_budget():
    """A busy domain with node overhead: PRS GPU caps must sum to <= the
    overhead-adjusted budget, not the raw budget."""
    budget, overhead, n = 4000.0, 1000.0, 4
    topo = _topo(budget=budget, overhead=overhead, n=n)
    brain = PRSBrain(topo)
    state = DomainState(
        domain_name=DOMAIN,
        gpu_draws={i: 1500.0 for i in range(n)},   # all hot, contended
        gpu_caps={i: 2000.0 for i in range(n)},
        gpu_max_caps={i: 2000.0 for i in range(n)},
        ts=time.time(),
    )
    last = None
    for _ in range(6):  # let EWMA converge
        last = brain.decide(DOMAIN, state)
    total = sum(last.caps.values())
    assert total <= (budget - overhead) + 1.0, f"caps {total} exceed GPU-available budget"
    assert total < budget, "must be strictly below the raw budget (overhead reserved)"


def test_validate_allocation_respects_overhead():
    topo = _topo(budget=4000.0, overhead=1000.0, n=4)
    # Total 3400 fits the GPU-available budget (3000)? No -> rejected.
    over = {i: 850.0 for i in range(4)}  # sum 3400 > 3000 available
    assert topo.validate_allocation(DOMAIN, over) is False
    # Total 3000 exactly fits the available budget -> accepted.
    ok = {i: 750.0 for i in range(4)}  # sum 3000 == available
    assert topo.validate_allocation(DOMAIN, ok) is True
