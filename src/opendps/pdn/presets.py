"""Preset PDN topologies for demo and testing."""

from __future__ import annotations

from .model import PDU, PDNTopology, PowerDomain


def demo_single_domain(n_gpus: int = 10, budget_w: float = 8000.0) -> PDNTopology:
    """One PDU, one domain — the headline 25 % oversubscription scenario.

    With n_gpus=10 and budget_w=8000 the domain is oversubscribed at 1000 W/GPU
    (10 000 W requested vs 8 000 W budget). This is the money-shot demo scenario.
    """
    pdu = PDU(name="pdu-A", capacity_w=10_000.0, derating=0.9)
    domain = PowerDomain(
        name="domain-0",
        budget_w=budget_w,
        gpu_indices=list(range(n_gpus)),
        pdu_name="pdu-A",
        priority=1,
        node_overhead_w=0.0,
    )
    return PDNTopology(pdus={"pdu-A": pdu}, domains={"domain-0": domain})


def demo_multi_domain(
    gpus_per_domain: int = 4,
    n_domains: int = 2,
    budget_per_domain: float = 3000.0,
) -> PDNTopology:
    """Two PDUs, each hosting one domain; balanced layout for multi-tenant demos."""
    pdus: dict = {}
    domains: dict = {}
    gpu_offset = 0
    for i in range(n_domains):
        pdu_name = f"pdu-{chr(ord('A') + i)}"
        domain_name = f"domain-{i}"
        pdus[pdu_name] = PDU(name=pdu_name, capacity_w=budget_per_domain / 0.9, derating=0.9)
        domains[domain_name] = PowerDomain(
            name=domain_name,
            budget_w=budget_per_domain,
            gpu_indices=list(range(gpu_offset, gpu_offset + gpus_per_domain)),
            pdu_name=pdu_name,
            priority=0,
            node_overhead_w=0.0,
        )
        gpu_offset += gpus_per_domain
    return PDNTopology(pdus=pdus, domains=domains)
