"""PDN topology and capacity model.

The hierarchy is: Facility → PDU(s) → PowerDomain(s) → GPU indices.
All arithmetic is intentionally float-only so the module stays dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PowerDomain:
    name: str
    budget_w: float
    gpu_indices: list[int]
    pdu_name: str
    priority: int = 0
    node_overhead_w: float = 0.0
    """Per-node non-GPU draw (CPU, NVSwitch, memory) in watts. Add from IPMI when available."""

    @property
    def available_gpu_budget_w(self) -> float:
        """Budget available for GPU caps after subtracting node overhead."""
        return max(0.0, self.budget_w - self.node_overhead_w)


@dataclass
class PDU:
    name: str
    capacity_w: float
    derating: float = 0.9

    @property
    def effective_capacity_w(self) -> float:
        return self.capacity_w * self.derating


@dataclass
class PDNTopology:
    pdus: dict[str, PDU]
    domains: dict[str, PowerDomain]

    def domain_budget_w(self, domain_name: str) -> float:
        return self.domains[domain_name].budget_w

    def pdu_for_domain(self, domain_name: str) -> PDU:
        return self.pdus[self.domains[domain_name].pdu_name]

    def domains_on_pdu(self, pdu_name: str) -> list[PowerDomain]:
        return [d for d in self.domains.values() if d.pdu_name == pdu_name]

    def validate_allocation(
        self, domain_name: str, proposed_caps: dict[int, float]
    ) -> bool:
        domain = self.domains[domain_name]
        proposed_total = sum(proposed_caps.values())
        if proposed_total > domain.budget_w:
            return False

        pdu = self.pdus[domain.pdu_name]
        # Sum caps for all domains on the same PDU, substituting proposed where it applies.
        pdu_total = 0.0
        for d in self.domains_on_pdu(domain.pdu_name):
            if d.name == domain_name:
                pdu_total += proposed_total
            else:
                # Unknown current draw for other domains; use their budget as the ceiling.
                pdu_total += d.budget_w
        return pdu_total <= pdu.effective_capacity_w

    def total_gpu_count(self) -> int:
        return sum(len(d.gpu_indices) for d in self.domains.values())

    def oversubscription_ratio(
        self, domain_name: str, proposed_caps: dict[int, float]
    ) -> float:
        budget = self.domains[domain_name].budget_w
        proposed_total = sum(proposed_caps.values())
        return proposed_total / budget


@dataclass
class AllocationResult:
    domain: str
    caps: dict[int, float]
    total_w: float
    budget_w: float
    oversubscribed: bool
    headroom_w: float


def from_dict(config: dict) -> PDNTopology:
    """Build a PDNTopology from a plain config dict (e.g. parsed from YAML/JSON)."""
    pdus: dict[str, PDU] = {}
    for name, spec in config["pdus"].items():
        pdus[name] = PDU(
            name=name,
            capacity_w=float(spec["capacity_w"]),
            derating=float(spec.get("derating", 0.9)),
        )

    domains: dict[str, PowerDomain] = {}
    for name, spec in config["domains"].items():
        domains[name] = PowerDomain(
            name=name,
            budget_w=float(spec["budget_w"]),
            gpu_indices=list(spec["gpu_indices"]),
            pdu_name=spec["pdu_name"],
            priority=int(spec.get("priority", 0)),
        )

    return PDNTopology(pdus=pdus, domains=domains)
