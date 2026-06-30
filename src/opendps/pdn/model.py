"""PDN topology and capacity model.

The hierarchy is: Facility → Rack(s) → PDU(s) → PowerDomain(s) → GPU indices.
All arithmetic is intentionally float-only so the module stays dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PowerDomain:
    name: str
    budget_w: float
    gpu_indices: list[int]
    pdu_name: str
    priority: int = 0
    node_overhead_w: float = 0.0
    """Per-node non-GPU draw (CPU, NVSwitch, memory) in watts. Add from IPMI when available."""
    rack_name: str | None = None
    """Optional rack membership. When set and the rack's budget is below the sum
    of its domains' budgets, each domain's effective budget is scaled to its
    proportional share of the rack."""

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
class Rack:
    """A rack-level power budget shared by its member domains (between the PDU
    and cluster tiers). Its budget constrains the sum of its domains' budgets."""
    name: str
    budget_w: float


@dataclass
class PDNTopology:
    pdus: dict[str, PDU]
    domains: dict[str, PowerDomain]
    racks: dict[str, Rack] = field(default_factory=dict)
    # Runtime per-domain budget overrides adopted from a cluster coordinator.
    # Empty by default; when set, it supersedes the static topology budget.
    adopted_budget_w: dict[str, float] = field(default_factory=dict)

    def adopt_budget(self, domain_name: str, budget_w: float) -> None:
        """Override a domain's base budget at runtime (e.g. a node budget handed
        down by the cluster coordinator). Supersedes the static/rack budget."""
        self.adopted_budget_w[domain_name] = budget_w

    def release_budget(self, domain_name: str) -> None:
        self.adopted_budget_w.pop(domain_name, None)

    def domain_budget_w(self, domain_name: str) -> float:
        """Budget the brains allocate GPU caps against: the domain budget minus
        node overhead. A coordinator-adopted budget (if present) supersedes the
        static budget; otherwise, when the domain belongs to a rack whose budget
        is below the sum of its domains' budgets, the domain's budget is scaled
        to its proportional share of the rack. With no override, no rack and zero
        overhead this is just ``budget_w``."""
        domain = self.domains[domain_name]
        if domain_name in self.adopted_budget_w:
            return max(0.0, self.adopted_budget_w[domain_name] - domain.node_overhead_w)
        budget = domain.budget_w
        rack = self.rack_for_domain(domain_name)
        if rack is not None:
            total = sum(d.budget_w for d in self.domains_on_rack(domain.rack_name))
            if total > rack.budget_w > 0:
                budget = budget * rack.budget_w / total
        return max(0.0, budget - domain.node_overhead_w)

    def pdu_for_domain(self, domain_name: str) -> PDU:
        return self.pdus[self.domains[domain_name].pdu_name]

    def domains_on_pdu(self, pdu_name: str) -> list[PowerDomain]:
        return [d for d in self.domains.values() if d.pdu_name == pdu_name]

    def domains_on_rack(self, rack_name: str | None) -> list[PowerDomain]:
        return [d for d in self.domains.values() if d.rack_name == rack_name]

    def rack_for_domain(self, domain_name: str) -> Rack | None:
        rack_name = self.domains[domain_name].rack_name
        return self.racks.get(rack_name) if rack_name else None

    def validate_allocation(
        self, domain_name: str, proposed_caps: dict[int, float]
    ) -> bool:
        domain = self.domains[domain_name]
        proposed_total = sum(proposed_caps.values())
        # GPU caps must fit the budget left for GPUs after node overhead (N18).
        if proposed_total > domain.available_gpu_budget_w:
            return False

        pdu = self.pdus[domain.pdu_name]
        # Sum draw for all domains on the same PDU, substituting proposed where it
        # applies. This domain draws its proposed GPU caps plus its node overhead;
        # other domains' draw is unknown, so use their full budget as the ceiling.
        pdu_total = 0.0
        for d in self.domains_on_pdu(domain.pdu_name):
            if d.name == domain_name:
                pdu_total += proposed_total + domain.node_overhead_w
            else:
                pdu_total += d.budget_w
        if pdu_total > pdu.effective_capacity_w:
            return False

        # Rack rollup: when the domain belongs to a rack, its proposed draw plus
        # the peer domains' budgets on that rack must fit the rack budget.
        rack = self.rack_for_domain(domain_name)
        if rack is not None:
            rack_total = 0.0
            for d in self.domains_on_rack(domain.rack_name):
                if d.name == domain_name:
                    rack_total += proposed_total + domain.node_overhead_w
                else:
                    rack_total += d.budget_w
            if rack_total > rack.budget_w:
                return False

        return True

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

    racks: dict[str, Rack] = {}
    for name, spec in config.get("racks", {}).items():
        racks[name] = Rack(name=name, budget_w=float(spec["budget_w"]))

    domains: dict[str, PowerDomain] = {}
    for name, spec in config["domains"].items():
        domains[name] = PowerDomain(
            name=name,
            budget_w=float(spec["budget_w"]),
            gpu_indices=list(spec["gpu_indices"]),
            pdu_name=spec["pdu_name"],
            priority=int(spec.get("priority", 0)),
            node_overhead_w=float(spec.get("node_overhead_w", 0.0)),
            rack_name=spec.get("rack_name"),
        )

    # A typo'd rack_name would silently disable rack enforcement; fail loudly.
    for d in domains.values():
        if d.rack_name is not None and d.rack_name not in racks:
            raise ValueError(f"domain {d.name!r} references unknown rack {d.rack_name!r}")

    return PDNTopology(pdus=pdus, domains=domains, racks=racks)
