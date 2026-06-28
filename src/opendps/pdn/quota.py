"""Per-tenant power quota model for N13."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TenantQuota:
    """Fraction of domain budget allocated to a tenant."""
    tenant_id: str
    domain_name: str
    gpu_indices: list[int]   # GPUs belonging to this tenant
    max_watts_pct: float     # 0.0–1.0, fraction of domain budget_w

    def __post_init__(self):
        if not 0.0 < self.max_watts_pct <= 1.0:
            raise ValueError(f"max_watts_pct must be in (0, 1], got {self.max_watts_pct}")


@dataclass
class QuotaConfig:
    """Collection of per-tenant quotas for one domain."""
    domain_name: str
    tenants: list[TenantQuota] = field(default_factory=list)

    def validate(self) -> None:
        """Raise if total quota > 100% or GPU indices overlap."""
        total = sum(t.max_watts_pct for t in self.tenants)
        if total > 1.0 + 1e-6:
            raise ValueError(f"Total quota {total:.1%} > 100% for domain {self.domain_name}")
        seen: set[int] = set()
        for t in self.tenants:
            overlap = seen & set(t.gpu_indices)
            if overlap:
                raise ValueError(f"GPU indices {overlap} assigned to multiple tenants")
            seen.update(t.gpu_indices)

    def tenant_budget_w(self, tenant: TenantQuota, domain_budget_w: float) -> float:
        return tenant.max_watts_pct * domain_budget_w
