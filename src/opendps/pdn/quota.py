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
        """Raise if total quota > 100%, GPUs overlap, or a tenant names a
        different domain (the brain silently skips mismatched tenants, so a
        stray domain_name would disable enforcement without warning)."""
        stray = [t.tenant_id for t in self.tenants if t.domain_name != self.domain_name]
        if stray:
            raise ValueError(
                f"tenants {stray} have a domain_name != {self.domain_name!r}"
            )
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

    @classmethod
    def from_dict(cls, data: dict) -> "QuotaConfig":
        """Build and validate a QuotaConfig from a parsed JSON dict.

        Schema::

            {
              "domain_name": "dom0",
              "tenants": [
                {"tenant_id": "tenant-a", "gpu_indices": [0,1,2], "max_watts_pct": 0.6},
                ...
              ]
            }

        A tenant's ``domain_name`` defaults to the top-level ``domain_name`` when
        omitted. Raises ``ValueError`` for malformed input or quota violations so
        a bad config fails loudly rather than silently mis-allocating power.
        """
        if not isinstance(data, dict):
            raise ValueError("quota config must be a JSON object")
        try:
            domain_name = data["domain_name"]
            tenants = [
                TenantQuota(
                    tenant_id=t["tenant_id"],
                    domain_name=t.get("domain_name", domain_name),
                    gpu_indices=[int(g) for g in t["gpu_indices"]],
                    max_watts_pct=float(t["max_watts_pct"]),
                )
                for t in data.get("tenants", [])
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed quota config: {exc}") from exc
        cfg = cls(domain_name=domain_name, tenants=tenants)
        cfg.validate()
        return cfg
