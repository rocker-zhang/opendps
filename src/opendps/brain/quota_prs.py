"""N13 — Per-tenant quota-aware PRS brain."""
from __future__ import annotations
import time

from opendps.brain.dpm import BrainDecision, DomainState
from opendps.brain.prs import PRSBrain
from opendps.pdn.model import PDNTopology, PowerDomain
from opendps.pdn.quota import QuotaConfig


class QuotaAwarePRSBrain:
    """
    Enforces per-tenant power budget slices before running PRS within each slice.

    Algorithm (per tick, per domain):
      1. Compute each tenant's budget = domain_budget * tenant.max_watts_pct
      2. Build a sub-DomainState for each tenant (only its GPU indices)
      3. Run PRSBrain.decide() with a mock topology capped at tenant budget
      4. Merge all tenant decisions; unassigned GPUs share remaining budget equally
    """

    def __init__(self, topology: PDNTopology, quota_config: QuotaConfig, **prs_kwargs):
        self._topo = topology
        self._quota = quota_config
        self._prs_kwargs = prs_kwargs
        # One PRSBrain per tenant (each has its own EWMA state)
        self._tenant_brains: dict[str, PRSBrain] = {}

    def decide(self, domain_name: str, state: DomainState) -> BrainDecision:
        domain = self._topo.domains[domain_name]
        budget = domain.budget_w
        all_caps: dict[int, float] = {}

        assigned_gpus: set[int] = set()
        used_budget = 0.0

        for tenant in self._quota.tenants:
            if tenant.domain_name != domain_name:
                continue

            tenant_budget = self._quota.tenant_budget_w(tenant, budget)
            tenant_gpus = [g for g in tenant.gpu_indices if g in state.gpu_draws]

            if not tenant_gpus:
                continue

            # Create a sub-topology with tenant budget.
            # pdus is empty since PRSBrain only uses domain_budget_w, which reads
            # domains[name].budget_w directly without touching the pdus dict.
            sub_domain_name = f"{domain_name}/{tenant.tenant_id}"
            sub_domain = PowerDomain(
                name=sub_domain_name,
                gpu_indices=tenant_gpus,
                budget_w=tenant_budget,
                pdu_name="_virtual",
                node_overhead_w=0.0,
            )
            sub_topo = PDNTopology(pdus={}, domains={sub_domain_name: sub_domain})

            sub_state = DomainState(
                domain_name=sub_domain_name,
                gpu_draws={g: state.gpu_draws[g] for g in tenant_gpus},
                gpu_caps={g: state.gpu_caps[g] for g in tenant_gpus},
                gpu_max_caps={g: state.gpu_max_caps[g] for g in tenant_gpus},
                ts=state.ts,
            )

            if tenant.tenant_id not in self._tenant_brains:
                self._tenant_brains[tenant.tenant_id] = PRSBrain(sub_topo, **self._prs_kwargs)
            brain = self._tenant_brains[tenant.tenant_id]

            decision = brain.decide(sub_domain_name, sub_state)
            all_caps.update(decision.caps)
            assigned_gpus.update(tenant_gpus)
            used_budget += sum(decision.caps.values())

        # Unassigned GPUs get equal share of remaining budget
        unassigned = [g for g in state.gpu_draws if g not in assigned_gpus]
        if unassigned:
            remaining = max(0.0, budget - used_budget)
            share = remaining / len(unassigned)
            for g in unassigned:
                all_caps[g] = min(share, state.gpu_max_caps[g])

        return BrainDecision(
            domain=domain_name,
            caps=all_caps,
            reason=f"quota-prs:{len(self._quota.tenants)}tenants",
            ts=time.time(),
        )

    def get_last_metrics(self, domain_name: str):
        return None
