"""N15 — SLA-tiered priority preemption brain.

PRS reclaims fairly by draw, but has no notion of workload importance. Under
power pressure a busy low-priority GPU competes equally with a busy high-priority
one. N15 adds priority tiers: among *contended* GPUs, the contended budget is
split in proportion to draw x tier weight, so a higher-tier GPU keeps more cap
(effectively preempting power from lower-tier neighbours). Idle / uncontended
GPUs keep their PRS floor untouched, and the domain budget is never exceeded.
"""
from __future__ import annotations

import time

from opendps.brain.dpm import BrainDecision, DomainState
from opendps.brain.prs import PRSBrain
from opendps.pdn.model import PDNTopology

# Relative power weight per SLA tier (matches the JobPowerPolicy priorityClass
# enum: low / normal / high / critical).
TIER_WEIGHTS: dict[str, float] = {
    "low": 0.5,
    "normal": 1.0,
    "high": 2.0,
    "critical": 4.0,
}
DEFAULT_TIER = "normal"


class PriorityTieredPRSBrain:
    """Wraps PRSBrain and biases the contended-GPU allocation by SLA tier."""

    def __init__(
        self,
        topology: PDNTopology,
        gpu_priority_tiers: dict[int, str],
        contention_threshold: float = 0.6,
        min_cap_w: float = 200.0,
        **prs_kwargs,
    ):
        bad = {t for t in gpu_priority_tiers.values() if t not in TIER_WEIGHTS}
        if bad:
            raise ValueError(f"unknown priority tier(s) {sorted(bad)}; valid: {sorted(TIER_WEIGHTS)}")
        self._topology = topology
        self._prs = PRSBrain(topology, **prs_kwargs)
        self._tiers = dict(gpu_priority_tiers)
        self._threshold = contention_threshold
        self._min_cap_w = min_cap_w

    def _weight(self, gpu: int) -> float:
        return TIER_WEIGHTS[self._tiers.get(gpu, DEFAULT_TIER)]

    def decide(self, domain_name: str, state: DomainState) -> BrainDecision:
        decision = self._prs.decide(domain_name, state)
        caps = dict(decision.caps)

        # A GPU is contended if it is drawing near its current cap — these are the
        # GPUs competing for budget and where priority should arbitrate.
        contended = [
            g for g in caps
            if state.gpu_caps.get(g, 0.0) > 0.0
            and state.gpu_draws.get(g, 0.0) / state.gpu_caps[g] >= self._threshold
        ]
        # Tier only matters when at least two contended GPUs compete.
        if len(contended) < 2:
            return decision

        budget = self._topology.domain_budget_w(domain_name)
        if not budget:
            return decision

        # Reserve the uncontended (idle) caps and a per-GPU floor for every
        # contended GPU (no GPU is ever starved below min_cap_w), then split the
        # remaining surplus by draw x tier weight so higher tiers keep more.
        reserved_idle = sum(caps[g] for g in caps if g not in contended)
        floor = self._min_cap_w
        surplus = max(0.0, budget - reserved_idle - floor * len(contended))
        weights = {g: state.gpu_draws.get(g, 0.0) * self._weight(g) for g in contended}
        total_w = sum(weights.values()) or 1.0
        for g in contended:
            hw_max = state.gpu_max_caps.get(g, caps[g])
            caps[g] = min(floor + surplus * weights[g] / total_w, hw_max)

        # Hard safety: if the budget is below the sum of floors (a misconfigured,
        # physically-infeasible budget), scale every cap down so the cluster
        # budget is never oversubscribed — the power ceiling wins over the floor.
        total = sum(caps.values())
        if budget and total > budget:
            scale = budget / total
            for g in caps:
                caps[g] *= scale

        return BrainDecision(
            domain=domain_name,
            caps=caps,
            reason=f"priority-prs:{len(contended)}contended",
            ts=time.time(),
        )

    def get_last_metrics(self, domain_name: str):
        return self._prs.get_last_metrics(domain_name)
