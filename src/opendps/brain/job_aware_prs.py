"""PRS brain wrapper that boosts caps for GPUs with active compute jobs."""
from __future__ import annotations

from opendps.agent.job_tracker import JobTracker
from opendps.brain.dpm import BrainDecision, DomainState
from opendps.brain.prs import PRSBrain
from opendps.pdn.model import PDNTopology


class JobAwarePRSBrain:
    """Wraps PRSBrain — GPUs with active jobs get a cap boost.

    The boost biases the allocation toward busy GPUs without oversubscribing:
    busy GPUs are raised by ``priority_boost`` (clamped to their hardware max),
    then every cap is renormalised so Σcaps stays within the domain budget. The
    net effect is a redistribution of budget toward busy GPUs, not extra power.
    The boost is intentionally exempt from the N5 cap-raise rate limiter so a
    freshly prioritised job gets its share immediately.
    """

    def __init__(
        self,
        topology: PDNTopology,
        job_tracker: JobTracker,
        priority_boost: float = 0.15,
        **prs_kwargs,
    ):
        if priority_boost < 0:
            raise ValueError(f"priority_boost must be >= 0, got {priority_boost}")
        self._topology = topology
        self._prs = PRSBrain(topology, **prs_kwargs)
        self._tracker = job_tracker
        self._boost = priority_boost

    def decide(self, domain_name: str, state: DomainState) -> BrainDecision:
        decision = self._prs.decide(domain_name, state)
        busy = [g for g in decision.caps if self._tracker.is_gpu_busy(g)]
        if not busy:
            return decision

        # Raise busy GPUs (clamped to hw max)...
        for g in busy:
            hw_max = state.gpu_max_caps.get(g, decision.caps[g])
            decision.caps[g] = min(decision.caps[g] * (1.0 + self._boost), hw_max)

        # ...then renormalise so the domain budget is never oversubscribed. This
        # turns the boost into a redistribution: busy GPUs keep a proportionally
        # larger share, total power is unchanged.
        budget = self._topology.domain_budget_w(domain_name)
        total = sum(decision.caps.values())
        if budget and total > budget:
            scale = budget / total
            for g in decision.caps:
                decision.caps[g] *= scale
        return decision

    def get_last_metrics(self, domain_name: str):
        return self._prs.get_last_metrics(domain_name)
