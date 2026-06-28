"""PRS brain wrapper that boosts caps for GPUs with active compute jobs."""
from __future__ import annotations
from opendps.brain.dpm import BrainDecision, DomainState
from opendps.brain.prs import PRSBrain
from opendps.pdn.model import PDNTopology
from opendps.agent.job_tracker import JobTracker


class JobAwarePRSBrain:
    """Wraps PRSBrain — GPUs with active jobs get a cap boost."""

    def __init__(
        self,
        topology: PDNTopology,
        job_tracker: JobTracker,
        priority_boost: float = 0.15,
        **prs_kwargs,
    ):
        self._prs = PRSBrain(topology, **prs_kwargs)
        self._tracker = job_tracker
        self._boost = priority_boost

    def decide(self, domain_name: str, state: DomainState) -> BrainDecision:
        # The priority boost is applied on top of the PRS decision and is
        # intentionally exempt from the N5 cap-raise rate limiter: a GPU that
        # just started a prioritised job should get its headroom immediately.
        decision = self._prs.decide(domain_name, state)
        for gpu, cap in list(decision.caps.items()):
            if self._tracker.is_gpu_busy(gpu):
                # gpu_max_caps may be empty/partial — fall back to the current
                # cap so a boost never raises above a known hardware max.
                hw_max = state.gpu_max_caps.get(gpu, cap)
                decision.caps[gpu] = min(cap * (1.0 + self._boost), hw_max)
        return decision

    def get_last_metrics(self, domain_name: str):
        return self._prs.get_last_metrics(domain_name)
