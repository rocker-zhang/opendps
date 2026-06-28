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
        decision = self._prs.decide(domain_name, state)
        for gpu, cap in list(decision.caps.items()):
            if self._tracker.is_gpu_busy(gpu):
                boosted = min(cap * (1.0 + self._boost), state.gpu_max_caps[gpu])
                decision.caps[gpu] = boosted
        return decision

    def get_last_metrics(self, domain_name: str):
        return self._prs.get_last_metrics(domain_name)
