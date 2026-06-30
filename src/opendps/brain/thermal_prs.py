"""N16 — thermal-aware PRS brain.

A thermally-throttling GPU is heat-limited, not power-limited: giving it more
power cap won't make it draw more, it just wastes budget (and invites more heat).
This brain wraps PRS and, for any GPU flagged thermal-throttled, refuses to raise
its cap above what it currently holds; the watts freed that way are handed to the
non-throttled GPUs that can actually use them. The domain budget is never
exceeded.
"""
from __future__ import annotations

from opendps.brain.dpm import BrainDecision, DomainState
from opendps.brain.prs import PRSBrain
from opendps.pdn.model import PDNTopology


class ThermalAwarePRSBrain:
    def __init__(self, topology: PDNTopology, thermal_derate: float = 0.15,
                 min_cap_w: float = 200.0, **prs_kwargs):
        if not 0.0 <= thermal_derate < 1.0:
            raise ValueError(f"thermal_derate must be in [0, 1), got {thermal_derate}")
        self._topology = topology
        self._prs = PRSBrain(topology, **prs_kwargs)
        self._derate = thermal_derate
        self._min_cap_w = min_cap_w

    def decide(self, domain_name: str, state: DomainState) -> BrainDecision:
        decision = self._prs.decide(domain_name, state)
        throttled = [g for g in decision.caps if state.gpu_thermal_throttled.get(g, False)]
        if not throttled:
            return decision

        caps = dict(decision.caps)
        # A thermal-throttled GPU is heat-limited — running it at full power just
        # makes heat it can't dissipate. Back its cap off by thermal_derate (down
        # to the floor) and free the watts for GPUs that can actually use them.
        freed = 0.0
        for g in throttled:
            reduced = max(self._min_cap_w, caps[g] * (1.0 - self._derate))
            freed += caps[g] - reduced
            caps[g] = reduced

        # Redistribute the freed watts to non-throttled GPUs that have headroom
        # below their hardware max, proportional to their current cap.
        if freed > 0:
            recipients = [g for g in caps if g not in throttled]
            headroom = {
                g: max(0.0, state.gpu_max_caps.get(g, caps[g]) - caps[g])
                for g in recipients
            }
            total_head = sum(headroom.values())
            if total_head > 0:
                give = min(freed, total_head)
                for g in recipients:
                    caps[g] += give * headroom[g] / total_head

        self._prs.note_applied_caps(domain_name, caps)
        return BrainDecision(
            domain=domain_name,
            caps=caps,
            reason=f"thermal-prs:{len(throttled)}throttled",
            ts=state.ts,
        )

    def get_last_metrics(self, domain_name: str):
        return self._prs.get_last_metrics(domain_name)
