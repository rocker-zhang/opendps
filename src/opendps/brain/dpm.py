"""Brain v1: static proportional cap allocation (DPM).

DPM = per-domain hard budget enforcement.  No prediction, no solver — just
proportional headroom distribution every tick.

Algorithm (per domain per tick):
  1. Observe total GPU power draw for the domain.
  2. If total draw <= domain budget: release caps to hardware max (no throttling).
  3. If total draw > domain budget: allocate budget proportionally to each GPU's
     current draw, clamping each resulting cap to [min_cap_w, max_cap_w].
     - Proportional allocation concentrates headroom on the hottest GPUs.
     - min_cap_w prevents complete GPU starvation.
     - max_cap_w is the current hardware cap from state.gpu_caps.

The brain is stateless between calls — the only mutable state lives in the
topology dataclass passed at construction time (which should not be mutated).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from opendps.pdn.model import PDNTopology


@dataclass
class DomainState:
    """Current observed state of one power domain."""

    domain_name: str
    gpu_draws: dict[int, float]    # gpu_index -> current power draw (W)
    gpu_caps: dict[int, float]     # gpu_index -> current reported cap (W)
    ts: float                      # epoch seconds when the sample was taken
    gpu_max_caps: dict[int, float] = field(default_factory=dict)  # hardware max per GPU
    # GPUs currently thermal-throttling (heat-limited, not power-limited).
    # Empty when unknown; brains treat an absent GPU as not-throttled.
    gpu_thermal_throttled: dict[int, bool] = field(default_factory=dict)


@dataclass
class BrainDecision:
    """What the brain wants the actuator to do next tick."""

    domain: str
    caps: dict[int, float]   # gpu_index -> new requested cap (W)
    ts: float
    reason: str              # "under_budget" | "over_budget" | "rebalance"


class DPMBrain:
    """
    Brain v1: static proportional allocation.

    If total draw <= budget: keep caps at current max (no throttling needed).
    If total draw > budget: proportionally reduce caps so total equals budget,
    clamping each to [min_cap_w, max_cap_w].  GPUs with higher draw receive
    proportionally more budget — the hot-GPU-priority property of this approach.

    Minimum cap per GPU: min_cap_w (default 200 W).  The brain will never push
    a cap below this floor even if doing so would bring the domain into budget.

    The brain is stateless: call decide() as many times as you like; it
    produces the same output for the same (topology, state) inputs.
    """

    def __init__(self, topology: PDNTopology, min_cap_w: float = 200.0) -> None:
        self._topology = topology
        self._min_cap_w = min_cap_w

    def decide(self, domain_name: str, state: DomainState) -> BrainDecision:
        """Compute new caps for all GPUs in domain given current state.

        Parameters
        ----------
        domain_name:
            Must be a key in topology.domains.
        state:
            Must contain entries for all gpu_indices in the domain.

        Returns
        -------
        BrainDecision with caps that respect the domain budget and the
        [min_cap_w, max_cap_w] per-GPU envelope.
        """
        budget = self._topology.domain_budget_w(domain_name)
        gpus = list(state.gpu_draws.keys())
        total_draw = sum(state.gpu_draws.values())

        if total_draw <= budget:
            # Under budget: restore caps to hardware max (not the last reported cap,
            # which may have been reduced by a prior over-budget tick).
            caps = {
                gpu: state.gpu_max_caps.get(gpu, state.gpu_caps.get(gpu, self._min_cap_w))
                for gpu in gpus
            }
            reason = "under_budget"
        else:
            # Over budget: proportionally allocate budget to GPUs by draw.
            caps: dict[int, float] = {}
            if total_draw > 0.0:
                for gpu in gpus:
                    max_cap = state.gpu_caps.get(gpu, self._min_cap_w)
                    proportion = state.gpu_draws[gpu] / total_draw
                    raw = proportion * budget
                    # Clamp: never exceed hardware max, never go below floor.
                    caps[gpu] = max(self._min_cap_w, min(max_cap, raw))
            else:
                # Degenerate: zero total draw but over budget (budget <= 0).
                # Distribute evenly clamped to min floor.
                equal_share = max(self._min_cap_w, budget / len(gpus)) if gpus else 0.0
                caps = {gpu: equal_share for gpu in gpus}
            reason = "over_budget"

        return BrainDecision(domain=domain_name, caps=caps, ts=state.ts, reason=reason)
