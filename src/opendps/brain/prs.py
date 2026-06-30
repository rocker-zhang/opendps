"""Brain v2: Power Reclaim Service (PRS) — EWMA-based oversubscription reclaim.

Improvements over DPM (brain v1):
- EWMA-smoothed power draw estimates per GPU (avoids over-reacting to transients)
- Hot/idle GPU classification: draw/cap ≥ threshold → hot; otherwise → idle
- Idle GPU caps reduced to ewma × (1+margin): frees stranded watts
- Reclaimed budget redistributed to hot GPUs proportionally by EWMA draw

Key metric: idle_stranded_watts = Σ(cap_i - draw_i) for idle GPUs
             (how much budget is allocated to GPUs that aren't using it)
PRS goal:  minimize idle_stranded_watts while keeping Σcap_i ≤ domain budget.

Algorithm (per tick):
  1. α-blend each GPU's EWMA: ewma_i ← α·draw_i + (1-α)·ewma_i
  2. Classify: idle if ewma_i/cap_i < reclaim_threshold (default 0.6)
  3. Idle floor: cap_i = max(min_cap, ewma_i · (1+idle_margin))
  4. Hot budget: budget − Σ(idle floors)
  5. Hot caps: proportional by EWMA draw, clamped to [min_cap, hw_max_i]
  6. Record metrics (stranded_w, hot/idle counts, EWMA values)
"""

from __future__ import annotations

from dataclasses import dataclass

from opendps.brain.dpm import BrainDecision, DomainState
from opendps.pdn.model import PDNTopology


@dataclass
class PRSMetrics:
    """Per-tick diagnostics produced alongside a PRS decision."""

    hot_gpus: list[int]
    idle_gpus: list[int]
    idle_stranded_w: float     # Σ(idle_cap_i - idle_draw_i) — the money metric
    domain_draw_w: float       # Σ draws
    domain_cap_w: float        # Σ new caps (≤ budget by construction)
    ewma_draws: dict[int, float]


class PRSBrain:
    """Brain v2: EWMA-based power reclaim / oversubscription management.

    The brain is stateful: it accumulates per-GPU EWMA estimates across
    decide() calls.  Thread-safety: not thread-safe; use one instance per
    controller thread.
    """

    def __init__(
        self,
        topology: PDNTopology,
        min_cap_w: float = 200.0,
        ewma_alpha: float = 0.3,
        reclaim_threshold: float = 0.6,
        idle_floor_margin: float = 0.3,
        cap_raise_rate_w_per_tick: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        min_cap_w:
            Hard floor for any GPU cap (W).  The brain never goes below this.
        ewma_alpha:
            EWMA smoothing factor α ∈ (0, 1).  Higher → more reactive.
        reclaim_threshold:
            GPU is classified as idle if ewma_draw / cap < threshold.
        idle_floor_margin:
            Idle GPU cap = ewma × (1 + margin).  Provides a draw-spike buffer.
        cap_raise_rate_w_per_tick:
            N5 transient smoothing.  Maximum watts a single GPU's cap may *rise*
            per tick.  Cap *lowering* is never rate-limited (safety: shedding
            power must be immediate, same principle as the cap-lower-only
            failsafe).  ``0.0`` disables the limiter (unbounded raises).
        """
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError(f"ewma_alpha must be in (0, 1], got {ewma_alpha}")
        if cap_raise_rate_w_per_tick < 0.0:
            raise ValueError("cap_raise_rate_w_per_tick must be >= 0")
        self._topology = topology
        self._min_cap_w = min_cap_w
        self._alpha = ewma_alpha
        self._threshold = reclaim_threshold
        self._margin = idle_floor_margin
        self._cap_raise_rate = cap_raise_rate_w_per_tick
        # domain_name → {gpu_index: ewma_watts}
        self._ewma: dict[str, dict[int, float]] = {}
        # domain_name → {gpu_index: last_applied_cap_w} (for the raise limiter)
        self._last_caps: dict[str, dict[int, float]] = {}
        # Last per-domain diagnostics (set by decide, read by get_last_metrics)
        self._last_metrics: dict[str, PRSMetrics] = {}

    # ------------------------------------------------------------------
    # Public API (same signature as DPMBrain.decide for drop-in swap)
    # ------------------------------------------------------------------

    def decide(self, domain_name: str, state: DomainState) -> BrainDecision:
        """Compute new caps for all GPUs in domain.

        Side-effects:
          - Updates internal EWMA state for the domain.
          - Stores PRSMetrics accessible via get_last_metrics(domain_name).
        """
        budget = self._topology.domain_budget_w(domain_name)
        gpus = list(state.gpu_draws.keys())
        n = max(len(gpus), 1)
        fallback_max = budget / n

        # 1. Update EWMA
        domain_ewma = self._ewma.setdefault(domain_name, {})
        for gpu, draw in state.gpu_draws.items():
            prev = domain_ewma.get(gpu, draw)
            domain_ewma[gpu] = self._alpha * draw + (1.0 - self._alpha) * prev

        # 2. Classify GPUs
        hot: list[int] = []
        idle: list[int] = []
        for gpu in gpus:
            cap = state.gpu_caps.get(gpu, fallback_max)
            ewma = domain_ewma[gpu]
            if cap > 0 and (ewma / cap) >= self._threshold:
                hot.append(gpu)
            else:
                idle.append(gpu)

        # 3. Idle floor caps
        idle_caps: dict[int, float] = {}
        for gpu in idle:
            ewma = domain_ewma[gpu]
            hw_max = state.gpu_max_caps.get(gpu, state.gpu_caps.get(gpu, fallback_max))
            floor = max(self._min_cap_w, ewma * (1.0 + self._margin))
            idle_caps[gpu] = min(floor, hw_max)

        idle_total = sum(idle_caps.values())
        hot_budget = max(0.0, budget - idle_total)

        # 4. Hot GPU caps — proportional by EWMA draw
        hot_caps: dict[int, float] = {}
        if hot:
            total_hot_ewma = sum(domain_ewma[g] for g in hot)
            for gpu in hot:
                hw_max = state.gpu_max_caps.get(gpu, state.gpu_caps.get(gpu, fallback_max))
                if total_hot_ewma > 0.0:
                    share = domain_ewma[gpu] / total_hot_ewma
                else:
                    share = 1.0 / len(hot)
                raw = share * hot_budget
                hot_caps[gpu] = max(self._min_cap_w, min(hw_max, raw))

        caps = {**idle_caps, **hot_caps}

        # 4b. N5 transient smoothing — rate-limit cap *raises* (lowering is
        #     always immediate). Prevents a GPU's allowance from jumping in one
        #     tick, which would let draw spike faster than the PDN can absorb.
        if self._cap_raise_rate > 0.0:
            last = self._last_caps.get(domain_name, {})
            for gpu, target in caps.items():
                # On the first tick for a GPU, seed from its currently-applied
                # cap so the very first decision is smoothed too (not a free jump).
                prev = last.get(gpu, state.gpu_caps.get(gpu))
                if prev is not None and target > prev:
                    caps[gpu] = min(target, prev + self._cap_raise_rate)
        self._last_caps[domain_name] = dict(caps)

        # 5. Compute metrics (on the actually-applied, rate-limited caps)
        idle_stranded = sum(
            caps[g] - state.gpu_draws.get(g, 0.0) for g in idle
        )
        domain_draw = sum(state.gpu_draws.values())
        domain_cap = sum(caps.values())

        metrics = PRSMetrics(
            hot_gpus=list(hot),
            idle_gpus=list(idle),
            idle_stranded_w=max(0.0, idle_stranded),
            domain_draw_w=domain_draw,
            domain_cap_w=domain_cap,
            ewma_draws=dict(domain_ewma),
        )
        self._last_metrics[domain_name] = metrics

        reason = (
            f"prs:hot={len(hot)},idle={len(idle)}"
            f",stranded={metrics.idle_stranded_w:.0f}W"
        )
        return BrainDecision(domain=domain_name, caps=caps, ts=state.ts, reason=reason)

    def get_last_metrics(self, domain_name: str) -> PRSMetrics | None:
        """Return the PRSMetrics from the most recent decide() for this domain."""
        return self._last_metrics.get(domain_name)

    def note_applied_caps(self, domain_name: str, caps: dict[int, float]) -> None:
        """Record the caps a wrapping brain actually applied this tick, so the
        N5 cap-raise rate limiter compares against reality (not PRS's own
        pre-rewrite output) on the next tick."""
        self._last_caps[domain_name] = dict(caps)

    def reset_ewma(self, domain_name: str | None = None) -> None:
        """Clear EWMA state (useful for tests or warm-restart)."""
        if domain_name is None:
            self._ewma.clear()
            self._last_caps.clear()
            self._last_metrics.clear()
        else:
            self._ewma.pop(domain_name, None)
            self._last_caps.pop(domain_name, None)
            self._last_metrics.pop(domain_name, None)
