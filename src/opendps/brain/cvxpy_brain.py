from __future__ import annotations
import time
from dataclasses import dataclass

from opendps.brain.dpm import BrainDecision, DomainState
from opendps.brain.prs import PRSBrain
from opendps.pdn.model import PDNTopology


@dataclass
class CVXPYMetrics:
    hot_gpus: list[int]
    idle_gpus: list[int]
    idle_stranded_w: float
    domain_draw_w: float
    domain_cap_w: float
    solver_status: str
    solve_time_ms: float


class CVXPYBrain:
    """
    LP-based GPU power allocator.

    Formulation (per domain, per tick):
      Variables: cap[i] for each GPU i
      Minimize:  sum(cap[i] - draw[i]) for hot GPUs  (minimize wasted headroom)
      Subject to:
        sum(cap) <= budget
        cap[i] >= min_cap_w
        cap[i] <= gpu_max_cap[i]
        cap[i] <= max(min_cap_w, ewma[i] * reclaim_ceil)  for idle GPUs

    Falls back to PRSBrain if cvxpy or solver unavailable.
    """

    def __init__(
        self,
        topology: PDNTopology,
        min_cap_w: float = 200.0,
        ewma_alpha: float = 0.3,
        reclaim_threshold: float = 0.6,
        idle_reclaim_ceil: float = 1.3,
    ):
        self._topo = topology
        self._min_cap = min_cap_w
        self._alpha = ewma_alpha
        self._threshold = reclaim_threshold
        self._idle_ceil = idle_reclaim_ceil
        self._ewma: dict[str, dict[int, float]] = {}
        self._last_metrics: dict[str, CVXPYMetrics] = {}
        self._fallback = PRSBrain(topology, min_cap_w=min_cap_w, ewma_alpha=ewma_alpha)

    # ------------------------------------------------------------------ public
    def decide(self, domain_name: str, state: DomainState) -> BrainDecision:
        try:
            import cvxpy as cp  # noqa: PLC0415
            return self._decide_cvxpy(domain_name, state, cp)
        except Exception:
            return self._fallback.decide(domain_name, state)

    def get_last_metrics(self, domain_name: str) -> CVXPYMetrics | None:
        return self._last_metrics.get(domain_name)

    # ----------------------------------------------------------------- private
    def _ewma_update(self, domain: str, draws: dict[int, float]) -> dict[int, float]:
        if domain not in self._ewma:
            self._ewma[domain] = dict(draws)
        state = self._ewma[domain]
        for gpu, w in draws.items():
            state[gpu] = self._alpha * w + (1 - self._alpha) * state.get(gpu, w)
        return state

    def _decide_cvxpy(
        self, domain_name: str, state: DomainState, cp
    ) -> BrainDecision:
        domain = self._topo.domains[domain_name]
        budget = domain.budget_w
        gpus = sorted(state.gpu_draws.keys())
        n = len(gpus)

        ewma = self._ewma_update(domain_name, state.gpu_draws)
        max_caps = [state.gpu_max_caps[g] for g in gpus]
        draws = [state.gpu_draws[g] for g in gpus]

        hot = [
            ewma[g] / max(max_caps[i], 1.0) >= self._threshold
            for i, g in enumerate(gpus)
        ]

        caps_var = cp.Variable(n, nonneg=True)

        # Objective: minimize unused headroom for hot GPUs
        hot_indices = [i for i, h in enumerate(hot) if h]
        if hot_indices:
            obj = cp.Minimize(
                cp.sum([caps_var[i] - draws[i] for i in hot_indices])
            )
        else:
            obj = cp.Minimize(cp.sum(caps_var))

        constraints: list = [
            cp.sum(caps_var) <= budget,
            caps_var >= self._min_cap,
            caps_var <= max_caps,
        ]
        for i, g in enumerate(gpus):
            if not hot[i]:
                idle_ceil = max(self._min_cap, ewma[g] * self._idle_ceil)
                constraints.append(caps_var[i] <= idle_ceil)

        prob = cp.Problem(obj, constraints)
        t0 = time.perf_counter()
        # Prefer GLPK (exact LP); fall back through available solvers.
        installed = cp.installed_solvers()
        for solver_name in ("GLPK", "HIGHS", "SCIPY", "CLARABEL", "SCS"):
            if solver_name in installed:
                prob.solve(solver=getattr(cp, solver_name), verbose=False)
                break
        else:
            prob.solve(verbose=False)
        solve_ms = (time.perf_counter() - t0) * 1000

        if prob.status not in ("optimal", "optimal_inaccurate") or caps_var.value is None:
            m = self._fallback.get_last_metrics(domain_name)
            result = self._fallback.decide(domain_name, state)
            self._last_metrics[domain_name] = CVXPYMetrics(
                hot_gpus=[g for i, g in enumerate(gpus) if hot[i]],
                idle_gpus=[g for i, g in enumerate(gpus) if not hot[i]],
                idle_stranded_w=getattr(m, "idle_stranded_w", 0.0) if m else 0.0,
                domain_draw_w=sum(draws),
                domain_cap_w=sum(result.caps.values()),
                solver_status=f"fallback:{prob.status}",
                solve_time_ms=solve_ms,
            )
            return result

        result_caps = {g: float(caps_var.value[i]) for i, g in enumerate(gpus)}
        hot_gpus = [g for i, g in enumerate(gpus) if hot[i]]
        idle_gpus = [g for i, g in enumerate(gpus) if not hot[i]]
        idle_stranded = sum(
            max_caps[i] - result_caps[g]
            for i, g in enumerate(gpus)
            if not hot[i]
        )

        self._last_metrics[domain_name] = CVXPYMetrics(
            hot_gpus=hot_gpus,
            idle_gpus=idle_gpus,
            idle_stranded_w=idle_stranded,
            domain_draw_w=sum(draws),
            domain_cap_w=sum(result_caps.values()),
            solver_status=prob.status,
            solve_time_ms=solve_ms,
        )
        return BrainDecision(
            domain=domain_name,
            caps=result_caps,
            ts=state.ts,
            reason=f"cvxpy:{prob.status}:{solve_ms:.1f}ms",
        )
