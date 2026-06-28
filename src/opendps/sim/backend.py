"""Simulated GPU fleet — digital twin backend.

Power model calibrated from real DCGM telemetry observations:

    power_draw = cap * (util_pct / 100) * LOAD_EFFICIENCY + IDLE_POWER_W

where LOAD_EFFICIENCY = 0.85 and IDLE_POWER_W = 50 W.  The result is clamped
to [0, cap] so the sim never exceeds the current cap.

All randomness is driven by a seeded random.Random instance — no global state —
so scenarios are fully reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

_LOAD_EFFICIENCY: float = 0.85
_IDLE_POWER_W: float = 50.0


@dataclass
class LoadProfile:
    """Synthetic workload description for one GPU in the demo scenario."""

    base_util_pct: float        # e.g. 90.0 for a hot GPU
    util_noise: float = 5.0     # ±noise added each tick
    idle_fraction: float = 0.0  # 0–1: fraction of ticks the GPU is idle

    def sample_util(self, rng: random.Random) -> float:
        """Return a utilization sample in [0, 100]."""
        if self.idle_fraction > 0.0 and rng.random() < self.idle_fraction:
            # GPU is idle this tick — return a low utilisation value.
            return max(0.0, rng.uniform(2.0, 8.0))
        noise = rng.uniform(-self.util_noise, self.util_noise)
        return max(0.0, min(100.0, self.base_util_pct + noise))


@dataclass
class SimGpu:
    """State for a single simulated GPU."""

    index: int
    cap_w: float           # current power cap (adjusted by controller)
    max_cap_w: float       # hardware maximum — set_power_cap cannot exceed this
    load: LoadProfile
    _rng: random.Random = field(repr=False)
    _current_util: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Seed the first utilisation sample so power_draw_w() is valid immediately.
        self._current_util = self.load.sample_util(self._rng)

    # ------------------------------------------------------------------
    # Sim-internal helpers
    # ------------------------------------------------------------------

    def _resample(self) -> None:
        """Advance internal state by one tick (re-draw utilisation noise)."""
        self._current_util = self.load.sample_util(self._rng)

    # ------------------------------------------------------------------
    # Observable quantities (read by SimBackend / snapshot)
    # ------------------------------------------------------------------

    def util_pct(self) -> float:
        """Return current utilisation percentage (consistent within a tick)."""
        return self._current_util

    def power_draw_w(self) -> float:
        """Return current power draw, clamped to cap_w."""
        draw = self.cap_w * (self._current_util / 100.0) * _LOAD_EFFICIENCY + _IDLE_POWER_W
        return min(draw, self.cap_w)


class SimBackend:
    """Implements the Actuator protocol for a simulated GPU fleet."""

    def __init__(self, gpus: list[SimGpu]) -> None:
        self._gpus: dict[int, SimGpu] = {g.index: g for g in gpus}

    # ------------------------------------------------------------------
    # Actuator protocol
    # ------------------------------------------------------------------

    def set_power_cap(self, gpu_index: int, watts: float) -> None:
        gpu = self._gpus[gpu_index]
        gpu.cap_w = min(max(0.0, watts), gpu.max_cap_w)

    def get_power_cap(self, gpu_index: int) -> float:
        return self._gpus[gpu_index].cap_w

    def get_power_draw(self, gpu_index: int) -> float:
        return self._gpus[gpu_index].power_draw_w()

    def get_util_pct(self, gpu_index: int) -> float:
        return self._gpus[gpu_index].util_pct()

    def get_max_cap_w(self, gpu_index: int) -> float:
        """Return the hardware-maximum power cap for gpu_index (never changes)."""
        return self._gpus[gpu_index].max_cap_w

    def gpu_count(self) -> int:
        return len(self._gpus)

    # ------------------------------------------------------------------
    # Sim-specific helpers
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance sim state (resample utilisation for all GPUs).

        Call this once between control ticks to model the passage of time.
        """
        for gpu in self._gpus.values():
            gpu._resample()

    def snapshot(self) -> list[dict]:
        """Return a list of {index, power_draw_w, cap_w, util_pct} for all GPUs."""
        return [
            {
                "index": gpu.index,
                "power_draw_w": gpu.power_draw_w(),
                "cap_w": gpu.cap_w,
                "util_pct": gpu.util_pct(),
            }
            for gpu in sorted(self._gpus.values(), key=lambda g: g.index)
        ]
