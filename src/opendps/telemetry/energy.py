"""Cumulative energy accounting (joules / watt-hours) per GPU and per tenant.

opendps otherwise exports only instantaneous power gauges. This integrates
per-GPU power draw over time (energy = Σ draw × dt) into a cumulative counter,
which the controller aggregates per tenant for showback / chargeback.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_J_PER_WH = 3600.0


@dataclass
class EnergyAccountant:
    """Accumulates per-GPU energy in joules from per-tick (draw, dt) samples."""

    energy_j: dict[int, float] = field(default_factory=dict)

    def add_tick(self, gpu_draws_w: dict[int, float], dt_s: float) -> dict[int, float]:
        """Integrate one tick: add ``draw × dt`` joules per GPU. Non-positive dt
        (a fixed interval is never negative; clock skew on the real path) is
        ignored so the counter is monotonic. Returns the joules added per GPU
        this tick, so callers (e.g. per-tenant metrics) attribute from the same
        source of truth rather than re-integrating."""
        added: dict[int, float] = {}
        if dt_s <= 0:
            return added
        for gpu, draw in gpu_draws_w.items():
            if draw is not None:
                joules = draw * dt_s
                self.energy_j[gpu] = self.energy_j.get(gpu, 0.0) + joules
                added[gpu] = joules
        return added

    def gpu_energy_wh(self, gpu: int) -> float:
        return self.energy_j.get(gpu, 0.0) / _J_PER_WH

    def tenant_energy_wh(self, gpu_indices: list[int]) -> float:
        """Total watt-hours across a tenant's GPUs (missing GPUs contribute 0)."""
        return sum(self.energy_j.get(g, 0.0) for g in gpu_indices) / _J_PER_WH

    def total_wh(self) -> float:
        return sum(self.energy_j.values()) / _J_PER_WH
