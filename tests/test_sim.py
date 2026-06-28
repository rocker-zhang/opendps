"""Sim backend tests — no GPU hardware required.

All five tests exercise the sim in isolation; they run on any workstation.
"""

from __future__ import annotations

import random


from opendps.sim import Actuator, LoadProfile, SimBackend, SimGpu
from opendps.sim.presets import oversub_scenario, uniform_load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(
    n: int = 1,
    cap_w: float = 1000.0,
    util_pct: float = 90.0,
    seed: int = 0,
) -> SimBackend:
    """Build a minimal SimBackend with deterministic, noise-free utilisation."""
    gpus = [
        SimGpu(
            index=i,
            cap_w=cap_w,
            max_cap_w=cap_w,
            load=LoadProfile(base_util_pct=util_pct, util_noise=0.0),
            _rng=random.Random(seed + i),
        )
        for i in range(n)
    ]
    return SimBackend(gpus)


# ---------------------------------------------------------------------------
# Test 1 — set_power_cap clamps power draw
# ---------------------------------------------------------------------------


def test_set_power_cap_clamps_power_draw():
    """After set_power_cap(100 W), power_draw must never exceed 100 W.

    At 90 % util without capping: 1000*(0.9)*0.85 + 50 = 815 W.
    After capping to 100 W: draw = min(100*(0.9)*0.85 + 50, 100) = min(126.5, 100) = 100.
    """
    backend = _make_backend(n=1, cap_w=1000.0, util_pct=90.0)
    backend.set_power_cap(0, 100.0)
    assert backend.get_power_draw(0) <= 100.0


# ---------------------------------------------------------------------------
# Test 2 — oversub_scenario hot vs idle GPU count
# ---------------------------------------------------------------------------


def test_oversub_scenario_hot_vs_idle_count():
    """oversub_scenario must produce exactly round(n*hot_fraction) hot GPUs."""
    n_gpus = 10
    hot_fraction = 0.6
    backend = oversub_scenario(n_gpus=n_gpus, cap_w=1000.0, hot_fraction=hot_fraction)

    assert backend.gpu_count() == n_gpus

    n_hot_expected = round(n_gpus * hot_fraction)   # 6
    n_idle_expected = n_gpus - n_hot_expected         # 4

    snap = backend.snapshot()
    # Hot GPUs are configured at 90 % base util; idle at 10 % — split at 50 %.
    hot = [s for s in snap if s["util_pct"] >= 50.0]
    idle = [s for s in snap if s["util_pct"] < 50.0]
    assert len(hot) == n_hot_expected, f"expected {n_hot_expected} hot GPUs, got {len(hot)}"
    assert len(idle) == n_idle_expected, f"expected {n_idle_expected} idle GPUs, got {len(idle)}"


# ---------------------------------------------------------------------------
# Test 3 — tick() changes power draw
# ---------------------------------------------------------------------------


def test_tick_changes_power_draw():
    """tick() must resample utilisation, producing different power draw values.

    With non-zero util_noise, after enough ticks the draw must differ from
    the initial reading on at least one GPU.
    """
    backend = oversub_scenario(n_gpus=4, seed=99)
    draws_before = [backend.get_power_draw(i) for i in range(4)]

    changed = False
    for _ in range(30):
        backend.tick()
        draws_after = [backend.get_power_draw(i) for i in range(4)]
        if draws_after != draws_before:
            changed = True
            break

    assert changed, "power draw should change after repeated tick() calls (util_noise > 0)"


# ---------------------------------------------------------------------------
# Test 4 — SimBackend satisfies the Actuator protocol
# ---------------------------------------------------------------------------


def test_sim_backend_satisfies_actuator_protocol():
    """SimBackend must pass an isinstance check against the runtime_checkable Actuator."""
    backend = oversub_scenario()
    assert isinstance(backend, Actuator), (
        "SimBackend does not satisfy the Actuator protocol — "
        "check that all five methods are present with the correct names"
    )


# ---------------------------------------------------------------------------
# Test 5 — snapshot() returns all GPU indices
# ---------------------------------------------------------------------------


def test_snapshot_returns_all_gpu_indices():
    """snapshot() must return one entry per GPU, covering every index 0..n-1."""
    n = 7
    backend = uniform_load(n_gpus=n)
    snap = backend.snapshot()

    assert len(snap) == n, f"expected {n} entries in snapshot, got {len(snap)}"
    indices = {s["index"] for s in snap}
    assert indices == set(range(n)), f"missing GPU indices: {set(range(n)) - indices}"
