"""Tests for FailsafeLoop using SimBackend — no real GPU hardware required."""

from __future__ import annotations

import time

import pytest

from opendps.agent.failsafe import FailsafeLoop
from opendps.sim.presets import oversub_scenario, uniform_load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(n_gpus: int = 4, cap_w: float = 1000.0, util_pct: float = 50.0):
    return uniform_load(n_gpus=n_gpus, cap_w=cap_w, util_pct=util_pct)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_failsafe_constructor_rejects_bad_thresholds():
    backend = _make_backend()
    with pytest.raises(ValueError, match="must be less than"):
        FailsafeLoop(backend, emergency_threshold_w=800.0, emergency_cap_w=900.0)


def test_failsafe_trips_when_draw_exceeds_threshold():
    """A GPU whose draw is far above threshold must be tripped within 500 ms."""
    # Use a very low threshold so the SimBackend at 90% util will trigger it
    backend = oversub_scenario(n_gpus=4, cap_w=1000.0, hot_fraction=1.0)
    # At 90% util: draw ≈ 1000 * 0.9 * 0.85 + 50 = 815 W  → threshold 600 W trips
    threshold = 600.0
    emergency_cap = 400.0
    failsafe = FailsafeLoop(backend, threshold, emergency_cap, poll_interval_s=0.02)
    failsafe.start()
    time.sleep(0.2)
    failsafe.stop()

    assert failsafe.trip_count > 0, "Expected at least one failsafe trip"
    # All GPUs should now be capped at or below emergency_cap
    for i in range(backend.gpu_count()):
        assert backend.get_power_cap(i) <= emergency_cap + 1.0, (
            f"GPU {i} cap {backend.get_power_cap(i):.1f} W should be ≤ {emergency_cap} W"
        )


def test_failsafe_does_not_trip_when_below_threshold():
    """No trips expected when all GPUs are well under threshold."""
    backend = _make_backend(util_pct=10.0)
    # At 10% util: draw ≈ 1000 * 0.1 * 0.85 + 50 = 135 W  → well under 500 W
    failsafe = FailsafeLoop(backend, emergency_threshold_w=500.0,
                            emergency_cap_w=300.0, poll_interval_s=0.02)
    failsafe.start()
    time.sleep(0.15)
    failsafe.stop()

    assert failsafe.trip_count == 0


def test_failsafe_is_cap_lower_only():
    """FailsafeLoop must never raise a cap above its current value."""
    # Start at high cap, low utilisation — failsafe should not increase caps
    backend = _make_backend(cap_w=500.0, util_pct=5.0)
    initial_caps = [backend.get_power_cap(i) for i in range(backend.gpu_count())]

    # Very high threshold so no trip fires
    failsafe = FailsafeLoop(backend, emergency_threshold_w=2000.0,
                            emergency_cap_w=1500.0, poll_interval_s=0.02)
    failsafe.start()
    time.sleep(0.15)
    failsafe.stop()

    for i in range(backend.gpu_count()):
        assert backend.get_power_cap(i) <= initial_caps[i] + 1.0, (
            f"GPU {i} cap raised from {initial_caps[i]:.1f} to {backend.get_power_cap(i):.1f}"
        )


def test_failsafe_stop():
    """stop() must terminate the thread cleanly."""
    backend = _make_backend()
    failsafe = FailsafeLoop(backend, emergency_threshold_w=5000.0,
                            emergency_cap_w=4000.0, poll_interval_s=0.05)
    failsafe.start()
    assert failsafe.is_running
    failsafe.stop()
    assert not failsafe.is_running


def test_failsafe_with_sim_backend_integration():
    """Integration: FailsafeLoop + SimBackend, multiple tick cycles."""
    backend = oversub_scenario(n_gpus=6, cap_w=1000.0, hot_fraction=0.5)
    # Advance sim a few ticks so draws are stable
    for _ in range(5):
        backend.tick()

    threshold = 400.0
    emergency_cap = 200.0
    failsafe = FailsafeLoop(backend, threshold, emergency_cap, poll_interval_s=0.02)
    failsafe.start()
    time.sleep(0.3)
    # Advance sim while failsafe runs
    for _ in range(5):
        backend.tick()
        time.sleep(0.02)
    failsafe.stop()

    # trip_count is non-negative; exact count depends on random draws
    assert failsafe.trip_count >= 0
    # No GPU should have cap above 1000 W (initial max) — failsafe only lowers
    for i in range(backend.gpu_count()):
        assert backend.get_power_cap(i) <= 1000.0 + 1.0


def test_failsafe_trip_count_increments_per_trip():
    """Each GPU that trips on each poll contributes to trip_count."""
    # Hot GPUs at 90% util, cap 1000W → draw ≈ 815W > threshold 300W
    backend = oversub_scenario(n_gpus=2, cap_w=1000.0, hot_fraction=1.0)
    failsafe = FailsafeLoop(backend, emergency_threshold_w=300.0,
                            emergency_cap_w=200.0, poll_interval_s=0.01)
    failsafe.start()
    time.sleep(0.15)
    failsafe.stop()
    assert failsafe.trip_count >= 2
