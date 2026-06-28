"""Tests for brain v1 (DPM) and StandaloneController.

All tests run without a GPU or Prometheus — PromClient is mocked and the
Actuator is a plain MagicMock.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from opendps.brain import DomainState, DPMBrain
from opendps.controller.standalone import ControllerConfig, StandaloneController
from opendps.pdn.model import PDU, PDNTopology, PowerDomain
from opendps.telemetry.model import GpuSample, NodeSample


# ---------------------------------------------------------------------------
# Shared topology helpers
# ---------------------------------------------------------------------------

def _make_topology(n_gpus: int = 4, budget_w: float = 2800.0) -> PDNTopology:
    """Single PDU, single domain with n_gpus GPUs and the given budget."""
    pdu = PDU(name="pdu-A", capacity_w=20_000.0, derating=0.9)
    domain = PowerDomain(
        name="domain-0",
        budget_w=budget_w,
        gpu_indices=list(range(n_gpus)),
        pdu_name="pdu-A",
        priority=1,
    )
    return PDNTopology(pdus={"pdu-A": pdu}, domains={"domain-0": domain})


def _make_state(
    draws: dict[int, float],
    caps: dict[int, float],
    domain_name: str = "domain-0",
    ts: float = 1_000.0,
    max_caps: dict[int, float] | None = None,
) -> DomainState:
    """Build a DomainState.

    If max_caps is not supplied it defaults to a copy of caps, meaning hardware
    max == reported cap.  Pass explicit max_caps in tests that exercise the
    cap-ratchet recovery path.
    """
    return DomainState(
        domain_name=domain_name,
        gpu_draws=draws,
        gpu_caps=caps,
        gpu_max_caps=max_caps if max_caps is not None else dict(caps),
        ts=ts,
    )


# ---------------------------------------------------------------------------
# Test 1 — under budget: caps stay at hardware max
# ---------------------------------------------------------------------------

def test_under_budget_returns_current_caps() -> None:
    """When total draw is within budget the brain must not reduce caps."""
    topo = _make_topology(n_gpus=4, budget_w=4000.0)
    brain = DPMBrain(topo)

    # 4 GPUs × 300 W draw = 1200 W total; budget 4000 W → comfortably under.
    draws = {i: 300.0 for i in range(4)}
    caps  = {i: 800.0 for i in range(4)}
    state = _make_state(draws, caps)

    decision = brain.decide("domain-0", state)

    # Caps must be unchanged (equal to current hardware max).
    assert decision.caps == {i: 800.0 for i in range(4)}, (
        "Under-budget decision should keep caps at hardware max"
    )


# ---------------------------------------------------------------------------
# Test 2 — over budget: proportional reduction, total ≤ budget
# ---------------------------------------------------------------------------

def test_over_budget_proportional_reduction_total_le_budget() -> None:
    """Brain must scale caps so the sum does not exceed the domain budget."""
    # Two GPUs with unequal draws.  GPU 0 is hotter so it should keep more headroom.
    draws = {0: 600.0, 1: 400.0}   # total 1000 W
    caps  = {0: 800.0, 1: 800.0}
    topo  = _make_topology(n_gpus=2, budget_w=800.0)  # total < 1000 → over budget
    brain = DPMBrain(topo)
    state = _make_state(draws, caps)

    decision = brain.decide("domain-0", state)

    total_caps = sum(decision.caps.values())
    assert total_caps <= 800.0 + 1e-9, (
        f"Total caps {total_caps:.3f} W must not exceed domain budget 800 W"
    )
    # GPU 0 should hold proportionally more (it draws more).
    assert decision.caps[0] > decision.caps[1], (
        "Hotter GPU should receive proportionally more headroom"
    )
    # Verify exact proportional math: 600/1000 × 800 = 480, 400/1000 × 800 = 320.
    assert abs(decision.caps[0] - 480.0) < 1e-9
    assert abs(decision.caps[1] - 320.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 3 — never below min_cap_w
# ---------------------------------------------------------------------------

def test_never_below_min_cap_w() -> None:
    """Proportional allocation must be floored at min_cap_w."""
    # GPU 1 draws very little; proportional share would be far below min floor.
    draws = {0: 950.0, 1: 50.0}   # total 1000 W
    caps  = {0: 1000.0, 1: 1000.0}
    topo  = _make_topology(n_gpus=2, budget_w=600.0)  # total 1000 > 600 → over budget
    brain = DPMBrain(topo, min_cap_w=200.0)
    state = _make_state(draws, caps)

    decision = brain.decide("domain-0", state)

    for gpu_idx, cap in decision.caps.items():
        assert cap >= 200.0, (
            f"GPU {gpu_idx} cap {cap:.1f} W is below min_cap_w=200 W"
        )


# ---------------------------------------------------------------------------
# Test 4 — reason field reflects budget status
# ---------------------------------------------------------------------------

def test_reason_is_over_budget_when_over() -> None:
    draws = {0: 900.0, 1: 900.0}   # total 1800 W
    caps  = {0: 800.0, 1: 800.0}
    topo  = _make_topology(n_gpus=2, budget_w=1000.0)
    brain = DPMBrain(topo)
    state = _make_state(draws, caps)

    decision = brain.decide("domain-0", state)

    assert decision.reason == "over_budget"


def test_reason_is_under_budget_when_under() -> None:
    draws = {0: 300.0, 1: 300.0}   # total 600 W
    caps  = {0: 800.0, 1: 800.0}
    topo  = _make_topology(n_gpus=2, budget_w=1000.0)
    brain = DPMBrain(topo)
    state = _make_state(draws, caps)

    decision = brain.decide("domain-0", state)

    assert decision.reason == "under_budget"


# ---------------------------------------------------------------------------
# Test 5 — StandaloneController.run_once() with mocked PromClient
# ---------------------------------------------------------------------------

def test_run_once_correct_decisions_mocked_prom(capsys: pytest.CaptureFixture) -> None:
    """
    Scenario: 4 GPUs, each drawing 900 W against an 800 W cap.
    Domain budget = 2800 W.  Total draw (3600 W) > budget (2800 W).

    Expected proportional caps:
        proportion = 900 / 3600 = 0.25 for each GPU
        new_cap    = 0.25 × 2800 = 700 W  (no clamping needed)

    The mock Actuator must receive set_power_cap(i, 700.0) for i in 0..3.
    """
    # Build topology: 4 GPUs, budget 2800 W.
    topology = _make_topology(n_gpus=4, budget_w=2800.0)

    # Build the fake NodeSample PromClient will "return".
    fake_sample = NodeSample(
        ts=42_000.0,
        hostname="sim-host",
        driver_version="dcgm",
        gpus=[
            GpuSample(index=i, name="SimGPU", power_draw_w=900.0, power_limit_w=800.0)
            for i in range(4)
        ],
    )

    # Mock actuator: captures set_power_cap calls.
    mock_actuator = MagicMock()
    mock_actuator.gpu_count.return_value = 4

    cfg = ControllerConfig(
        topology=topology,
        actuator=mock_actuator,
        prom_url="http://mock-prom:9090",
        dry_run=False,
        brain_type="dpm",  # this test verifies DPM reason strings specifically
    )

    with patch("opendps.controller.standalone.NodeSampleFromProm", return_value=fake_sample):
        controller = StandaloneController(cfg)
        decisions = controller.run_once()

    # One decision for the one managed domain.
    assert len(decisions) == 1
    decision = decisions[0]

    # Domain and timestamp must match.
    assert decision.domain == "domain-0"
    assert decision.ts == pytest.approx(42_000.0)

    # Reason: over budget.
    assert decision.reason == "over_budget"

    # Proportional caps: each GPU should receive 700 W.
    expected_cap = 700.0  # = (900 / 3600) × 2800
    assert len(decision.caps) == 4
    for gpu_idx, cap in decision.caps.items():
        assert cap == pytest.approx(expected_cap, rel=1e-6), (
            f"GPU {gpu_idx} cap {cap:.3f} W != expected {expected_cap} W"
        )

    # Total caps must equal budget exactly (no clamping in this scenario).
    total = sum(decision.caps.values())
    assert total == pytest.approx(2800.0, rel=1e-6)

    # Actuator must have been called once per GPU with the proportional cap.
    assert mock_actuator.set_power_cap.call_count == 4
    for i in range(4):
        mock_actuator.set_power_cap.assert_any_call(i, pytest.approx(expected_cap, rel=1e-6))


# ---------------------------------------------------------------------------
# Test 6 — cap-ratchet recovery: two-tick test
# ---------------------------------------------------------------------------

def test_cap_ratchet_recovers_to_hardware_max() -> None:
    """Caps must recover toward hardware max on an under-budget tick.

    Regression for the cap-ratchet bug: previously the under-budget path
    returned state.gpu_caps (the last reported cap, which was already reduced),
    causing caps to stick at the reduced value forever.

    Two-tick scenario:
      Tick 1: 4 GPUs × 800 W draw, budget 2800 W → OVER budget.
              Brain reduces caps proportionally to 700 W each.
      Tick 2: Simulate the controller applying those caps; load drops to 200 W
              per GPU → total 800 W, well UNDER budget.
              Brain must restore caps to hardware max (1000 W), NOT keep them
              at the reduced 700 W.
    """
    topo = _make_topology(n_gpus=4, budget_w=2800.0)
    brain = DPMBrain(topo)
    hardware_max_w = 1000.0

    # --- Tick 1: over budget ---
    draws1 = {i: 800.0 for i in range(4)}   # total 3200 W > 2800 W budget
    caps1  = {i: hardware_max_w for i in range(4)}
    max_caps = {i: hardware_max_w for i in range(4)}
    state1 = _make_state(draws1, caps1, max_caps=max_caps)

    decision1 = brain.decide("domain-0", state1)
    assert decision1.reason == "over_budget", "Tick 1 should be over budget"
    for cap in decision1.caps.values():
        assert cap < hardware_max_w, "Tick 1 should reduce caps below hardware max"

    # --- Tick 2: under budget — controller applied reduced caps from tick 1 ---
    draws2 = {i: 200.0 for i in range(4)}   # total 800 W < 2800 W budget
    caps2  = decision1.caps                  # reported caps are now the reduced values
    state2 = _make_state(draws2, caps2, max_caps=max_caps)

    decision2 = brain.decide("domain-0", state2)
    assert decision2.reason == "under_budget", "Tick 2 should be under budget"

    for gpu, cap in decision2.caps.items():
        assert cap == pytest.approx(hardware_max_w), (
            f"GPU {gpu} cap {cap:.1f} W did not recover to hardware max "
            f"{hardware_max_w} W — cap-ratchet bug still present"
        )
