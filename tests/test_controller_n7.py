"""N7 — demo metrics: brain-agnostic stranded-watts computation.

The headline before/after comparison requires that the DPM baseline reports its
stranded watts (it previously hard-coded 0, making the comparison impossible)
and that PRS reports far less under the *same* definition.
"""
from __future__ import annotations

from opendps.controller.standalone import _domain_stats


def test_domain_stats_counts_idle_headroom():
    # GPU 0 hot (900/1000=0.9), GPU 1 idle (100/1000=0.1).
    stats = _domain_stats({0: 900.0, 1: 100.0}, {0: 1000.0, 1: 1000.0})
    assert stats["hot_count"] == 1
    assert stats["idle_count"] == 1
    # Stranded = idle headroom only: (1000 - 100) on GPU 1.
    assert stats["idle_stranded_w"] == 900.0
    assert stats["draw_w"] == 1000.0
    assert stats["cap_w"] == 2000.0


def test_domain_stats_dpm_baseline_is_nonzero():
    """A static DPM allocation (all GPUs at hw max) must show stranded watts on
    the idle GPUs — not 0."""
    draws = {i: (800.0 if i < 6 else 150.0) for i in range(10)}
    caps = {i: 1000.0 for i in range(10)}  # DPM static: everyone at hw max
    stats = _domain_stats(draws, caps)
    assert stats["idle_count"] == 4
    # 4 idle GPUs × (1000 - 150) = 3400 W stranded.
    assert stats["idle_stranded_w"] == 3400.0


def test_domain_stats_handles_zero_cap():
    stats = _domain_stats({0: 0.0}, {0: 0.0})
    assert stats["idle_count"] == 1
    assert stats["idle_stranded_w"] == 0.0


def test_dpm_strands_more_than_prs_in_sim():
    """End-to-end sim: run one tick of DPM and one of PRS over the same
    oversubscribed scenario; DPM must strand materially more than PRS."""
    from opendps.controller.standalone import ControllerConfig, StandaloneController
    from opendps.pdn.presets import demo_single_domain

    topo = demo_single_domain(n_gpus=10, budget_w=8000.0)

    def stranded(brain: str) -> float:
        cfg = ControllerConfig(
            topology=topo,
            actuator=None,  # replaced below
            sim_mode=True,
            brain_type=brain,
            metrics_port=None,
            actuator_type="sim",
        )
        from opendps.sim.presets import oversub_scenario
        cfg.actuator = oversub_scenario(n_gpus=10)
        ctl = StandaloneController(cfg)
        last = 0.0
        for _ in range(8):  # let EWMA converge
            decisions = ctl.run_once()
            draws = {g: cfg.actuator.get_power_draw(g) for g in range(10)}
            caps = decisions[0].caps
            last = _domain_stats(draws, caps)["idle_stranded_w"]
        return last

    dpm = stranded("dpm")
    prs = stranded("prs")
    assert dpm > 1000.0, f"DPM baseline should strand >1000 W, got {dpm:.0f}"
    # Same bound as scripts/demo.sh (>70% reclaim); scenario is deterministic.
    assert prs < dpm * 0.3, f"PRS should reclaim >70% of stranded watts: prs={prs:.0f} dpm={dpm:.0f}"


def test_telemetry_actuator_reads_draws_without_prometheus():
    """--telemetry actuator closes the loop on a real NVML node: draws come from
    the actuator, no PromClient is created, and caps are still pushed."""
    from opendps.controller.standalone import ControllerConfig, StandaloneController
    from opendps.pdn.presets import demo_single_domain
    from opendps.sim.presets import oversub_scenario

    topo = demo_single_domain(n_gpus=10, budget_w=8000.0)
    cfg = ControllerConfig(
        topology=topo,
        actuator=oversub_scenario(n_gpus=10),
        sim_mode=False,            # not sim — but...
        telemetry="actuator",     # ...read draws straight from the actuator
        brain_type="prs",
        metrics_port=None,
        actuator_type="sim",
    )
    ctl = StandaloneController(cfg)
    assert ctl._client is None, "no PromClient should be created for --telemetry actuator"
    decisions = ctl.run_once()
    assert decisions and decisions[0].caps, "controller should produce caps from actuator draws"
