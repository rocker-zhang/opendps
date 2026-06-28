"""Serialization tests for the telemetry data model (no GPU required)."""

import json

from opendps.collector.sample import GpuSample, NodeSample


def test_gpu_sample_preserves_none_vs_zero():
    g = GpuSample(index=0, name="GB10", power_draw_w=5.5, power_limit_w=None,
                  gpu_util_pct=0)
    d = g.to_dict()
    assert d["power_limit_w"] is None  # unsupported field stays None
    assert d["gpu_util_pct"] == 0      # genuine zero is preserved


def test_node_sample_total_power_ignores_none():
    node = NodeSample(
        ts=1.0, hostname="h", driver_version="580.142",
        gpus=[
            GpuSample(index=0, name="B300", power_draw_w=233.0),
            GpuSample(index=1, name="B300", power_draw_w=None),
        ],
    )
    assert node.total_power_draw_w == 233.0


def test_node_sample_jsonl_roundtrips():
    node = NodeSample(
        ts=1.5, hostname="box", driver_version="580.142",
        gpus=[GpuSample(index=0, name="GB200", power_draw_w=166.3,
                        power_max_limit_w=1200.0)],
    )
    parsed = json.loads(node.to_jsonl())
    assert parsed["total_power_draw_w"] == 166.3
    assert parsed["gpus"][0]["power_max_limit_w"] == 1200.0
    assert parsed["hostname"] == "box"
