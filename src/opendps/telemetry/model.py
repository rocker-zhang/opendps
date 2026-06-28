"""Telemetry sample data model.

Kept free of any NVML import so the serialization logic can be unit-tested on a
host without an NVIDIA driver.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class GpuSample:
    """A single point-in-time reading for one GPU.

    Power is in watts, clocks in MHz, temperature in Celsius, utilization in
    percent, energy in joules. Any field the device does not expose is ``None``
    rather than a sentinel, so consumers can tell "unsupported" from "zero".
    """

    index: int
    name: str
    power_draw_w: float | None = None
    power_instant_w: float | None = None
    power_limit_w: float | None = None
    power_min_limit_w: float | None = None
    power_max_limit_w: float | None = None
    sm_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None
    temperature_c: int | None = None
    gpu_util_pct: int | None = None
    mem_util_pct: int | None = None
    energy_j: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class NodeSample:
    """All GPUs on one node at one timestamp."""

    ts: float
    hostname: str
    driver_version: str
    gpus: list[GpuSample] = field(default_factory=list)

    @property
    def total_power_draw_w(self) -> float:
        return sum(g.power_draw_w or 0.0 for g in self.gpus)

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "hostname": self.hostname,
            "driver_version": self.driver_version,
            "total_power_draw_w": self.total_power_draw_w,
            "gpus": [g.to_dict() for g in self.gpus],
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))
