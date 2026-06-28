"""Actuator protocol — the pluggable enforcement target.

Both the sim backend and the real NVML/dcgmi agent implement this interface so
the control brain can be swapped between simulation and production without any
code changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Actuator(Protocol):
    """Pluggable enforcement target: sim OR real NVML/dcgmi."""

    def set_power_cap(self, gpu_index: int, watts: float) -> None: ...
    def get_power_cap(self, gpu_index: int) -> float: ...
    def get_power_draw(self, gpu_index: int) -> float: ...
    def get_util_pct(self, gpu_index: int) -> float: ...
    def gpu_count(self) -> int: ...
