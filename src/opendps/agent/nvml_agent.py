"""NvmlActuator — real GPU cap enforcement via pynvml.

Implements the Actuator Protocol using the NVIDIA Management Library.
All power values cross the API boundary in watts; NVML uses milliwatts internally.

Usage on a cap-capable GPU node:
    actuator = NvmlActuator()
    actuator.set_power_cap(0, 800.0)  # cap GPU 0 to 800 W
    draw = actuator.get_power_draw(0)
    actuator.shutdown()

This requires NVML to be available (libcuda.so / nvidia-ml.so on the node).
On nodes without NVML (e.g. GB10 dev boards), import succeeds but instantiation
raises NVMLError — callers should fall back to --sim mode.
"""

from __future__ import annotations

import logging

import pynvml

log = logging.getLogger(__name__)


class NvmlActuator:
    """Actuator implementation backed by NVML / pynvml.

    Implements the five-method Actuator Protocol plus helpers used by the
    standalone controller (get_max_cap_w) and diagnostics (get_name).
    """

    def __init__(self) -> None:
        pynvml.nvmlInit()
        self._count: int = pynvml.nvmlDeviceGetCount()
        self._handles: list = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self._count)
        ]

    # ------------------------------------------------------------------
    # Actuator Protocol
    # ------------------------------------------------------------------

    def set_power_cap(self, gpu_index: int, watts: float) -> None:
        """Set the power cap for gpu_index to watts (W).

        NVML enforces a [min_cap, max_cap] range per GPU; values outside that
        range raise NVMLError_InvalidArgument.  Callers should clamp before
        calling (the brain already does this via min_cap_w / gpu_max_caps).
        """
        try:
            pynvml.nvmlDeviceSetPowerManagementLimit(
                self._handles[gpu_index], int(watts * 1000)
            )
        except pynvml.NVMLError as exc:
            log.error("set_power_cap gpu=%d watts=%.1f failed: %s", gpu_index, watts, exc)

    def get_power_cap(self, gpu_index: int) -> float:
        try:
            return pynvml.nvmlDeviceGetPowerManagementLimit(self._handles[gpu_index]) / 1000.0
        except pynvml.NVMLError as exc:
            log.error("get_power_cap gpu=%d failed: %s", gpu_index, exc)
            return 0.0

    def get_power_draw(self, gpu_index: int) -> float:
        try:
            return pynvml.nvmlDeviceGetPowerUsage(self._handles[gpu_index]) / 1000.0
        except pynvml.NVMLError as exc:
            log.error("get_power_draw gpu=%d failed: %s", gpu_index, exc)
            return 0.0

    def get_util_pct(self, gpu_index: int) -> float:
        try:
            return float(
                pynvml.nvmlDeviceGetUtilizationRates(self._handles[gpu_index]).gpu
            )
        except pynvml.NVMLError as exc:
            log.error("get_util_pct gpu=%d failed: %s", gpu_index, exc)
            return 0.0

    def gpu_count(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # Extended helpers (not in base Protocol)
    # ------------------------------------------------------------------

    def get_max_cap_w(self, gpu_index: int) -> float:
        """Return the hardware-enforced maximum power cap (W)."""
        try:
            _, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(
                self._handles[gpu_index]
            )
            return max_mw / 1000.0
        except pynvml.NVMLError as exc:
            log.error("get_max_cap_w gpu=%d failed: %s", gpu_index, exc)
            return 0.0

    def get_name(self, gpu_index: int) -> str:
        try:
            raw = pynvml.nvmlDeviceGetName(self._handles[gpu_index])
            return raw.decode() if isinstance(raw, bytes) else str(raw)
        except pynvml.NVMLError as exc:
            log.error("get_name gpu=%d failed: %s", gpu_index, exc)
            return f"gpu-{gpu_index}"

    def shutdown(self) -> None:
        """Release NVML resources.  Call once when the process exits."""
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
