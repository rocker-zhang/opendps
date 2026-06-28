import os
import random
import time

from prometheus_client import Gauge, start_http_server

# Synthetic GPU telemetry for the demo. A fraction of the GPUs run hot and the
# rest idle, so idle headroom shows up as stranded power that PRS can reclaim.
# Everything is env-driven; defaults are illustrative, not a real topology.


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    """Parse a bounded positive int from the environment, falling back on the
    default for missing/invalid/out-of-range values (no import-time crash)."""
    try:
        val = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, val))


NUM_GPUS = _env_int("SIM_NUM_GPUS", 8, 1, 4096)
HOT_GPUS = _env_int("SIM_HOT_GPUS", max(1, NUM_GPUS // 2), 0, NUM_GPUS)
MODEL_NAME = os.getenv("SIM_MODEL_NAME", "SIM-GPU")
HOSTNAME = os.getenv("SIM_HOSTNAME", "sim-node")
PORT = int(os.getenv("SIM_PORT", "9401"))

LABELS = ["gpu", "modelName", "hostname"]

power_usage = Gauge("DCGM_FI_DEV_POWER_USAGE", "Power draw W", LABELS)
# Sim-friendly alias — used by the Grafana dashboard so sim data is visible
# without a real dcgm-exporter providing DCGM_FI_DEV_POWER_USAGE.
sim_power_usage = Gauge("sim_gpu_power_usage_watts", "Power draw W (sim alias)", LABELS)
sm_clock = Gauge("DCGM_FI_DEV_SM_CLOCK", "SM clock MHz", LABELS)
gpu_util = Gauge("DCGM_FI_DEV_GPU_UTIL", "GPU utilization pct", LABELS)
power_cap = Gauge("DCGM_FI_DEV_POWER_CAP", "Power cap W", LABELS)


def _is_hot(i: int) -> bool:
    return i < HOT_GPUS


def _band(i: int) -> tuple[float, float]:
    """(low, high) draw band for GPU i."""
    return (650.0, 850.0) if _is_hot(i) else (120.0, 220.0)


_state: dict[int, float] = {}
for _i in range(NUM_GPUS):
    _lo, _hi = _band(_i)
    _state[_i] = random.uniform(_lo, _hi)


def _update() -> None:
    for i in range(NUM_GPUS):
        lbl = {"gpu": str(i), "modelName": MODEL_NAME, "hostname": HOSTNAME}
        lo, hi = _band(i)
        _state[i] = max(lo, min(hi, _state[i] + random.gauss(0, 15)))
        power_usage.labels(**lbl).set(_state[i])
        sim_power_usage.labels(**lbl).set(_state[i])
        sm_clock.labels(**lbl).set(random.randint(1800, 2100) if _is_hot(i) else random.randint(300, 600))
        gpu_util.labels(**lbl).set(random.uniform(70, 99) if _is_hot(i) else random.uniform(0, 15))
        power_cap.labels(**lbl).set(1000.0)


if __name__ == "__main__":
    _update()  # seed all series before first scrape
    start_http_server(PORT)
    while True:
        _update()
        time.sleep(5)
