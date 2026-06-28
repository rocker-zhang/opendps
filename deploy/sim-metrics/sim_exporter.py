import random
import time

from prometheus_client import Gauge, start_http_server

NUM_GPUS = 4
MODEL_NAME = "NVIDIA-SIM-GPU"
HOSTNAME = "sim-host-0"
PORT = 9401

LABELS = ["gpu", "modelName", "hostname"]

power_usage = Gauge("DCGM_FI_DEV_POWER_USAGE", "Power draw W", LABELS)
# Sim-friendly alias — used by the Grafana dashboard so sim data is visible
# without a real dcgm-exporter providing DCGM_FI_DEV_POWER_USAGE.
sim_power_usage = Gauge("sim_gpu_power_usage_watts", "Power draw W (sim alias)", LABELS)
sm_clock = Gauge("DCGM_FI_DEV_SM_CLOCK", "SM clock MHz", LABELS)
gpu_util = Gauge("DCGM_FI_DEV_GPU_UTIL", "GPU utilization pct", LABELS)
power_cap = Gauge("DCGM_FI_DEV_POWER_CAP", "Power cap W", LABELS)

_state: dict[int, float] = {i: 200.0 + random.uniform(0, 600) for i in range(NUM_GPUS)}

for _i in range(NUM_GPUS):
    _lbl = {"gpu": str(_i), "modelName": MODEL_NAME, "hostname": HOSTNAME}
    power_usage.labels(**_lbl).set(_state[_i])
    sim_power_usage.labels(**_lbl).set(_state[_i])
    sm_clock.labels(**_lbl).set(1980)
    gpu_util.labels(**_lbl).set(random.uniform(30, 95))
    power_cap.labels(**_lbl).set(1000.0)


def _update() -> None:
    for i in range(NUM_GPUS):
        lbl = {"gpu": str(i), "modelName": MODEL_NAME, "hostname": HOSTNAME}
        _state[i] = max(150.0, min(900.0, _state[i] + random.gauss(0, 20)))
        power_usage.labels(**lbl).set(_state[i])
        sim_power_usage.labels(**lbl).set(_state[i])
        sm_clock.labels(**lbl).set(random.randint(1800, 2100))
        gpu_util.labels(**lbl).set(random.uniform(30, 95))


if __name__ == "__main__":
    start_http_server(PORT)
    while True:
        _update()
        time.sleep(5)
