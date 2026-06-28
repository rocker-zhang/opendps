# Hardware validation

The sim path (`scripts/demo.sh`) runs without a GPU. The items below were
exercised on a **cap-capable NVIDIA datacenter GPU node** (per-GPU max power
limit 1100 W, min 200 W) to validate the real NVML control path. Reproduce with
`scripts/hw_failsafe.sh` (failsafe) and the closed-loop command at the bottom.

## 1. Rust failsafe — real NVML cap-lower

`opendps-agent --nvml` with a trip threshold just below idle draw forces the
brain-independent failsafe to lower every GPU's cap.

| Measurement | Result |
|---|---|
| Trips observed | all managed GPUs, repeatedly |
| Cap applied on trip | hardware cap lowered to the configured emergency value, confirmed via `nvidia-smi` |
| **NVML round-trip latency** (detect → `nvmlDeviceSetPowerManagementLimit` confirmed) | **mean ~23 ms, p≈ all < 50 ms** |

**Important honesty note.** The end-to-end failsafe latency is dominated by the
NVML `set_power_management_limit` driver round-trip (~10–50 ms), **not** by the
detection loop. The detection loop itself is sub-millisecond, and that is where
Rust (a dedicated `SCHED_FIFO` thread, no GIL jitter) helps versus a Python
poller. Earlier "tens of microseconds" figures measured the detection loop
against an in-memory mock sink, not the real NVML call — the real cap-apply cost
is set by the driver and is similar regardless of host language. The value of the
Rust agent is fast, deterministic *detection and scheduling*, not a faster NVML
call.

## 2. Python NVML cap round-trip

`NvmlActuator.set_power_cap()` / read-back:

```text
set 900 W -> read back 900 W
set 700 W -> read back 700 W
set 1100 W -> read back 1100 W   (restored)
```

Exact round-trip on real hardware via `pynvml`.

## 3. Closed-loop PRS reclaim on real GPUs

The controller can read draws straight from the NVML actuator (no Prometheus)
with `--telemetry actuator`, so the full loop runs on a bare GPU node:

```bash
opendps-controller --actuator nvml --telemetry actuator --brain prs \
  --config <domain.json> --interval 1
```

With an idle, oversubscribed domain the PRS brain reclaimed the stranded
headroom on real hardware: every idle GPU's cap dropped from 1100 W to ~305 W
(idle EWMA × margin), confirmed by `nvidia-smi`, and the controller reported the
stranded-watts metric falling as EWMA converged. Caps were restored to 1100 W
afterward.
