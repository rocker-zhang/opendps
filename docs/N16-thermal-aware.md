# N16 — Thermal-aware control

The DCGM throttle-violation counters and GPU temperature are collected
(`deploy/dcgm-fields.csv`, Grafana) but no brain ingested them, so the control
loop would happily raise the cap on a GPU that is already thermal-throttling.
That wastes budget: a heat-limited GPU can't clock up no matter how much power
headroom it is given. N16 makes the brain thermal-aware.

## Signal

`DomainState` gains `gpu_thermal_throttled: dict[int, bool]`. It is populated:

- **Real node**: `NodeSampleFromProm` now also queries `DCGM_FI_DEV_GPU_TEMP`;
  a GPU at or above `--thermal-throttle-temp-c` (default 85 °C) counts as
  thermal-throttling.
- **Sim/demo**: `--hot-gpus IDX,IDX` forces a set of GPUs throttled (a sim-only
  override, like `--busy-gpus`).

An absent GPU is treated as not-throttled.

## Algorithm

`ThermalAwarePRSBrain` wraps PRS. For each thermal-throttled GPU it backs the cap
off by `thermal_derate` (default 0.15) down to the `min_cap_w` floor, and hands
the freed watts to non-throttled GPUs in proportion to their remaining headroom
below their hardware max. The domain budget is never exceeded
(freed watts are only redistributed, capped at hardware max — any remainder is
left as headroom).

## Demonstration

`scripts/demo.sh` step **DC12** forces one GPU thermal-throttled under the demo
topology:

```text
thermal-throttled GPU0 = 850 W; non-throttled GPU1 = 1000 W
```

The throttled GPU is derated 15 % below its equally-hot peer; the check asserts
`thermal < non-throttled`.

## Limitations

- Throttle detection on the real path is temperature-threshold based; the
  cumulative `DCGM_FI_DEV_*_VIOLATION` counters are collected for Grafana but not
  yet delta-tracked into the control loop (temperature is the simpler, stateless
  signal).
- The derate is a fixed fraction, not a closed thermal-control loop; a
  PID/setpoint controller against junction temperature is future work.
- In the draw-follows-cap simulator a derated GPU draws a little less over
  subsequent ticks; the directional result (thermal < non-throttled) holds
  regardless.
