# N20 — Energy accounting & per-tenant showback

opendps exported only instantaneous power gauges — there was no cumulative
energy (watt-hour) accounting and no way to attribute energy to a tenant for
showback / chargeback. N20 adds both.

## Accounting

`EnergyAccountant` (`src/opendps/telemetry/energy.py`) integrates power over
time: each tick it adds `draw × dt` joules per GPU. A non-positive `dt` (clock
skew on the real path; a fixed interval is never negative) is ignored so the
counter stays monotonic.

The controller calls it each tick with the live `gpu_draws` and the tick
interval as `dt`. Using the configured interval as `dt` keeps accounting
deterministic and independent of wall-clock jitter between ticks.

## Per-tenant attribution & metrics

Tenants are taken from the existing `QuotaConfig` (tenant → GPU indices). When a
quota config is present, the controller attributes each tick's energy to its
tenants and increments a Prometheus counter:

```text
opendps_tenant_energy_kwh_total{domain="...", tenant="..."}
```

`StandaloneController.energy_showback()` returns the cumulative per-tenant kWh as
a dict for a CLI/report. With no quota config, accounting still runs per GPU but
the showback is empty (no tenants to attribute to).

## Validation

`tests/test_energy_n20.py`: `draw × dt` integration, per-tenant aggregation,
non-positive-dt and `None`-draw handling, missing-GPU = 0, and an end-to-end
controller run where a busy tenant accrues more energy than an idle one.

## Limitations

- Energy is synthesised from `draw × dt`, not read from the DCGM
  `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` counter; the synthesised value avoids
  counter resets/gaps and works in sim, but a hardware-counter cross-check is
  future work.
- Attribution is per GPU→tenant; sub-GPU (MIG) attribution is out of scope.
