# N13 — Per-tenant power quota enforcement

When several teams share one power domain, EWMA reclaim alone (PRS, v2) is
*fair by draw* but not *bounded by ownership*: a single busy tenant can consume
the whole domain budget. N13 adds a hard per-tenant ceiling on top of PRS so
each tenant can reclaim freely **within its own slice** but never starves the
others.

## Model

`src/opendps/pdn/quota.py`:

- **`TenantQuota`** — `tenant_id`, `domain_name`, `gpu_indices: list[int]`, and
  `max_watts_pct ∈ (0, 1]`. The percentage is a *fraction of the domain budget*,
  not an absolute watt figure, so the same quota survives a budget change.
- **`QuotaConfig`** — a domain's set of tenants, with:
  - `validate()` — rejects total quota > 100% and any GPU assigned to more than
    one tenant.
  - `tenant_budget_w(tenant, domain_budget_w)` — fraction → watts.
  - `from_dict(data)` — parse + validate from JSON; malformed input raises
    `ValueError` rather than silently degrading.

## Algorithm

`QuotaAwarePRSBrain` (`src/opendps/brain/quota_prs.py`) composes one independent
`PRSBrain` per tenant rather than subclassing it:

1. For each tenant in the domain, build a virtual sub-domain whose budget is
   `max_watts_pct × domain_budget_w` and whose GPUs are the tenant's GPUs that
   are present in this tick's telemetry.
2. Run that tenant's own `PRSBrain` over the sub-domain. Each tenant keeps its
   own EWMA history, so one tenant's idle/busy transitions never perturb
   another's reclaim.
3. GPUs in no tenant share the *remaining* domain budget equally (static), so an
   incompletely-specified quota still produces a safe allocation.

Each tenant's caps are renormalised down to its slice after PRS runs, so the sum
of caps stays within the domain budget — the hard guarantee N13 exists to
provide. The one exception is a physically infeasible slice: PRS will not cap a
GPU below the hardware/`min_cap_w` floor, so if `min_cap_w × (tenant GPUs)`
exceeds the tenant budget the floor wins and that tenant slightly overshoots its
slice. Size slices above that floor to keep the guarantee tight.

## Configuration

Selected with `--brain quota-prs`. The quota is loaded from `--quota-config
FILE`, or a `quota.json` sitting next to `--config` (same convention as
`params.json`). A required-but-missing or malformed file is a hard CLI error,
never a silent fall-back to no enforcement.

```json
{
  "domain_name": "domain0",
  "tenants": [
    {"tenant_id": "tenant-a", "gpu_indices": [0, 1, 2, 3, 4, 5], "max_watts_pct": 0.6},
    {"tenant_id": "tenant-b", "gpu_indices": [6, 7, 8, 9], "max_watts_pct": 0.4}
  ]
}
```

At startup the controller cross-checks the quota against the live topology
(`_validate_quota_against_topology`): the domain must exist and every tenant GPU
must belong to it, so a typo'd index fails loudly instead of under-allocating a
tenant. The k8s `PowerPolicy` CRD already accepts `quota-prs` in its `brain`
enum.

## Demonstration

`scripts/demo.sh` step **DC8** runs the demo topology (`domain0`, 8000 W, GPUs
0–9) with the quota above. In the oversubscribed scenario tenant-a's GPUs are
busy and tenant-b's are idle:

```text
tenant-a caps = 4800 W (60% slice = 4800)   # pinned to its slice — enforcement binds
tenant-b caps =  800 W (40% slice = 3200)   # idle, reclaimed below its ceiling
```

tenant-a, busy, is held exactly at its 60% slice; tenant-b, idle, is reclaimed
far below its 40% ceiling. The check asserts each tenant's cap sum stays within
its slice (`Σcaps_a ≤ 4800 W`, `Σcaps_b ≤ 3200 W`).

## Edge cases & limitations

Handled:

- Over-100% / overlapping-GPU quotas rejected at load *and* at brain
  construction.
- A tenant GPU absent from a tick's telemetry is skipped, not crashed on.
- `gpu_caps` lagging `gpu_max_caps` falls back to the max instead of `KeyError`.

Known limitations (not yet wired):

- The k8s operator accepts the `quota-prs` brain but does not yet surface tenant
  definitions as a CRD; quotas are supplied via the controller config file.
- Unassigned GPUs use a static equal split rather than their own PRS instance.
- Per-tenant EWMA state does not migrate if a GPU is reassigned between tenants
  at runtime.
