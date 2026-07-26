# N12 — Job-aware priority boost

PRS (v2) reclaims stranded power fairly by *draw*, but it has no notion of which
GPUs are running important work. N12 adds a job-aware layer: GPUs with an active
compute job get a configurable cap boost, so under contention a busy GPU keeps a
larger share of the domain budget than an equally-loaded GPU that happens to be
between jobs.

## Components

- **`JobTracker`** (`src/opendps/agent/job_tracker.py`) — determines which GPUs
  are busy.
  - *Real path*: polls `nvidia-smi --query-compute-apps` every few seconds and
    maps compute PIDs → GPU index.
  - *Sim/demo path*: `set_busy_gpus([...])` marks a fixed set busy without a
    driver (used by `--busy-gpus`).
- **`JobAwarePRSBrain`** (`src/opendps/brain/job_aware_prs.py`) — wraps `PRSBrain`:
  1. run PRS to get the base allocation;
  2. multiply each busy GPU's cap by `(1 + priority_boost)`, clamped to its
     hardware max;
  3. renormalise *all* caps so `Σcaps` stays within the domain budget.
  The net effect is a redistribution toward busy GPUs, not extra power. The
  boost is intentionally exempt from the N5 cap-raise rate limiter so a freshly
  scheduled job gets its share immediately.

## Configuration

The boost fraction is configurable (it was previously hardcoded at `0.15`):

- **CLI**: `--priority-boost FRAC` (default `0.15`, e.g. `0.30` = +30%).
- **params.json**: a `priority_boost` key next to `--config` overrides the CLI
  default (the same convention as `ewma_alpha` / `cap_raise_rate_w_per_tick`),
  which is how the operator can drive it from a `PowerPolicy`.
- Negative values are rejected at parse time and at brain construction.

## Demonstration

`scripts/demo.sh` step **DC9** runs the illustrative `deploy/topology-jobdemo.json`
(a deliberately oversubscribed domain — budget set below the sum of per-GPU
maxima — so the boost binds rather than everyone sitting at hardware max) with
`--brain job-prs --busy-gpus 0,1 --priority-boost 0.30`. Several GPUs carry the
same load, but only the two in `--busy-gpus` have an active job:

```text
busy(job) GPU avg cap = ~734 W
no-job   GPU avg cap = ~352 W
```

The check asserts the busy GPUs are capped clearly above the equally-loaded
no-job GPUs.

## Closed-loop dynamics (honest note)

The *per-tick* boost is exactly `1 + priority_boost` (a 30% boost makes a busy
GPU's pre-renormalisation cap 1.3× an identical idle-job GPU). In the closed
loop the steady-state gap is **larger** than 1.3×: a higher cap lets the GPU
draw more, its EWMA rises, and PRS grants it more next tick. This is bounded —
renormalisation keeps `Σcaps` within the domain budget, so it converges to a
higher steady-state share rather than running away. The demo's directional
assertion (busy > no-job) holds at any tick count.

## Limitations

- Kubernetes priority loading requires
  `OPENDPS_PRIORITY_CONFIG_ENABLED=true` and the local node identity in
  `OPENDPS_NODE_NAME`. Without that explicit configuration, the controller
  performs no ConfigMap reads and continues to use its CLI baseline.
- **`priorityClass` tiers** (low/normal/high/critical) are captured in the CRD
  but not yet mapped to distinct boost fractions — every busy GPU gets the same
  boost.
- Boost is GPU-level (busy/idle), not per-job; multiple jobs on one GPU share
  the single boost.

## Resolved GPU assignments

The Kubernetes handoff uses an explicit Pod annotation:

```yaml
metadata:
  annotations:
    opendps.io/gpu-indices: "0,2"
```

The value is a comma-separated list of GPU indices on the Pod's assigned node.
The operator accepts only unique, non-negative decimal indices. A Pod without a
node name, UID, or valid annotation is not published as a resolved assignment.

For every resolved Pod/GPU pair, the operator writes one entry to
`resolved-assignments.json` in the `opendps-job-boosts` ConfigMap. The payload
has `schemaVersion: 1` and an `assignments` array. Each assignment contains:

- `policyUid` and `policyGeneration`
- `podUid`, `nodeName`, and `gpuIndex`
- `priorityClass` and `gpuBoostPct`

Entries are sorted deterministically. The controller reads only assignments
whose `nodeName` matches its configured node, so GPU indices remain node-local.

This annotation is a demo-grade, explicit identity mapping. It makes the
policy-to-device handoff auditable without claiming that a Pod's allocated
devices can always be inferred from the Kubernetes API. Automatic mapping from
device-plugin allocation data or Dynamic Resource Allocation remains future
work.

## Controller reload and fallback

The controller polls the ConfigMap and applies resolved priority tiers without
a process restart. `--gpu-priority-tiers` remains the baseline: dynamic
assignments override the same GPU index, while baseline entries for other GPUs
remain in effect.

A valid empty assignment set or a missing ConfigMap (`404`) restores the CLI
baseline. Malformed data and non-`404` API failures retain the last-known-good
snapshot. This avoids replacing a valid policy with a partial or unreadable
update.

When `OPENDPS_PRIORITY_CONFIG_ENABLED=true` and `OPENDPS_NODE_NAME` identifies
the local node, the CLI baseline may be empty; resolved assignments can
populate the priority brain after startup. Without a dynamic source,
`priority-prs` still requires an explicit CLI tier mapping.

Pod lifecycle events trigger reconciliation of the namespace's
`JobPowerPolicy` objects. This covers Pods that receive the GPU annotation
after policy creation and removes assignments when Pods disappear. Registry
updates use the ConfigMap resource version and retry conflicts, while
preserving assignments owned by sibling policies.
