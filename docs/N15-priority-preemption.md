# N15 — SLA-tiered priority preemption

PRS reclaims power fairly by draw, and N12 boosts GPUs with active jobs, but
neither knows how *important* a workload is. Under power pressure a busy
low-priority GPU competes on equal footing with a busy high-priority one. N15
adds SLA tiers so that, under contention, higher-tier GPUs keep more cap —
effectively preempting power from lower-tier neighbours — while no GPU is
starved and the domain budget is never exceeded.

## Tiers

The four tiers match the `JobPowerPolicy.priorityClass` enum
(low / normal / high / critical), each with a relative power weight
(`src/opendps/brain/priority_prs.py`):

| tier | weight |
|---|---|
| low | 0.5 |
| normal | 1.0 (default for unmapped GPUs) |
| high | 2.0 |
| critical | 4.0 |

## Algorithm

`PriorityTieredPRSBrain` wraps `PRSBrain`:

1. Run PRS to get a base allocation (idle floors + proportional hot caps).
2. Classify **contended** GPUs — those drawing at or above
   `contention_threshold` (default 0.6) of their current cap. Tier arbitration
   only applies when ≥ 2 GPUs contend; otherwise the PRS result stands.
3. Reserve the uncontended (idle) caps and a per-GPU floor (`min_cap_w`, default
   200 W) for every contended GPU — so nothing is starved.
4. Split the remaining **surplus** in proportion to `draw × tier_weight`, capped
   at each GPU's hardware max. Ceiling-capped surplus is left as headroom.

By construction `Σ(caps) ≤ domain budget`. Idle GPUs keep their PRS floor — tier
never boosts an idle GPU.

## Configuration

Selected with `--brain priority-prs`; the GPU→tier map is supplied as JSON:

```bash
opendps-controller --brain priority-prs --config <topology.json> \
  --gpu-priority-tiers '{"0":"critical","1":"low","2":"normal"}'
```

Unmapped GPUs default to `normal`; an unknown tier is rejected at brain
construction, and `--gpu-priority-tiers` is only valid with `--brain
priority-prs`. In k8s the same tiers come from `JobPowerPolicy.priorityClass`
(resolved by the operator into the `opendps-job-boosts` ConfigMap). With the
Kubernetes priority source enabled, the controller polls and applies those
node-local assignments between control ticks.

## Demonstration

`scripts/demo.sh` step **DC11** runs the tight-budget demo topology with GPUs
0/1/2 tagged critical/low/normal under equal load:

```text
critical GPU0 = 1000 W; normal GPU2 = ~320 W; low GPU1 = ~250 W
```

The check asserts `critical > normal > low`.

## Limitations

- Process mode uses CLI/JSON tiers by default. Kubernetes assignment loading
  requires `OPENDPS_PRIORITY_CONFIG_ENABLED=true` and the local node identity
  in `OPENDPS_NODE_NAME`; the CLI mapping remains the baseline when both
  sources are present.
- Preemption is expressed through cap weighting, not hard job suspension — a
  low-tier GPU keeps its floor, it is not driven to zero.
- Tier is a per-GPU attribute here; per-job tiering on a shared GPU is future
  work.

## Kubernetes priority handoff

`JobPowerPolicy.priorityClass` is resolved to concrete, node-local GPUs through
the `opendps.io/gpu-indices` Pod annotation. The annotation is a comma-separated
list such as `"0,2"`. Pods without a UID, assigned node, or valid annotation do
not produce assignments.

The operator publishes `resolved-assignments.json` in the
`opendps-job-boosts` ConfigMap. Its versioned envelope is:

```json
{
  "schemaVersion": 1,
  "assignments": [
    {
      "policyUid": "example-policy",
      "policyGeneration": 3,
      "podUid": "example-pod",
      "nodeName": "example-node",
      "gpuIndex": 0,
      "priorityClass": "high",
      "gpuBoostPct": 20.0
    }
  ]
}
```

The identifiers above are illustrative. Entries are deterministic and contain
one record per resolved Pod/GPU pair.

Each controller polls this ConfigMap and filters assignments to its configured
node. When multiple assignments target one GPU, the highest priority tier
wins. The CLI `--gpu-priority-tiers` mapping is the baseline; a resolved
assignment overrides the same GPU, and unrelated CLI entries remain active.
Updates are loaded between control ticks without restarting the controller.

A `404` response or a valid empty assignment array restores the CLI baseline.
Malformed payloads and other API failures preserve the last-known-good
configuration.

The CLI baseline may be empty when `OPENDPS_PRIORITY_CONFIG_ENABLED=true` and
`OPENDPS_NODE_NAME` identifies the local node, allowing a dynamic-only
`priority-prs` startup. The controller retains the existing validation when no
dynamic source is configured.

Pod lifecycle events recompute affected policy assignments, including late
annotation and Pod deletion. The operator aggregates entries from all policies
and uses resource-version compare-and-swap retries, so updating or deleting one
policy does not discard a sibling policy's assignments.

The annotation is an explicit demo-grade mapping, not automatic discovery of
the devices allocated to a Pod. Device-plugin allocation introspection and
Dynamic Resource Allocation integration remain future work.
