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
(captured by the operator into the `opendps-job-boosts` ConfigMap); wiring that
ConfigMap into the in-cluster controller is the follow-up (tracked with N19).

## Demonstration

`scripts/demo.sh` step **DC11** runs the tight-budget demo topology with GPUs
0/1/2 tagged critical/low/normal under equal load:

```text
critical GPU0 = 1000 W; normal GPU2 = ~320 W; low GPU1 = ~250 W
```

The check asserts `critical > normal > low`.

## Limitations

- Tiers are supplied via CLI/JSON in process mode; the k8s
  `priorityClass`→controller path lands with the in-cluster deployment (N19).
- Preemption is expressed through cap weighting, not hard job suspension — a
  low-tier GPU keeps its floor, it is not driven to zero.
- Tier is a per-GPU attribute here; per-job tiering on a shared GPU is future
  work.
