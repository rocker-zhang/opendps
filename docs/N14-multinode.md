# N14 — Multi-node cluster power coordination

A single controller manages one node's domains. N14 adds a layer above that: a
**cluster coordinator** that splits one cluster-wide power budget across several
nodes in proportion to their recent draw, so a busy node can borrow headroom
from idle ones without any node — or the cluster as a whole — exceeding its
power budget.

## Components

- **`NodeState`** — what each node reports: `node_id`, `domain_name`, `draw_w`,
  `cap_w`, `budget_w`, `ts`.
- **`NodeStateStore`** (Protocol) — `publish(state)` (a node pushes its state)
  and `get_all()` (the coordinator reads every live node).
  - `InMemoryStore` — thread-safe dict, single process (sim/test).
  - `RedisStore` — JSON under `opendps:node:{id}` with a TTL, for a real
    multi-process cluster (`pip install ".[redis]"`).
- **`ClusterCoordinator`** — holds the cluster budget and runs `rebalance()`.

## Rebalancing algorithm

`rebalance()` holds one hard invariant: **`Σ(node budgets) ≤ cluster_budget`**.
Power is a physical ceiling — oversubscribing the cluster is the dangerous
failure mode — so the budget is never exceeded.

For `n` nodes with `fair = cluster_budget / n`:

1. **Floor**: every node is reserved `0.5 × fair`. The floors sum to exactly 50%
   of the cluster budget, so they are always affordable (no infeasible case).
2. **Surplus**: the remaining 50% is handed out in proportion to each node's
   recent draw.
3. **Ceiling**: each node is capped at `2.0 × fair`. If the cap bites a hot
   node, the unused surplus is left as cluster headroom rather than
   oversubscribed.

So an idle node keeps a guaranteed floor, a busy node gets a proportionally
larger share, and the sum never exceeds the cluster budget.

## State flow

Push/pull through the store: each node periodically `publish()`es its state; the
coordinator `rebalance()`s, reading all states and writing back per-node
budgets. In production the store is Redis (cross-process, TTL-evicted); in the
sim/demo it is the in-memory store within one process.

## Demonstration

`scripts/demo.sh` step **DC10** runs the coordinator once over a busy node and
two idle nodes sharing a cluster budget:

```bash
python -m opendps.controller.cluster_coordinator --sim \
  --cluster-budget-w 12000 --nodes node0=8000,node1=500,node2=500
```

```text
busy node0 = 7333 W; idle node1 = 2333 W; idle node2 = 2333 W; total = 12000 W
```

The check asserts the busy node gets the larger share **and** the total never
exceeds the cluster budget.

## Limitations (not yet wired)

- **Per-node budget adoption**: `rebalance()` computes per-node budgets, but the
  path by which each node's controller *adopts* its new budget (re-pushed
  topology, advisory log, or direct config update) is not yet wired — the
  coordinator is run one-shot in the demo.
- **Live multi-process run**: the demo uses the in-memory store in a single
  process; a real multi-node run needs the Redis store and a node-side publisher
  loop (the `RedisStore` is unit-tested against a mock, not a live server).
- **Stale-node eviction**: the Redis store evicts via TTL; the in-memory store
  has no heartbeat, so a silently-departed node lingers until overwritten.
- **`AgentBridge`** (controller → Rust agent over TCP) is a defined protocol
  with graceful-degradation tests, but no agent is deployed in the demo path.
