# N19 — Per-node budget adoption

The cluster coordinator computed per-node budgets, but nothing consumed them —
the documented gap in the multi-node design ("the path by which each node's
controller adopts its new budget is not yet wired"). N19 wires it: the
coordinator publishes each node's rebalanced budget, and that node's controller
adopts it.

## Data flow

1. **Publish** — `ClusterCoordinator.rebalance()` writes each node/domain's new
   budget into the shared store via `set_adopted_budget(node_id, domain, watts)`
   (added to the `NodeStateStore` protocol; implemented for both
   `InMemoryStore` and `RedisStore`, the latter under a separate
   `opendps:budget:` keyspace so `get_all()` still returns only node states).
2. **Adopt** — a `StandaloneController` linked to the store (`node_state_store` +
   `node_id`) reads its node's budget at the start of each tick and calls
   `topology.adopt_budget(domain, watts)`. Because every brain sizes caps
   against `PDNTopology.domain_budget_w()`, the adopted budget supersedes the
   static (and rack-scaled) budget for that tick and binds the whole control
   loop. An absent or implausibly small budget (< `adopted_budget_min_w`)
   releases the override and falls back to the topology budget.

In sim both run in one process sharing an `InMemoryStore`; in production the
controller and coordinator are separate processes sharing Redis.

## Safety

- Default-safe: with no store linked, behaviour is exactly the topology budget.
- The adopted budget is clamped against `min_cap_w`/hardware-max by the brain as
  usual, and the coordinator's `Σ ≤ cluster_budget` invariant already bounds the
  published budgets, so no node can be handed more than the cluster allows.

## Validation

`tests/test_cluster_coordinator.py` adds: the coordinator publishes a budget per
node/domain after rebalance; a controller's total caps **track** its adopted
budget (a larger budget raises caps; a low budget binds them below it); an
implausibly small budget falls back to the topology budget; an unknown node has
no adopted budget. This closes the N14 "per-node budget adoption" limitation.

## Limitations

- Live multi-process still needs the Redis store and a node-side publisher loop;
  the demo exercises the in-memory store in one process.
- Budgets are published per node/domain key; the publish is batched after the
  full rebalance but is not a single atomic store transaction (a Redis MULTI /
  epoch marker would close the last small window).
