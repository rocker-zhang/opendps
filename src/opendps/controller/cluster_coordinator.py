"""Multi-node cluster power coordinator (N14 skeleton)."""
from __future__ import annotations
import dataclasses
import math
import threading
from typing import Protocol


@dataclasses.dataclass
class NodeState:
    node_id: str
    domain_name: str
    draw_w: float
    cap_w: float
    budget_w: float
    ts: float


class NodeStateStore(Protocol):
    def publish(self, state: NodeState) -> None: ...
    def get_all(self) -> list[NodeState]: ...


class InMemoryStore:
    """Single-process stub — replace with Redis for production."""

    def __init__(self) -> None:
        self._states: dict[str, NodeState] = {}
        self._lock = threading.Lock()

    def publish(self, state: NodeState) -> None:
        with self._lock:
            self._states[state.node_id] = state

    def get_all(self) -> list[NodeState]:
        with self._lock:
            return list(self._states.values())


class RedisStore:
    """
    Production NodeStateStore backed by Redis.

    Each node publishes its state as a JSON string under key
    `opendps:node:{node_id}` with a TTL of 3 × publish_interval.
    get_all() scans for all `opendps:node:*` keys.

    Requires: pip install redis
    """
    _PREFIX = "opendps:node:"

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl_s: int = 30):
        self._url = redis_url
        self._ttl = ttl_s
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import redis as redis_lib
                self._client = redis_lib.Redis.from_url(self._url, decode_responses=True)
            except ImportError as e:
                raise ImportError("pip install redis to use RedisStore") from e
        return self._client

    def publish(self, state: NodeState) -> None:
        import json
        import dataclasses
        key = f"{self._PREFIX}{state.node_id}"
        data = json.dumps(dataclasses.asdict(state))
        self._get_client().setex(key, self._ttl, data)

    def get_all(self) -> list[NodeState]:
        import json
        client = self._get_client()
        keys = client.keys(f"{self._PREFIX}*")
        states = []
        for key in keys:
            raw = client.get(key)
            if raw:
                d = json.loads(raw)
                states.append(NodeState(**d))
        return states


class ClusterCoordinator:
    """
    Redistributes cluster-wide power budget across nodes based on draw.

    Algorithm: proportional-to-draw with min-floor = 50% of fair share.
    Designed to run on a dedicated coordinator node or as a sidecar.
    """

    def __init__(
        self,
        store: NodeStateStore,
        total_cluster_budget_w: float,
        rebalance_interval_s: float = 10.0,
    ):
        self._store = store
        self._cluster_budget = total_cluster_budget_w
        self._interval = rebalance_interval_s
        self._node_budgets: dict[str, float] = {}
        self._lock = threading.Lock()

    def get_node_budget(self, node_id: str) -> float | None:
        with self._lock:
            return self._node_budgets.get(node_id)

    def rebalance(self) -> dict[str, float]:
        """Redistribute the cluster budget across nodes proportionally to recent
        draw, holding the hard invariant ``Σ(budgets) ≤ cluster_budget``.

        Power is a physical ceiling: oversubscribing the cluster is the
        dangerous failure, so the budget is never exceeded. Each node first gets
        a floor of 50% of its fair share — the floors sum to exactly 50% of the
        cluster budget, so they are always affordable — then the remaining 50%
        surplus is handed out in proportion to draw, with each node capped at
        200% of fair share. If the ceiling caps a hot node, the unused surplus is
        left as cluster headroom rather than oversubscribed.
        """
        states = self._store.get_all()
        if not states:
            return {}
        n = len(states)
        fair = self._cluster_budget / n
        floor = fair * 0.5
        ceil = fair * 2.0
        surplus = self._cluster_budget - floor * n  # = 50% of the cluster budget
        total_draw = sum(s.draw_w for s in states) or 1.0
        new_budgets: dict[str, float] = {}
        for s in states:
            extra = (s.draw_w / total_draw) * surplus
            new_budgets[s.node_id] = min(floor + extra, ceil)
        with self._lock:
            self._node_budgets = dict(new_budgets)
        return new_budgets


def _parse_nodes(spec: str) -> list[tuple[str, float]]:
    """Parse ``node0=8000,node1=500`` into ``[(node_id, draw_w), ...]``."""
    nodes: list[tuple[str, float]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"bad --nodes entry {item!r}, expected node_id=draw_w")
        nid, draw = item.split("=", 1)
        draw_w = float(draw)  # raises ValueError on non-numeric junk
        # NaN/Inf parse fine via float() but would poison the rebalance math
        # (NaN budgets, broken Σ≤budget); negative draw is physically impossible.
        if not math.isfinite(draw_w):
            raise ValueError(f"draw for {nid.strip()!r} must be finite, got {draw!r}")
        if draw_w < 0:
            raise ValueError(f"draw for {nid.strip()!r} must be >= 0, got {draw_w}")
        nodes.append((nid.strip(), draw_w))
    return nodes


def main(argv: list[str] | None = None) -> int:
    """One-shot sim rebalance: publish the given node draws to an in-memory
    store, run the coordinator once, and print the per-node budgets. This is the
    demo/CLI entry point for N14 (``python -m opendps.controller.cluster_coordinator``)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="opendps-coordinator",
        description="N14 multi-node cluster power coordinator (one-shot sim rebalance).",
    )
    parser.add_argument("--cluster-budget-w", type=float, required=True,
                        help="total power budget shared across all nodes (W)")
    parser.add_argument("--nodes", required=True, metavar="ID=DRAW,...",
                        help="comma-separated node states as node_id=draw_w")
    parser.add_argument("--sim", action="store_true",
                        help="one-shot in-memory rebalance over --nodes (the only mode today)")
    args = parser.parse_args(argv)

    # NaN/Inf pass `float()` but break the rebalance math (`nan <= 0` is False),
    # so reject non-finite budgets too — not just non-positive ones.
    if not math.isfinite(args.cluster_budget_w) or args.cluster_budget_w <= 0:
        parser.error("--cluster-budget-w must be a finite number > 0")
    try:
        nodes = _parse_nodes(args.nodes)
    except ValueError as exc:
        parser.error(str(exc))
    if not nodes:
        parser.error("--nodes must list at least one node")

    store = InMemoryStore()
    for nid, draw in nodes:
        store.publish(NodeState(node_id=nid, domain_name=nid, draw_w=draw,
                                cap_w=draw, budget_w=0.0, ts=0.0))
    coord = ClusterCoordinator(store, total_cluster_budget_w=args.cluster_budget_w)
    budgets = coord.rebalance()
    total = sum(budgets.values())
    for nid, b in budgets.items():
        print(f'opendps_cluster_node_budget_w{{node="{nid}"}} {b}')
    print(f"cluster_budget_w={args.cluster_budget_w} total_allocated_w={total}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
