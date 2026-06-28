"""Multi-node cluster power coordinator (N14 skeleton)."""
from __future__ import annotations
import dataclasses, threading, time
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
        """Redistribute budget proportionally to recent draw."""
        states = self._store.get_all()
        if not states:
            return {}
        n = len(states)
        fair = self._cluster_budget / n
        total_draw = sum(s.draw_w for s in states) or 1.0
        new_budgets: dict[str, float] = {}
        for s in states:
            proportional = (s.draw_w / total_draw) * self._cluster_budget
            # Clamp: never drop below 50% of fair share
            new_budgets[s.node_id] = max(fair * 0.5, min(proportional, fair * 2.0))
        with self._lock:
            self._node_budgets = new_budgets
        return new_budgets
