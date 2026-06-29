from unittest.mock import MagicMock

from opendps.controller.cluster_coordinator import ClusterCoordinator, InMemoryStore, NodeState
import time

def _state(node_id, draw_w, budget_w=8000.0):
    return NodeState(node_id=node_id, domain_name="d0", draw_w=draw_w,
                     cap_w=draw_w, budget_w=budget_w, ts=time.time())

def test_rebalance_proportional():
    store = InMemoryStore()
    store.publish(_state("node0", draw_w=6000.0))
    store.publish(_state("node1", draw_w=2000.0))
    coord = ClusterCoordinator(store, total_cluster_budget_w=16000.0)
    budgets = coord.rebalance()
    # node0 draws 75%, node1 draws 25% — proportional allocation
    assert budgets["node0"] > budgets["node1"]
    assert abs(sum(budgets.values()) - 16000.0) < 1.0  # total preserved

def test_rebalance_min_floor():
    store = InMemoryStore()
    store.publish(_state("node0", draw_w=9999.0))
    store.publish(_state("node1", draw_w=1.0))
    coord = ClusterCoordinator(store, total_cluster_budget_w=16000.0)
    budgets = coord.rebalance()
    fair = 8000.0
    assert budgets["node1"] >= fair * 0.5  # min floor = 50% of fair share


def test_redis_store_publish_and_get_all():
    """Mock redis to verify RedisStore serializes/deserializes NodeState."""
    from opendps.controller.cluster_coordinator import RedisStore, NodeState
    import time

    mock_redis = MagicMock()
    mock_redis.keys.return_value = ["opendps:node:node0", "opendps:node:node1"]

    state0 = NodeState("node0", "d0", 6000.0, 6000.0, 8000.0, time.time())
    state1 = NodeState("node1", "d0", 2000.0, 2000.0, 8000.0, time.time())

    import json
    import dataclasses
    mock_redis.get.side_effect = [
        json.dumps(dataclasses.asdict(state0)),
        json.dumps(dataclasses.asdict(state1)),
    ]

    store = RedisStore("redis://localhost:6379")
    store._client = mock_redis  # inject mock

    retrieved = store.get_all()
    assert len(retrieved) == 2
    node_ids = {s.node_id for s in retrieved}
    assert node_ids == {"node0", "node1"}


def test_redis_store_publish_sets_ttl():
    """publish() calls setex with correct key and TTL."""
    from opendps.controller.cluster_coordinator import RedisStore, NodeState
    import time

    mock_redis = MagicMock()
    store = RedisStore("redis://localhost:6379", ttl_s=30)
    store._client = mock_redis

    state = NodeState("node0", "d0", 1000.0, 1000.0, 8000.0, time.time())
    store.publish(state)

    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args[0]
    assert args[0] == "opendps:node:node0"
    assert args[1] == 30  # TTL


# --- N14 hardening: hard Σ<=budget invariant, ceiling, CLI sim ---


def test_rebalance_never_oversubscribes_under_extreme_skew():
    """The previous clamp-only rebalancer could sum above the cluster budget on
    skewed demand. The floor-reserve + proportional-surplus version must never
    oversubscribe the cluster power budget."""
    store = InMemoryStore()
    store.publish(_state("hot", draw_w=9999.0))
    store.publish(_state("idle", draw_w=1.0))
    coord = ClusterCoordinator(store, total_cluster_budget_w=16000.0)
    budgets = coord.rebalance()
    assert sum(budgets.values()) <= 16000.0 + 1e-6, f"oversubscribed: {budgets}"
    # idle node still keeps its 50%-of-fair floor (feasible: floors = 50% of budget).
    assert budgets["idle"] >= 16000.0 / 2 * 0.5 - 1e-6


def test_rebalance_respects_ceiling():
    """A single dominant node is capped at 200% of fair share; the unused
    surplus is left as headroom (never handed out past the cap)."""
    store = InMemoryStore()
    store.publish(_state("hot", draw_w=10000.0))
    store.publish(_state("a", draw_w=0.0))
    store.publish(_state("b", draw_w=0.0))
    store.publish(_state("c", draw_w=0.0))
    budget = 8000.0
    coord = ClusterCoordinator(store, total_cluster_budget_w=budget)
    budgets = coord.rebalance()
    fair = budget / 4
    assert budgets["hot"] <= fair * 2.0 + 1e-6
    assert sum(budgets.values()) <= budget + 1e-6


def test_rebalance_all_idle_gives_floors():
    """Zero total draw -> everyone gets their floor, still within budget."""
    store = InMemoryStore()
    for nid in ("n0", "n1", "n2", "n3"):
        store.publish(_state(nid, draw_w=0.0))
    budget = 8000.0
    budgets = ClusterCoordinator(store, total_cluster_budget_w=budget).rebalance()
    assert sum(budgets.values()) <= budget + 1e-6
    assert all(b >= budget / 4 * 0.5 - 1e-6 for b in budgets.values())


def test_rebalance_empty_store():
    assert ClusterCoordinator(InMemoryStore(), total_cluster_budget_w=8000.0).rebalance() == {}


def test_cli_sim_prints_budgets(capsys):
    import re

    import pytest

    from opendps.controller.cluster_coordinator import main

    rc = main(["--sim", "--cluster-budget-w", "12000",
               "--nodes", "node0=8000,node1=500,node2=500"])
    assert rc == 0
    out = capsys.readouterr().out
    assert 'opendps_cluster_node_budget_w{node="node0"}' in out
    # Parse the actual total rather than substring-match (12000 ⊂ 120001).
    m = re.search(r"total_allocated_w=([0-9.eE+-]+)", out)
    assert m is not None
    assert float(m.group(1)) == pytest.approx(12000.0, abs=0.01)


def test_cli_rejects_bad_args():
    import pytest

    from opendps.controller.cluster_coordinator import main

    bad = [
        ["--cluster-budget-w", "0", "--nodes", "a=1"],      # non-positive budget
        ["--cluster-budget-w", "nan", "--nodes", "a=1"],    # NaN budget
        ["--cluster-budget-w", "inf", "--nodes", "a=1"],    # Inf budget
        ["--cluster-budget-w", "100", "--nodes", "bogus"],  # no '='
        ["--cluster-budget-w", "100", "--nodes", "a=nan"],  # NaN draw
        ["--cluster-budget-w", "100", "--nodes", "a=inf"],  # Inf draw
        ["--cluster-budget-w", "100", "--nodes", "a=-5"],   # negative draw
        ["--cluster-budget-w", "100", "--nodes", "a=xyz"],  # non-numeric
    ]
    for argv in bad:
        with pytest.raises(SystemExit):
            main(["--sim", *argv])
