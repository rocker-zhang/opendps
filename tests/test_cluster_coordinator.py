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


from unittest.mock import MagicMock, patch


def test_redis_store_publish_and_get_all():
    """Mock redis to verify RedisStore serializes/deserializes NodeState."""
    from opendps.controller.cluster_coordinator import RedisStore, NodeState
    import time

    mock_redis = MagicMock()
    mock_redis.keys.return_value = ["opendps:node:node0", "opendps:node:node1"]

    state0 = NodeState("node0", "d0", 6000.0, 6000.0, 8000.0, time.time())
    state1 = NodeState("node1", "d0", 2000.0, 2000.0, 8000.0, time.time())

    import json, dataclasses
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
