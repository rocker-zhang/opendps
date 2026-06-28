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
