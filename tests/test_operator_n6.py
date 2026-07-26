"""N6 — job/policy intake tests.

Operator side: JobPowerPolicy reconcile counts *real* matching pods (not a
hardcoded 0) and publishes a boost registry. Brain side: JobAwarePRSBrain
boosts a GPU that the tracker reports busy.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# Stub kubernetes/kopf before importing handlers (mirrors test_operator.py).
kubernetes_stub = MagicMock()
sys.modules.setdefault("kubernetes", kubernetes_stub)
sys.modules.setdefault("kubernetes.client", kubernetes_stub.client)
sys.modules.setdefault("kubernetes.client.exceptions", kubernetes_stub.client.exceptions)
try:
    import kopf  # noqa: F401
except ImportError:
    kopf_stub = MagicMock()
    kopf_stub.PermanentError = Exception
    sys.modules["kopf"] = kopf_stub

from opendps.operator import handlers  # noqa: E402
from opendps.operator.handlers import (  # noqa: E402
    _count_matching_pods,
    on_jobpowerpolicy_change,
)


def _pod_list(n):
    """A fake list_namespaced_pod return value with n items."""
    result = MagicMock()
    result.items = [MagicMock() for _ in range(n)]
    return result


# ---------------------------------------------------------------------------
# _count_matching_pods — must return the REAL count, distinguishable from a stub
# ---------------------------------------------------------------------------


def test_count_matching_pods_returns_real_count():
    api = MagicMock()
    api.list_namespaced_pod.return_value = _pod_list(3)
    with patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api):
        n = _count_matching_pods("opendps", {"opendps.io/workload": "training"})
    assert n == 3, "must reflect the actual number of matching pods, not 0"
    # Selector must be built from the labels and passed to the API.
    _, kwargs = api.list_namespaced_pod.call_args
    assert kwargs["label_selector"] == "opendps.io/workload=training"


def test_count_matching_pods_empty_labels_is_zero():
    # No selector → no match; must not even call the API.
    api = MagicMock()
    with patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api):
        assert _count_matching_pods("opendps", {}) == 0
    api.list_namespaced_pod.assert_not_called()


def test_on_jobpowerpolicy_writes_real_matchedpods_and_registry():
    patch_obj = MagicMock()
    patch_obj.status = {}
    api = MagicMock()
    api.list_namespaced_pod.return_value = _pod_list(2)
    with (
        patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api),
        patch("opendps.operator.handlers._write_boost_registry") as wbr,
    ):
        on_jobpowerpolicy_change(
            spec={
                "matchLabels": {"opendps.io/workload": "training"},
                "gpuBoostPct": 20.0,
                "priorityClass": "high",
            },
            name="jpp-test",
            namespace="opendps",
            patch=patch_obj,
        )
    assert patch_obj.status["matchedPods"] == 2
    assert patch_obj.status["activeBoosts"] == 2
    # Boost registry got the policy entry with the real match count.
    (_, policy_name, entry), _ = wbr.call_args
    assert policy_name == "jpp-test"
    assert entry["gpu_boost_pct"] == 20.0
    assert entry["matched_pods"] == 2


def test_on_jobpowerpolicy_zero_boost_no_active():
    patch_obj = MagicMock()
    patch_obj.status = {}
    api = MagicMock()
    api.list_namespaced_pod.return_value = _pod_list(5)
    with (
        patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api),
        patch("opendps.operator.handlers._write_boost_registry"),
    ):
        on_jobpowerpolicy_change(
            spec={"matchLabels": {"a": "b"}, "gpuBoostPct": 0.0},
            name="jpp-zero",
            namespace="opendps",
            patch=patch_obj,
        )
    assert patch_obj.status["matchedPods"] == 5
    assert patch_obj.status["activeBoosts"] == 0


# ---------------------------------------------------------------------------
# Brain side — JobAwarePRSBrain boosts a busy GPU
# ---------------------------------------------------------------------------


def test_job_aware_brain_boosts_busy_gpu():
    import time

    from opendps.agent.job_tracker import JobTracker
    from opendps.brain.dpm import DomainState
    from opendps.brain.job_aware_prs import JobAwarePRSBrain
    from opendps.pdn.presets import demo_single_domain

    # Tight budget (1200 W for two hot GPUs) so PRS caps land well below the
    # 1000 W hw max, leaving headroom for the boost to be observable.
    topo = demo_single_domain(n_gpus=2, budget_w=1200.0)
    tracker = JobTracker()
    tracker.set_busy_gpus([0])  # GPU 0 busy, GPU 1 idle — no nvidia-smi needed
    brain = JobAwarePRSBrain(topo, tracker, priority_boost=0.20)

    # Symmetric draws so PRS alone would cap both GPUs identically (~600 W each).
    state = DomainState(
        domain_name="domain-0",
        gpu_draws={0: 500.0, 1: 500.0},
        gpu_caps={0: 600.0, 1: 600.0},
        gpu_max_caps={0: 1000.0, 1: 1000.0},
        ts=time.time(),
    )
    decision = brain.decide("domain-0", state)
    assert decision.caps[0] > decision.caps[1], "busy GPU 0 should get the boost"
    assert decision.caps[0] <= 1000.0, "boost must not exceed hw max"


def test_set_busy_gpus_marks_busy():
    from opendps.agent.job_tracker import JobTracker

    tracker = JobTracker()
    tracker.set_busy_gpus([2, 5])
    assert tracker.is_gpu_busy(2)
    assert tracker.is_gpu_busy(5)
    assert not tracker.is_gpu_busy(0)


def _resolved_pod(uid, node_name, gpu_indices):
    from types import SimpleNamespace

    annotations = {}
    if gpu_indices is not None:
        annotations[handlers.GPU_INDICES_ANNOTATION] = gpu_indices
    return SimpleNamespace(
        metadata=SimpleNamespace(uid=uid, annotations=annotations),
        spec=SimpleNamespace(node_name=node_name),
    )


def test_resolved_assignments_contract_is_sorted_and_complete():
    pods = [
        _resolved_pod("pod-z", "node-b", "2,0"),
        _resolved_pod("pod-a", "node-a", "1"),
    ]

    assignments = handlers._resolved_assignments(
        pods,
        policy_uid="policy-uid",
        policy_generation=7,
        priority_class="high",
        boost_pct=20.0,
    )

    assert assignments == [
        {
            "policyUid": "policy-uid",
            "policyGeneration": 7,
            "podUid": "pod-a",
            "nodeName": "node-a",
            "gpuIndex": 1,
            "priorityClass": "high",
            "gpuBoostPct": 20.0,
        },
        {
            "policyUid": "policy-uid",
            "policyGeneration": 7,
            "podUid": "pod-z",
            "nodeName": "node-b",
            "gpuIndex": 0,
            "priorityClass": "high",
            "gpuBoostPct": 20.0,
        },
        {
            "policyUid": "policy-uid",
            "policyGeneration": 7,
            "podUid": "pod-z",
            "nodeName": "node-b",
            "gpuIndex": 2,
            "priorityClass": "high",
            "gpuBoostPct": 20.0,
        },
    ]


def test_resolved_assignments_skip_unresolved_and_malformed_pods():
    pods = [
        _resolved_pod("missing", "node-a", None),
        _resolved_pod("malformed", "node-a", "not-an-index"),
        _resolved_pod("unscheduled", None, "0"),
        _resolved_pod("valid", "node-b", "3"),
    ]

    assignments = handlers._resolved_assignments(
        pods,
        policy_uid="policy-uid",
        policy_generation=1,
        priority_class="normal",
        boost_pct=15.0,
    )

    assert assignments == [
        {
            "policyUid": "policy-uid",
            "policyGeneration": 1,
            "podUid": "valid",
            "nodeName": "node-b",
            "gpuIndex": 3,
            "priorityClass": "normal",
            "gpuBoostPct": 15.0,
        }
    ]


def test_gpu_index_annotation_rejects_duplicates_and_negative_indices():
    assert handlers._pod_gpu_indices(_resolved_pod("duplicate", "node-a", "1,1")) is None
    assert handlers._pod_gpu_indices(_resolved_pod("negative", "node-a", "-1")) is None
