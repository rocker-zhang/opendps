import json
import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from opendps.controller.priority_config import PriorityConfigSource
from opendps.operator import handlers


class FakeKubeError(Exception):
    def __init__(self, status):
        super().__init__(f"Kubernetes API status {status}")
        self.status = status


class RegistryApi:
    def __init__(self):
        self.config_map = SimpleNamespace(
            metadata=SimpleNamespace(resource_version="1"),
            data={},
        )
        self.conflict_once = False
        self.conflicts_remaining = 0
        self.missing_once = False
        self.on_conflict = None
        self.create_calls = 0
        self.replace_calls = 0
        self.read_calls = 0
        self.pods = []

    def read_namespaced_config_map(self, name, namespace, **kwargs):
        self.read_calls += 1
        if self.missing_once:
            self.missing_once = False
            raise FakeKubeError(status=404)
        return self.config_map

    def list_namespaced_pod(self, namespace, label_selector):
        return SimpleNamespace(items=list(self.pods))

    def patch_namespaced_config_map(self, name, namespace, body):
        self.replace_calls += 1
        if self.conflict_once or self.conflicts_remaining:
            self.conflict_once = False
            self.conflicts_remaining = max(0, self.conflicts_remaining - 1)
            if self.on_conflict is not None:
                self.on_conflict()
            raise FakeKubeError(status=409)
        data = body["data"] if isinstance(body, dict) else body.data
        self.config_map = SimpleNamespace(
            metadata=SimpleNamespace(resource_version="2"),
            data=dict(data),
        )
        return self.config_map

    replace_namespaced_config_map = patch_namespaced_config_map

    def create_namespaced_config_map(self, namespace, body):
        self.create_calls += 1
        if self.conflicts_remaining:
            self.conflicts_remaining -= 1
            raise FakeKubeError(status=409)
        data = body["data"] if isinstance(body, dict) else body.data
        self.config_map = SimpleNamespace(
            metadata=SimpleNamespace(resource_version="1"),
            data=dict(data),
        )
        return self.config_map


@pytest.fixture(autouse=True)
def _restore_operator_module_state():
    importlib.reload(handlers)
    with (
        patch.object(
            handlers.kubernetes.client,
            "V1ConfigMap",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        patch.object(
            handlers.kubernetes.client,
            "V1ObjectMeta",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        patch.object(
            handlers.kubernetes.client.exceptions,
            "ApiException",
            FakeKubeError,
        ),
    ):
        yield


def _assignment(policy, pod, gpu, tier, node="node-a"):
    return {
        "policyUid": policy,
        "policyGeneration": 1,
        "podUid": pod,
        "nodeName": node,
        "gpuIndex": gpu,
        "priorityClass": tier,
        "gpuBoostPct": 20.0,
    }


def test_operator_registry_output_is_consumed_without_schema_translation():
    api = RegistryApi()
    entry = {
        "assignments": [
            _assignment("policy-a", "pod-a", 0, "high"),
            _assignment("policy-a", "pod-a", 1, "low", node="node-b"),
        ]
    }

    with patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api):
        handlers._write_boost_registry("power-system", "policy-a", entry)

    source = PriorityConfigSource(
        namespace="power-system",
        node_name="node-a",
        baseline={2: "normal"},
        api=api,
    )

    assert source.poll() == {0: "high", 2: "normal"}
    payload = json.loads(api.config_map.data[handlers.RESOLVED_ASSIGNMENTS_KEY])
    assert payload["schemaVersion"] == 1
    assert payload["assignments"] == entry["assignments"]


def test_registry_delete_keeps_sibling_policy_assignments():
    api = RegistryApi()
    first = {"assignments": [_assignment("policy-a", "pod-a", 0, "high")]}
    sibling = {"assignments": [_assignment("policy-b", "pod-b", 1, "critical")]}

    with patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api):
        handlers._write_boost_registry("default", "policy-a", first)
        handlers._write_boost_registry("default", "policy-b", sibling)
        handlers._write_boost_registry("default", "policy-a", None)

    payload = json.loads(api.config_map.data[handlers.RESOLVED_ASSIGNMENTS_KEY])
    assert payload == {
        "schemaVersion": 1,
        "assignments": sibling["assignments"],
    }


def test_pod_late_annotation_and_delete_recompute_aggregate():
    api = RegistryApi()
    pod = SimpleNamespace(
        metadata=SimpleNamespace(
            uid="pod-a",
            labels={"job": "a"},
            annotations={},
        ),
        spec=SimpleNamespace(node_name="node-a"),
    )
    api.pods = [pod]
    custom_api = SimpleNamespace(
        list_namespaced_custom_object=lambda *_args, **_kwargs: {
            "items": [
                {
                    "metadata": {
                        "name": "policy-a",
                        "uid": "policy-a",
                        "generation": 1,
                    },
                    "spec": {
                        "matchLabels": {"job": "a"},
                        "priorityClass": "high",
                        "gpuBoostPct": 20.0,
                    },
                }
            ]
        }
    )
    with (
        patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api),
        patch.object(
            handlers.kubernetes.client,
            "CustomObjectsApi",
            return_value=custom_api,
        ),
    ):
        event_body = {
            "metadata": {
                "uid": "pod-a",
                "labels": {"job": "a"},
                "annotations": {},
            },
            "spec": {"nodeName": "node-a"},
        }
        handlers.on_pod_event(
            event={"type": "ADDED", "object": event_body},
            namespace="training",
        )
        payload = json.loads(api.config_map.data[handlers.RESOLVED_ASSIGNMENTS_KEY])
        assert payload["assignments"] == []

        api.pods = [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    uid="pod-a",
                    labels={"job": "a"},
                    annotations={handlers.GPU_INDICES_ANNOTATION: "0"},
                ),
                spec=SimpleNamespace(node_name="node-a"),
            )
        ]
        assert handlers._pod_gpu_indices(api.pods[0]) == [0]
        annotated_body = {
            **event_body,
            "metadata": {
                **event_body["metadata"],
                "annotations": {handlers.GPU_INDICES_ANNOTATION: "0"},
            },
        }
        handlers.on_pod_event(
            event={"type": "MODIFIED", "object": annotated_body},
            namespace="training",
        )
        payload = json.loads(api.config_map.data[handlers.RESOLVED_ASSIGNMENTS_KEY])
        assert len(payload["assignments"]) == 1

        api.pods = []
        handlers.on_pod_event(
            event={"type": "DELETED", "object": annotated_body},
            namespace="training",
        )
        payload = json.loads(api.config_map.data[handlers.RESOLVED_ASSIGNMENTS_KEY])
        assert payload["assignments"] == []


def test_pod_event_handler_is_silent_kopf_event_handler():
    source = inspect.getsource(handlers)

    assert '@kopf.on.event("", "v1", "pods")' in source
    assert (
        '@kopf.timer("opendps.io", "v1alpha1", "jobpowerpolicies", interval=30.0)'
        in source
    )
    assert '@kopf.on.create("", "v1", "pods"' not in source
    assert '@kopf.on.update("", "v1", "pods"' not in source
    assert '@kopf.on.delete("", "v1", "pods"' not in source
    assert '@kopf.on.resume("", "v1", "pods"' not in source


def test_pod_event_ignores_status_only_updates():
    pod = {
        "metadata": {
            "uid": "pod-a",
            "annotations": {handlers.GPU_INDICES_ANNOTATION: "0"},
            "labels": {"job": "a"},
        },
        "spec": {"nodeName": "node-a"},
    }

    with (
        patch.object(handlers, "_pod_matches_assignment_policy", return_value=True),
        patch.object(handlers, "_reconcile_jobpowerpolicies_for_namespace") as reconcile,
    ):
        handlers.on_pod_event(
            event={"type": "ADDED", "object": {**pod, "status": {"phase": "Pending"}}},
            namespace="training",
        )
        handlers.on_pod_event(
            event={"type": "MODIFIED", "object": {**pod, "status": {"phase": "Running"}}},
            namespace="training",
        )

    reconcile.assert_called_once_with("training")


def test_pod_event_annotation_removal_and_delete_recompute():
    annotated = {
        "metadata": {
            "uid": "pod-a",
            "annotations": {handlers.GPU_INDICES_ANNOTATION: "0"},
            "labels": {"job": "a"},
        },
        "spec": {"nodeName": "node-a"},
    }
    unannotated = {
        **annotated,
        "metadata": {**annotated["metadata"], "annotations": {}},
    }

    with (
        patch.object(handlers, "_pod_matches_assignment_policy", return_value=True),
        patch.object(handlers, "_reconcile_jobpowerpolicies_for_namespace") as reconcile,
    ):
        handlers.on_pod_event(
            event={"type": "ADDED", "object": annotated},
            namespace="training",
        )
        handlers.on_pod_event(
            event={"type": "MODIFIED", "object": unannotated},
            namespace="training",
        )
        handlers.on_pod_event(
            event={"type": "DELETED", "object": unannotated},
            namespace="training",
        )

    assert reconcile.call_count == 3
    assert "pod-a" not in handlers._POD_ASSIGNMENT_DIGESTS


def test_pod_event_delete_rechecks_policy_after_cached_irrelevant():
    body = {
        "metadata": {
            "uid": "pod-a",
            "annotations": {},
            "labels": {"job": "a"},
        },
        "spec": {"nodeName": "node-a"},
    }
    handlers._POD_ASSIGNMENT_DIGESTS["pod-a"] = (
        handlers._pod_assignment_digest(body),
        False,
    )

    with (
        patch.object(handlers, "_pod_matches_assignment_policy", return_value=True),
        patch.object(handlers, "_reconcile_jobpowerpolicies_for_namespace") as reconcile,
    ):
        handlers.on_pod_event(
            event={"type": "DELETED", "object": body},
            namespace="training",
        )

    reconcile.assert_called_once_with("training")
    assert "pod-a" not in handlers._POD_ASSIGNMENT_DIGESTS


def test_timer_converges_registry_after_silent_event_failure():
    api = RegistryApi()
    pod = SimpleNamespace(
        metadata=SimpleNamespace(
            uid="pod-a",
            labels={"job": "a"},
            annotations={handlers.GPU_INDICES_ANNOTATION: "0"},
        ),
        spec=SimpleNamespace(node_name="node-a"),
    )
    api.pods = [pod]
    original_list = api.list_namespaced_pod
    calls = 0

    def fail_once(namespace, label_selector):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("connection reset")
        return original_list(namespace, label_selector)

    api.list_namespaced_pod = fail_once
    policy_spec = {
        "matchLabels": {"job": "a"},
        "priorityClass": "high",
        "gpuBoostPct": 20.0,
    }
    policy_meta = {"uid": "policy-a", "generation": 1}
    custom_api = SimpleNamespace(
        list_namespaced_custom_object=lambda *_args, **_kwargs: {
            "items": [
                {
                    "metadata": {"name": "policy-a", **policy_meta},
                    "spec": policy_spec,
                }
            ]
        }
    )
    body = {
        "metadata": {
            "uid": "pod-a",
            "annotations": {handlers.GPU_INDICES_ANNOTATION: "0"},
            "labels": {"job": "a"},
        },
        "spec": {"nodeName": "node-a"},
    }

    with (
        patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api),
        patch.object(
            handlers.kubernetes.client,
            "CustomObjectsApi",
            return_value=custom_api,
        ),
    ):
        with pytest.raises(handlers.kopf.TemporaryError):
            handlers.on_pod_event(
                event={"type": "ADDED", "object": body},
                namespace="training",
            )

        status_patch = SimpleNamespace(status={})
        handlers.resync_jobpowerpolicy(
            policy_spec,
            "policy-a",
            "training",
            status_patch,
            meta=policy_meta,
        )

    payload = json.loads(api.config_map.data[handlers.RESOLVED_ASSIGNMENTS_KEY])
    assert len(payload["assignments"]) == 1
    assert payload["assignments"][0]["podUid"] == "pod-a"
    assert status_patch.status == {"matchedPods": 1, "activeBoosts": 1}


def test_transient_pod_list_error_never_publishes_empty_assignments():
    api = RegistryApi()

    def fail(*_args, **_kwargs):
        raise OSError("connection reset")

    api.list_namespaced_pod = fail
    with (
        patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api),
        pytest.raises(handlers.kopf.TemporaryError),
    ):
        handlers._list_matching_pods("training", {"job": "a"})


def test_aggregate_sort_tolerates_hand_edited_field_types():
    data = {
        "policy-a.json": json.dumps(
            {
                "assignments": [
                    {
                        "nodeName": 2,
                        "gpuIndex": "0",
                        "podUid": None,
                        "policyUid": 4,
                    },
                    {
                        "nodeName": "1",
                        "gpuIndex": 1,
                        "podUid": "pod",
                        "policyUid": "policy",
                    },
                ]
            }
        )
    }

    payload = json.loads(handlers._aggregate_resolved_assignments(data))
    assert len(payload["assignments"]) == 2


def test_registry_retries_conflict_without_losing_sibling_policy():
    api = RegistryApi()
    sibling = {"assignments": [_assignment("policy-b", "pod-b", 1, "critical")]}
    updated = {"assignments": [_assignment("policy-a", "pod-a", 0, "high")]}
    initial = {"assignments": [_assignment("policy-a", "pod-a", 0, "normal")]}

    with patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api):
        handlers._write_boost_registry("default", "policy-a", initial)
        api.on_conflict = lambda: handlers._write_boost_registry("default", "policy-b", sibling)
        api.conflict_once = True
        handlers._write_boost_registry("default", "policy-a", updated)

    payload = json.loads(api.config_map.data[handlers.RESOLVED_ASSIGNMENTS_KEY])
    assert payload == {
        "schemaVersion": 1,
        "assignments": updated["assignments"] + sibling["assignments"],
    }
    assert api.replace_calls == 4


def test_registry_create_path_publishes_assignments():
    api = RegistryApi()
    api.missing_once = True
    entry = {"assignments": [_assignment("policy-a", "pod-a", 0, "high")]}

    with patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api):
        handlers._write_boost_registry("default", "policy-a", entry)

    payload = json.loads(api.config_map.data[handlers.RESOLVED_ASSIGNMENTS_KEY])
    assert payload["assignments"] == entry["assignments"]
    assert api.read_calls == 1
    assert api.create_calls == 1
    assert api.replace_calls == 0


def test_registry_conflict_exhaustion_requests_kopf_retry():
    api = RegistryApi()
    api.missing_once = True
    api.conflicts_remaining = 3
    entry = {"assignments": [_assignment("policy-a", "pod-a", 0, "high")]}

    with (
        patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api),
        patch.object(handlers.time, "sleep") as sleep,
        pytest.raises(handlers.kopf.TemporaryError),
    ):
        handlers._write_boost_registry("default", "policy-a", entry)

    assert api.read_calls == 3
    assert api.create_calls == 1
    assert api.replace_calls == 2
    assert sleep.call_count == 2
