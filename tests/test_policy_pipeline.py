import json
import importlib
from concurrent.futures import ThreadPoolExecutor
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

    def list_namespaced_pod(self, namespace, label_selector=None):
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
        "training",
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

    def fail_once(namespace, label_selector=None):
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


def test_stable_registry_entry_does_not_replace_configmap():
    api = RegistryApi()
    entry = {"assignments": [_assignment("policy-a", "pod-a", 0, "high")]}

    with patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api):
        handlers._write_boost_registry("training", "policy-a", entry)
        first_replace_count = api.replace_calls
        handlers._write_boost_registry("training", "policy-a", entry)

    assert first_replace_count == 1
    assert api.replace_calls == first_replace_count


def test_stable_timer_status_does_not_patch_or_replace():
    patch_status = SimpleNamespace(status={})

    with (
        patch.object(handlers, "_reconcile_jobpowerpolicy", return_value=(1, 1)),
        patch.object(handlers, "_prune_pod_assignment_digests"),
    ):
        handlers.resync_jobpowerpolicy(
            {"matchLabels": {"job": "a"}},
            "policy-a",
            "training",
            patch_status,
            status={"matchedPods": 1, "activeBoosts": 1},
        )

    assert patch_status.status == {}


def test_timer_prunes_missed_deleted_pod_digest():
    api = RegistryApi()
    api.pods = [
        SimpleNamespace(
            metadata=SimpleNamespace(uid="pod-live"),
            spec=SimpleNamespace(node_name="node-a"),
        )
    ]
    digest = (None, (("job", "a"),), "node-a")
    handlers._POD_ASSIGNMENT_DIGESTS.update(
        {
            "pod-live": (digest, True, "training"),
            "pod-stale": (digest, True, "training"),
            "pod-other": (digest, True, "other"),
        }
    )

    with (
        patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api),
        patch.object(handlers, "_reconcile_jobpowerpolicy", return_value=(0, 0)),
    ):
        handlers.resync_jobpowerpolicy(
            {"matchLabels": {"job": "a"}},
            "policy-a",
            "training",
            SimpleNamespace(status={}),
            status={"matchedPods": 0, "activeBoosts": 0},
        )

    assert set(handlers._POD_ASSIGNMENT_DIGESTS) == {"pod-live", "pod-other"}


def test_digest_cache_allows_concurrent_pod_events_and_timer_prunes():
    api = RegistryApi()

    def emit(index):
        handlers.on_pod_event(
            event={
                "type": "ADDED",
                "object": {
                    "metadata": {
                        "uid": f"pod-{index}",
                        "annotations": {handlers.GPU_INDICES_ANNOTATION: "0"},
                        "labels": {"job": "a"},
                    },
                    "spec": {"nodeName": "node-a"},
                },
            },
            namespace="training",
        )

    with (
        patch.object(handlers.kubernetes.client, "CoreV1Api", return_value=api),
        patch.object(handlers, "_reconcile_jobpowerpolicies_for_namespace"),
        ThreadPoolExecutor(max_workers=8) as executor,
    ):
        futures = [executor.submit(emit, index) for index in range(100)]
        futures.extend(
            executor.submit(handlers._prune_pod_assignment_digests, "training")
            for _ in range(25)
        )
        for future in futures:
            future.result()


def test_policy_selector_cache_reuses_and_invalidates_namespace_list():
    custom_api = SimpleNamespace(calls=0)

    def list_policies(*_args, **_kwargs):
        custom_api.calls += 1
        return {"items": [{"spec": {"matchLabels": {"job": "a"}}}]}

    custom_api.list_namespaced_custom_object = list_policies
    body = {
        "metadata": {"annotations": {}, "labels": {"job": "a"}},
        "spec": {"nodeName": "node-a"},
    }

    with patch.object(
        handlers.kubernetes.client,
        "CustomObjectsApi",
        return_value=custom_api,
    ):
        assert handlers._pod_matches_assignment_policy(body, "training")
        assert handlers._pod_matches_assignment_policy(body, "training")
        assert custom_api.calls == 1
        handlers._invalidate_policy_selector_cache("training")
        assert handlers._pod_matches_assignment_policy(body, "training")

    assert custom_api.calls == 2


def test_policy_handlers_invalidate_selector_cache():
    handlers._POLICY_SELECTOR_CACHE["training"] = (float("inf"), ((("job", "a"),),))
    patch_status = SimpleNamespace(status={})

    with patch.object(handlers, "_reconcile_jobpowerpolicy", return_value=(0, 0)):
        handlers.on_jobpowerpolicy_change(
            {"matchLabels": {"job": "a"}},
            "policy-a",
            "training",
            patch_status,
        )

    assert "training" not in handlers._POLICY_SELECTOR_CACHE
    handlers._POLICY_SELECTOR_CACHE["training"] = (float("inf"), ((("job", "a"),),))
    with patch.object(handlers, "_write_boost_registry"):
        handlers.on_jobpowerpolicy_delete("policy-a", "training")
    assert "training" not in handlers._POLICY_SELECTOR_CACHE


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
