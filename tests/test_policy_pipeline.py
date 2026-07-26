import json
import importlib
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
        self.on_conflict = None
        self.replace_calls = 0
        self.pods = []

    def read_namespaced_config_map(self, name, namespace):
        return self.config_map

    def list_namespaced_pod(self, namespace, label_selector):
        return SimpleNamespace(items=list(self.pods))

    def patch_namespaced_config_map(self, name, namespace, body):
        self.replace_calls += 1
        if self.conflict_once:
            self.conflict_once = False
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
        handlers.on_pod_lifecycle(namespace="training")
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
        handlers.on_pod_lifecycle(namespace="training")
        payload = json.loads(api.config_map.data[handlers.RESOLVED_ASSIGNMENTS_KEY])
        assert len(payload["assignments"]) == 1

        api.pods = []
        handlers.on_pod_lifecycle(namespace="training")
        payload = json.loads(api.config_map.data[handlers.RESOLVED_ASSIGNMENTS_KEY])
        assert payload["assignments"] == []


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
