"""N4/N22 — opt-in live-Kubernetes integration tests.

Skipped unless OPENDPS_K8S_TEST=1 and kubectl can reach a cluster with the
opendps CRDs, operator, and demo controller running. This is the CI-automatable
equivalent of manual kind validation: it applies real resources to a real API
server and asserts reconciliation and controller hot reload (no mocks).

Bring-up (see scripts/demo.sh / README):
    docker build -f deploy/operator.Dockerfile -t opendps-operator:latest .
    docker build -f deploy/controller.Dockerfile -t opendps-controller:latest .
    kind load docker-image opendps-operator:latest opendps-controller:latest
    kubectl create namespace opendps
    kubectl apply -f deploy/k8s/crds/
    kubectl apply -f deploy/k8s/operator-deployment.yaml
    kubectl apply -f deploy/k8s/controller-deployment.yaml
    OPENDPS_K8S_TEST=1 pytest tests/test_operator_k8s_integration.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("OPENDPS_K8S_TEST") != "1",
    reason="set OPENDPS_K8S_TEST=1 (needs a cluster with opendps CRDs + operator)",
)

NS = "opendps"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _kubectl(*args, check=True, input_text=None):
    return subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        check=check,
        input=input_text,
        timeout=60,
    )


def _wait_for(fn, timeout=30, interval=2):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fn()
        if last:
            return last
        time.sleep(interval)
    return last


def _apply_json(body):
    return _kubectl("apply", "-f", "-", input_text=json.dumps(body))


def _priority_config():
    raw = _kubectl(
        "get", "configmap", "opendps-job-boosts", "-n", NS,
        "-o", "json", check=False,
    ).stdout
    if not raw:
        return None
    config_map = json.loads(raw)
    payload = (config_map.get("data") or {}).get("resolved-assignments.json")
    if payload is None:
        return None
    assignments = json.loads(payload).get("assignments")
    if assignments is None:
        return None
    return config_map["metadata"]["resourceVersion"], assignments


def _resolved_assignments():
    config = _priority_config()
    return config[1] if config else None


def _controller_pod():
    raw = _kubectl(
        "get", "pods", "-n", NS, "-l", "app=opendps-controller",
        "-o", "json", check=False,
    ).stdout
    if not raw:
        return None
    for pod in json.loads(raw).get("items", []):
        conditions = pod.get("status", {}).get("conditions", [])
        if any(
            item.get("type") == "Ready" and item.get("status") == "True"
            for item in conditions
        ):
            return pod
    return None


@pytest.fixture(scope="module", autouse=True)
def _crds_present():
    r = _kubectl("get", "crd", "powerdomains.opendps.io", check=False)
    if r.returncode != 0:
        pytest.skip("opendps CRDs not installed on the target cluster")


def _apply_demo_powerdomain():
    """Apply the demo PowerDomain and wait for it to reconcile to Active.
    Idempotent — each test that needs the domain calls this so tests are
    independent of execution order."""
    _kubectl("apply", "-n", NS, "-f",
             f"{ROOT}/deploy/k8s/examples/demo-powerdomain.yaml")
    # Poll until the phase is *Active*, not merely the first non-empty phase:
    # the controller passes through Pending/Reconciling first, so returning on
    # any non-empty value would assert too early and flake.
    def _active_phase():
        p = _kubectl(
            "get", "powerdomain", "demo", "-n", NS,
            "-o", "jsonpath={.status.phase}", check=False).stdout.strip()
        return p if p == "Active" else None

    phase = _wait_for(_active_phase)
    assert phase == "Active", f"expected phase=Active, got {phase!r}"


def test_powerdomain_reconciles_to_active():
    _apply_demo_powerdomain()

    # The operator must have written the topology ConfigMap with our spec.
    cm = _kubectl("get", "configmap", "opendps-topology-demo", "-n", NS,
                  "-o", "jsonpath={.data.topology\\.json}").stdout
    topo = json.loads(cm)
    dom = topo["domains"]["demo"]
    assert len(dom["gpu_indices"]) >= 1
    assert dom["budget_w"] > 0


def test_powerpolicy_params_propagate():
    _apply_demo_powerdomain()  # self-contained: ensure the domain exists first
    _kubectl("apply", "-n", NS, "-f",
             f"{ROOT}/deploy/k8s/examples/demo-powerpolicy.yaml")

    def _params():
        out = _kubectl("get", "configmap", "opendps-topology-demo", "-n", NS,
                       "-o", "jsonpath={.data.params\\.json}", check=False).stdout
        return out if "cap_raise_rate_w_per_tick" in out else None

    raw = _wait_for(_params)
    assert raw, "params.json never appeared in the domain ConfigMap"
    params = json.loads(raw)
    assert params["cap_raise_rate_w_per_tick"] == 50.0
    assert params["ewma_alpha"] == 0.5


def test_n22_job_policy_hot_reloads_controller_without_restart():
    controller = _wait_for(_controller_pod, timeout=60)
    assert controller, "opendps-controller never became Ready"
    controller_name = controller["metadata"]["name"]
    controller_uid = controller["metadata"]["uid"]
    controller_node = controller["spec"]["nodeName"]
    initial_restarts = controller["status"]["containerStatuses"][0]["restartCount"]
    pod_name = "n22-live-workload"
    policy_name = "n22-live-priority"

    _kubectl("delete", "jobpowerpolicy", policy_name, "-n", NS, "--ignore-not-found")
    _kubectl("delete", "pod", pod_name, "-n", NS, "--ignore-not-found", "--wait=true")
    workload = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": NS,
            "labels": {"opendps.io/workload": "n22-live"},
        },
        "spec": {
            "nodeName": controller_node,
            "containers": [
                {"name": "pause", "image": "registry.k8s.io/pause:3.10"}
            ],
        },
    }
    policy = {
        "apiVersion": "opendps.io/v1alpha1",
        "kind": "JobPowerPolicy",
        "metadata": {"name": policy_name, "namespace": NS},
        "spec": {
            "matchLabels": {"opendps.io/workload": "n22-live"},
            "gpuBoostPct": 20.0,
            "priorityClass": "high",
        },
    }

    try:
        _apply_json(workload)
        _apply_json(policy)

        def _matched_without_identity():
            matched = _kubectl(
                "get", "jobpowerpolicy", policy_name, "-n", NS,
                "-o", "jsonpath={.status.matchedPods}", check=False,
            ).stdout
            return matched == "1" and _resolved_assignments() == []

        assert _wait_for(_matched_without_identity), (
            "policy did not match the unannotated Pod with an empty assignment set"
        )
        _kubectl(
            "annotate", "pod", pod_name, "-n", NS, "opendps.io/gpu-indices=0"
        )

        def _high_assignment():
            assignments = _resolved_assignments()
            if not assignments or len(assignments) != 1:
                return None
            assignment = assignments[0]
            return assignment if (
                assignment["nodeName"] == controller_node
                and assignment["gpuIndex"] == 0
                and assignment["priorityClass"] == "high"
            ) else None

        assert _wait_for(_high_assignment), "late GPU annotation was not resolved"
        _kubectl(
            "patch", "jobpowerpolicy", policy_name, "-n", NS,
            "--type=merge", "-p", '{"spec":{"priorityClass":"critical"}}',
        )

        def _controller_loaded_critical():
            config = _priority_config()
            if not config:
                return None
            resource_version, assignments = config
            if not assignments or assignments[0].get("priorityClass") != "critical":
                return None
            logs = _kubectl("logs", controller_name, "-n", NS, check=False).stdout
            markers = [
                line for line in logs.splitlines()
                if "Priority configuration applied:" in line
            ]
            expected_version = f"resourceVersion={resource_version} "
            return markers[-1] if (
                markers
                and expected_version in markers[-1]
                and "tiers=0=critical" in markers[-1]
            ) else None

        assert _wait_for(_controller_loaded_critical), (
            "controller did not report applying the critical tier"
        )
        current = _wait_for(_controller_pod, timeout=30)
        assert current, "opendps-controller Pod disappeared during the test"
        assert current["metadata"]["uid"] == controller_uid
        assert current["status"]["containerStatuses"][0]["restartCount"] == initial_restarts

        _kubectl("delete", "pod", pod_name, "-n", NS, "--wait=true")

        def _controller_loaded_empty():
            config = _priority_config()
            if not config:
                return None
            resource_version, assignments = config
            if assignments != []:
                return None
            logs = _kubectl("logs", controller_name, "-n", NS, check=False).stdout
            expected_version = f"resourceVersion={resource_version} "
            return any(
                expected_version in line and "tiers=none" in line
                for line in logs.splitlines()
                if "Priority configuration applied:" in line
            )

        assert _wait_for(_controller_loaded_empty), (
            "controller did not apply the empty snapshot after Pod deletion"
        )
    finally:
        _kubectl("delete", "jobpowerpolicy", policy_name, "-n", NS, "--ignore-not-found")
        _kubectl(
            "delete", "pod", pod_name, "-n", NS, "--ignore-not-found", "--wait=true"
        )


def test_n22_controller_service_account_has_read_only_configmap_access():
    subject = "system:serviceaccount:opendps:opendps-controller"
    can_get = _kubectl(
        "auth", "can-i", "get", "configmap/opendps-job-boosts",
        "-n", NS, "--as", subject,
    ).stdout.strip()
    cannot_list = _kubectl(
        "auth", "can-i", "list", "configmaps", "-n", NS, "--as", subject,
        check=False,
    ).stdout.strip()
    cannot_get_other = _kubectl(
        "auth", "can-i", "get", "configmap/opendps-topology-demo",
        "-n", NS, "--as", subject, check=False,
    ).stdout.strip()
    assert can_get == "yes"
    assert cannot_list == "no"
    assert cannot_get_other == "no"
