"""N4 — opt-in in-cluster integration test for the operator.

Skipped unless OPENDPS_K8S_TEST=1 and kubectl can reach a cluster with the
opendps CRDs installed and the operator running. This is the CI-automatable
equivalent of the manual kind validation: it applies real CRs to a real API
server and asserts the operator reconciles them (no mocks).

Bring-up (see scripts/demo.sh / README):
    docker build -f deploy/operator.Dockerfile -t opendps-operator:latest .
    kind load docker-image opendps-operator:latest
    kubectl create namespace opendps
    kubectl apply -f deploy/k8s/crds/
    kubectl apply -f deploy/k8s/operator-deployment.yaml
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


def _kubectl(*args, check=True):
    return subprocess.run(
        ["kubectl", *args], capture_output=True, text=True, check=check, timeout=60
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
    phase = _wait_for(lambda: _kubectl(
        "get", "powerdomain", "demo", "-n", NS,
        "-o", "jsonpath={.status.phase}", check=False).stdout.strip() or None)
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
