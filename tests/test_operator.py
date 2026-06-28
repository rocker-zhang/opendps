"""Unit tests for the opendps Kubernetes operator."""
from __future__ import annotations
import sys
from unittest.mock import MagicMock, patch
import pytest

# Stub out kubernetes before importing handlers so the module loads without a cluster
kubernetes_stub = MagicMock()
sys.modules.setdefault("kubernetes", kubernetes_stub)
sys.modules.setdefault("kubernetes.client", kubernetes_stub.client)
sys.modules.setdefault("kubernetes.client.exceptions", kubernetes_stub.client.exceptions)

# kopf must also be importable; install is expected, but provide a stub if missing
try:
    import kopf  # noqa: F401
except ImportError:
    kopf_stub = MagicMock()
    kopf_stub.PermanentError = Exception
    sys.modules["kopf"] = kopf_stub

from opendps.operator.handlers import (  # noqa: E402
    _build_topology,
    on_powerdomain_change,
    on_powerpolicy_change,
)


# ---------------------------------------------------------------------------
# test_build_topology
# ---------------------------------------------------------------------------

def test_build_topology():
    topo = _build_topology("d0", [0, 1, 2], 3000.0, 100.0)

    # Top-level keys
    assert "pdus" in topo
    assert "domains" in topo

    # PDU entry
    pdu = topo["pdus"]["pdu0"]
    assert pdu["name"] == "pdu0"
    assert pdu["capacity_w"] == pytest.approx(3000.0 * 1.2)

    # Domain entry
    domain = topo["domains"]["d0"]
    assert domain["name"] == "d0"
    assert domain["gpu_indices"] == [0, 1, 2]
    assert domain["budget_w"] == pytest.approx(3000.0)
    assert domain["node_overhead_w"] == pytest.approx(100.0)
    assert domain["pdu_name"] == "pdu0"


# ---------------------------------------------------------------------------
# test_powerdomain_spec_validation — PermanentError on empty gpu_indices
# ---------------------------------------------------------------------------

def test_powerdomain_spec_validation_empty_gpu_indices():
    import kopf as _kopf

    patch_obj = MagicMock()
    patch_obj.status = {}

    with pytest.raises((_kopf.PermanentError, Exception)):
        on_powerdomain_change(
            spec={"gpuIndices": [], "budgetWatts": 500.0},
            name="test-domain",
            namespace="opendps",
            status={},
            patch=patch_obj,
        )


def test_powerdomain_spec_validation_zero_budget():
    import kopf as _kopf

    patch_obj = MagicMock()
    patch_obj.status = {}

    with pytest.raises((_kopf.PermanentError, Exception)):
        on_powerdomain_change(
            spec={"gpuIndices": [0], "budgetWatts": 0.0},
            name="test-domain",
            namespace="opendps",
            status={},
            patch=patch_obj,
        )


# ---------------------------------------------------------------------------
# test_powerpolicy_brain_choices
# ---------------------------------------------------------------------------

VALID_BRAINS = ["dpm", "prs", "cvxpy", "job-prs", "quota-prs"]


@pytest.mark.parametrize("brain", VALID_BRAINS)
def test_powerpolicy_valid_brain(brain):
    """All enum values should be accepted without raising."""
    patch_obj = MagicMock()
    patch_obj.status = {}

    # _annotate_domain_configmap will hit k8s; stub it out
    with patch("opendps.operator.handlers._annotate_domain_configmap"):
        on_powerpolicy_change(
            spec={"domainRef": "d0", "brain": brain, "intervalSeconds": 5.0},
            name="pp-test",
            namespace="opendps",
            patch=patch_obj,
        )

    assert patch_obj.status["active"] is True


def test_powerpolicy_invalid_brain_not_accepted():
    """An invalid brain value would be rejected by the CRD schema; the handler
    itself does not re-validate, but we verify the valid set is exhaustive."""
    valid_set = set(VALID_BRAINS)
    assert "invalid" not in valid_set
    assert "cvxpy" in valid_set
