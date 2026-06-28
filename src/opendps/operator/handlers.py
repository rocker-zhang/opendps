"""opendps Kubernetes operator — reconciles PowerDomain, PowerPolicy, JobPowerPolicy CRDs."""
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path

import kopf
import kubernetes

log = logging.getLogger(__name__)

# Config map name where the controller reads its topology
CONFIG_MAP_NAME = os.getenv("OPENDPS_CONFIGMAP", "opendps-topology")
NAMESPACE = os.getenv("OPENDPS_NAMESPACE", "opendps")


# ---------------------------------------------------------------------------
# PowerDomain handlers
# ---------------------------------------------------------------------------

@kopf.on.create("opendps.io", "v1alpha1", "powerdomains")
@kopf.on.update("opendps.io", "v1alpha1", "powerdomains")
def on_powerdomain_change(spec, name, namespace, status, patch, **kwargs):
    """Reconcile PowerDomain → update topology ConfigMap."""
    gpu_indices = list(spec.get("gpuIndices", []))
    budget_w = float(spec.get("budgetWatts", 1000.0))
    overhead_w = float(spec.get("nodeOverheadWatts", 0.0))

    if not gpu_indices:
        raise kopf.PermanentError("gpuIndices must be non-empty")
    if budget_w <= 0:
        raise kopf.PermanentError("budgetWatts must be positive")

    topology = _build_topology(name, gpu_indices, budget_w, overhead_w)
    _upsert_configmap(namespace, name, topology)

    patch.status["phase"] = "Active"
    patch.status["lastUpdated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log.info("PowerDomain %s/%s reconciled: %d GPUs @ %.0fW", namespace, name, len(gpu_indices), budget_w)


@kopf.on.delete("opendps.io", "v1alpha1", "powerdomains")
def on_powerdomain_delete(name, namespace, **kwargs):
    """Clean up ConfigMap when domain is deleted."""
    try:
        v1 = kubernetes.client.CoreV1Api()
        v1.delete_namespaced_config_map(f"{CONFIG_MAP_NAME}-{name}", namespace)
        log.info("Deleted ConfigMap for PowerDomain %s/%s", namespace, name)
    except kubernetes.client.exceptions.ApiException as e:
        if e.status != 404:
            raise


# ---------------------------------------------------------------------------
# PowerPolicy handlers
# ---------------------------------------------------------------------------

@kopf.on.create("opendps.io", "v1alpha1", "powerpolicies")
@kopf.on.update("opendps.io", "v1alpha1", "powerpolicies")
def on_powerpolicy_change(spec, name, namespace, patch, **kwargs):
    domain_ref = spec["domainRef"]
    brain = spec.get("brain", "prs")
    interval = float(spec.get("intervalSeconds", 5.0))

    # Annotate the corresponding domain ConfigMap with brain selection
    _annotate_domain_configmap(namespace, domain_ref, {
        "opendps.io/brain": brain,
        "opendps.io/interval": str(interval),
    })

    patch.status["active"] = True
    patch.status["lastDecisionTs"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log.info("PowerPolicy %s/%s: domain=%s brain=%s", namespace, name, domain_ref, brain)


# ---------------------------------------------------------------------------
# JobPowerPolicy handlers
# ---------------------------------------------------------------------------

@kopf.on.create("opendps.io", "v1alpha1", "jobpowerpolicies")
@kopf.on.update("opendps.io", "v1alpha1", "jobpowerpolicies")
def on_jobpowerpolicy_change(spec, name, namespace, patch, **kwargs):
    match_labels = spec.get("matchLabels", {})
    boost_pct = float(spec.get("gpuBoostPct", 15.0))
    priority = spec.get("priorityClass", "normal")

    log.info(
        "JobPowerPolicy %s/%s: labels=%s boost=%.0f%% priority=%s",
        namespace, name, match_labels, boost_pct, priority,
    )
    patch.status["matchedPods"] = 0
    patch.status["activeBoosts"] = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_topology(domain_name: str, gpu_indices: list[int], budget_w: float, overhead_w: float) -> dict:
    return {
        "pdus": {"pdu0": {"name": "pdu0", "capacity_w": budget_w * 1.2}},
        "domains": {
            domain_name: {
                "name": domain_name,
                "pdu_name": "pdu0",
                "gpu_indices": gpu_indices,
                "budget_w": budget_w,
                "node_overhead_w": overhead_w,
            }
        },
    }


def _upsert_configmap(namespace: str, domain_name: str, topology: dict) -> None:
    v1 = kubernetes.client.CoreV1Api()
    cm_name = f"{CONFIG_MAP_NAME}-{domain_name}"
    data = {"topology.json": json.dumps(topology, indent=2)}
    body = kubernetes.client.V1ConfigMap(
        metadata=kubernetes.client.V1ObjectMeta(name=cm_name, namespace=namespace),
        data=data,
    )
    try:
        v1.replace_namespaced_config_map(cm_name, namespace, body)
    except kubernetes.client.exceptions.ApiException as e:
        if e.status == 404:
            v1.create_namespaced_config_map(namespace, body)
        else:
            raise


def _annotate_domain_configmap(namespace: str, domain_name: str, annotations: dict[str, str]) -> None:
    v1 = kubernetes.client.CoreV1Api()
    cm_name = f"{CONFIG_MAP_NAME}-{domain_name}"
    try:
        v1.patch_namespaced_config_map(cm_name, namespace, {
            "metadata": {"annotations": annotations}
        })
    except kubernetes.client.exceptions.ApiException as e:
        if e.status == 404:
            log.warning("ConfigMap %s not found for annotation; create PowerDomain first", cm_name)
        else:
            raise
