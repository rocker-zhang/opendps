"""opendps Kubernetes operator — reconciles PowerDomain, PowerPolicy, JobPowerPolicy CRDs."""
from __future__ import annotations
import json
import logging
import os
import time

import kopf
import kubernetes

log = logging.getLogger(__name__)

# Config map name where the controller reads its topology
CONFIG_MAP_NAME = os.getenv("OPENDPS_CONFIGMAP", "opendps-topology")
NAMESPACE = os.getenv("OPENDPS_NAMESPACE", "opendps")
# Config map holding all active JobPowerPolicy boosts (keyed by policy name)
BOOST_CONFIG_MAP_NAME = os.getenv("OPENDPS_BOOST_CONFIGMAP", "opendps-job-boosts")


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

    # N5 — brain/failsafe params. Write these into the domain ConfigMap *data*
    # (params.json) so a controller that mounts the ConfigMap reads them as a
    # file. Annotations alone are not visible through a volume mount, so the
    # propagation path must go through data, not metadata.
    params = {
        "brain": brain,
        "interval_s": interval,
        "cap_raise_rate_w_per_tick": float(spec.get("capRaiseRateWattsPerTick", 50.0)),
        "ewma_alpha": float(spec.get("ewmaAlpha", 0.3)),
    }
    if "failsafeThresholdWatts" in spec:
        params["failsafe_threshold_w"] = float(spec["failsafeThresholdWatts"])
    if "failsafeEmergencyCapWatts" in spec:
        params["failsafe_emergency_cap_w"] = float(spec["failsafeEmergencyCapWatts"])

    wrote = _write_domain_params(namespace, domain_ref, params)

    # Keep the lightweight annotation too (handy for `kubectl describe`).
    _annotate_domain_configmap(namespace, domain_ref, {
        "opendps.io/brain": brain,
        "opendps.io/interval": str(interval),
    })

    patch.status["active"] = bool(wrote)
    patch.status["lastDecisionTs"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log.info(
        "PowerPolicy %s/%s: domain=%s brain=%s capRaiseRate=%.0f ewmaAlpha=%.2f (params written=%s)",
        namespace, name, domain_ref, brain,
        params["cap_raise_rate_w_per_tick"], params["ewma_alpha"], wrote,
    )


# ---------------------------------------------------------------------------
# JobPowerPolicy handlers
# ---------------------------------------------------------------------------

@kopf.on.create("opendps.io", "v1alpha1", "jobpowerpolicies")
@kopf.on.update("opendps.io", "v1alpha1", "jobpowerpolicies")
def on_jobpowerpolicy_change(spec, name, namespace, patch, **kwargs):
    match_labels = spec.get("matchLabels", {})
    boost_pct = float(spec.get("gpuBoostPct", 15.0))
    priority = spec.get("priorityClass", "normal")

    # Count pods that actually match the selector. This runs inside the operator
    # pod and only touches the k8s API (pods list) — it must NOT call nvidia-smi,
    # which is unavailable in the driverless operator container. The GPU↔job
    # binding is done node-side by the agent's JobTracker; here we only resolve
    # how many workloads the policy currently applies to.
    matched = _count_matching_pods(namespace, match_labels)

    # Publish the boost policy to a ConfigMap the controller reads, so a busy
    # GPU running a matched job gets its cap boosted (consumed by JobAwarePRSBrain).
    _write_boost_registry(namespace, name, {
        "matchLabels": match_labels,
        "gpu_boost_pct": boost_pct,
        "priority": priority,
        "matched_pods": matched,
    })

    active = matched if boost_pct > 0.0 else 0
    patch.status["matchedPods"] = matched
    patch.status["activeBoosts"] = active
    log.info(
        "JobPowerPolicy %s/%s: labels=%s boost=%.0f%% priority=%s matchedPods=%d activeBoosts=%d",
        namespace, name, match_labels, boost_pct, priority, matched, active,
    )


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
    topology_json = json.dumps(topology, indent=2)
    # Merge-patch only the topology.json key so a sibling params.json written by
    # the PowerPolicy handler is preserved (replace_* would wipe it).
    _patch_or_create_cm(v1, namespace, cm_name, {"topology.json": topology_json})


def _patch_or_create_cm(v1, namespace: str, cm_name: str, data: dict) -> None:
    """Merge-patch the given data keys into a ConfigMap, creating it if absent.

    Tolerates the create/patch race: if a concurrent handler creates the
    ConfigMap first (409), re-patch so our data keys still land (a plain
    tolerated 409 would drop them when the other create used different keys).
    """
    try:
        v1.patch_namespaced_config_map(cm_name, namespace, {"data": data})
        return
    except kubernetes.client.exceptions.ApiException as e:
        if e.status != 404:
            raise
    body = kubernetes.client.V1ConfigMap(
        metadata=kubernetes.client.V1ObjectMeta(name=cm_name, namespace=namespace),
        data=data,
    )
    try:
        v1.create_namespaced_config_map(namespace, body)
    except kubernetes.client.exceptions.ApiException as ce:
        if ce.status != 409:  # created concurrently — fall through and patch
            raise
        v1.patch_namespaced_config_map(cm_name, namespace, {"data": data})


def _count_matching_pods(namespace: str, match_labels: dict) -> int:
    """Count pods in the namespace matching the given label selector.

    In-pod safe: only calls the k8s API (pods list), never nvidia-smi. Returns 0
    when no labels are given. Auth/permission failures (401/403) and other
    non-transient API errors are re-raised so kopf surfaces and retries them
    rather than silently reporting zero matches; only transient connection
    blips degrade to 0.
    """
    if not match_labels:
        return 0
    selector = ",".join(f"{k}={v}" for k, v in match_labels.items())
    try:
        v1 = kubernetes.client.CoreV1Api()
        pods = v1.list_namespaced_pod(namespace, label_selector=selector)
        return len(pods.items)
    except kubernetes.client.exceptions.ApiException as e:
        if e.status in (401, 403, 404, 422):  # auth/permission/bad-request: surface it
            raise
        log.warning("pod list failed for selector %r: %s", selector, e)
        return 0
    except Exception as e:  # transient connection error — degrade to 0
        log.warning("pod list connection error for selector %r: %s", selector, e)
        return 0


def _write_boost_registry(namespace: str, policy_name: str, entry: dict) -> None:
    """Publish one JobPowerPolicy's boost entry into the shared boost-registry
    ConfigMap (merge-patch on the policy-named key; preserves sibling policies).

    This is the published, k8s-native record of the boost policy (and an audit
    artifact). The process-mode controller derives live per-GPU boosts from its
    JobTracker, not from this ConfigMap; a future in-cluster controller would
    consume it directly.
    """
    v1 = kubernetes.client.CoreV1Api()
    _patch_or_create_cm(
        v1, namespace, BOOST_CONFIG_MAP_NAME,
        {f"{policy_name}.json": json.dumps(entry, indent=2)},
    )


def _write_domain_params(namespace: str, domain_name: str, params: dict) -> bool:
    """Write PowerPolicy-derived brain params into the domain ConfigMap's
    ``params.json`` data key (merge-patch, preserves topology.json).

    Returns True on success, False if the ConfigMap does not exist yet (the
    PowerDomain must be reconciled first). Mirrors the controller-side reader in
    standalone._load_brain_params.

    Raises kopf.TemporaryError on 404 so a PowerPolicy reconciled before its
    PowerDomain retries (rather than silently dropping the params forever).
    """
    v1 = kubernetes.client.CoreV1Api()
    cm_name = f"{CONFIG_MAP_NAME}-{domain_name}"
    try:
        v1.patch_namespaced_config_map(
            cm_name, namespace, {"data": {"params.json": json.dumps(params, indent=2)}}
        )
        return True
    except kubernetes.client.exceptions.ApiException as e:
        if e.status == 404:
            raise kopf.TemporaryError(
                f"ConfigMap {cm_name} not ready; create PowerDomain first", delay=15
            ) from e
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
