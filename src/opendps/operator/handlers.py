"""opendps Kubernetes operator — reconciles PowerDomain, PowerPolicy, JobPowerPolicy CRDs."""

from __future__ import annotations
import json
import logging
import os
import random
import threading
import time
from typing import Any

import kopf
import kubernetes

log = logging.getLogger(__name__)

# Config map name where the controller reads its topology
CONFIG_MAP_NAME = os.getenv("OPENDPS_CONFIGMAP", "opendps-topology")
NAMESPACE = os.getenv("OPENDPS_NAMESPACE", "opendps")
# Config map holding all active JobPowerPolicy boosts (keyed by policy name)
BOOST_CONFIG_MAP_NAME = os.getenv("OPENDPS_BOOST_CONFIGMAP", "opendps-job-boosts")
GPU_INDICES_ANNOTATION = "opendps.io/gpu-indices"
BOOST_SCHEMA_VERSION = 1
RESOLVED_ASSIGNMENTS_KEY = "resolved-assignments.json"
POLICY_SELECTOR_CACHE_TTL_S = 5.0
# Raw Pod event handlers are silent in Kopf: unlike create/update/delete
# handlers, they do not write progress or finalizers back to watched Pods.
# Cache only assignment-relevant fields so status-only MODIFIED events do not
# repeatedly re-resolve every JobPowerPolicy.
_POD_ASSIGNMENT_DIGESTS: dict[
    str,
    tuple[
        tuple[str | None, tuple[tuple[str, str], ...], str | None],
        bool,
        str,
    ],
] = {}
_POD_ASSIGNMENT_DIGESTS_LOCK = threading.Lock()
_POLICY_SELECTOR_CACHE: dict[
    str,
    tuple[float, tuple[tuple[tuple[str, str], ...], ...]],
] = {}


def _pod_gpu_indices(pod: Any) -> list[int] | None:
    """Return explicitly assigned GPU indices, or None when not advertised.

    Kubernetes does not expose device allocation through the Pod API.  The
    operator therefore only publishes identities supplied explicitly by the
    node-side allocator and never guesses from resource requests.
    """
    metadata = getattr(pod, "metadata", None)
    annotations = getattr(metadata, "annotations", None) or {}
    raw = annotations.get(GPU_INDICES_ANNOTATION)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None

    indices: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token or not token.isascii() or not token.isdecimal():
            return None
        index = int(token)
        if index < 0:
            return None
        indices.append(index)
    if len(indices) != len(set(indices)):
        return None
    return sorted(indices)


def _resolved_assignments(
    pods: list[Any],
    *,
    policy_uid: str,
    policy_generation: int,
    priority_class: str,
    boost_pct: float,
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for pod in pods:
        metadata = getattr(pod, "metadata", None)
        spec = getattr(pod, "spec", None)
        pod_uid = getattr(metadata, "uid", None)
        node_name = getattr(spec, "node_name", None)
        if not pod_uid or not node_name:
            continue
        gpu_indices = _pod_gpu_indices(pod)
        if gpu_indices is None:
            continue
        for gpu_index in gpu_indices:
            assignments.append(
                {
                    "gpuBoostPct": boost_pct,
                    "gpuIndex": gpu_index,
                    "nodeName": str(node_name),
                    "podUid": str(pod_uid),
                    "policyGeneration": policy_generation,
                    "policyUid": policy_uid,
                    "priorityClass": priority_class,
                }
            )
    return sorted(
        assignments,
        key=lambda item: (
            item["nodeName"],
            item["gpuIndex"],
            item["podUid"],
            item["policyUid"],
        ),
    )


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
    log.info(
        "PowerDomain %s/%s reconciled: %d GPUs @ %.0fW", namespace, name, len(gpu_indices), budget_w
    )


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
    _annotate_domain_configmap(
        namespace,
        domain_ref,
        {
            "opendps.io/brain": brain,
            "opendps.io/interval": str(interval),
        },
    )

    patch.status["active"] = bool(wrote)
    patch.status["lastDecisionTs"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log.info(
        "PowerPolicy %s/%s: domain=%s brain=%s capRaiseRate=%.0f ewmaAlpha=%.2f (params written=%s)",
        namespace,
        name,
        domain_ref,
        brain,
        params["cap_raise_rate_w_per_tick"],
        params["ewma_alpha"],
        wrote,
    )


# ---------------------------------------------------------------------------
# JobPowerPolicy handlers
# ---------------------------------------------------------------------------


@kopf.on.create("opendps.io", "v1alpha1", "jobpowerpolicies")
@kopf.on.update("opendps.io", "v1alpha1", "jobpowerpolicies")
def on_jobpowerpolicy_change(spec, name, namespace, patch, meta=None, **kwargs):
    _invalidate_policy_selector_cache(namespace)
    matched, active = _reconcile_jobpowerpolicy(
        spec,
        name,
        namespace,
        meta=meta,
    )
    patch.status["matchedPods"] = matched
    patch.status["activeBoosts"] = active


@kopf.timer("opendps.io", "v1alpha1", "jobpowerpolicies", interval=30.0)
def resync_jobpowerpolicy(spec, name, namespace, patch, status=None, meta=None, **kwargs):
    """Periodically compensate for failures in the silent Pod event path."""
    matched, active = _reconcile_jobpowerpolicy(
        spec,
        name,
        namespace,
        meta=meta,
    )
    _prune_pod_assignment_digests(namespace)
    current_status = status or {}
    if current_status.get("matchedPods") != matched:
        patch.status["matchedPods"] = matched
    if current_status.get("activeBoosts") != active:
        patch.status["activeBoosts"] = active


def _reconcile_jobpowerpolicy(
    spec: dict[str, Any],
    name: str,
    namespace: str,
    *,
    meta: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Resolve one policy, publish its assignments, and return status counts."""
    meta = meta or {}
    match_labels = spec.get("matchLabels", {})
    boost_pct = float(spec.get("gpuBoostPct", 15.0))
    priority = spec.get("priorityClass", "normal")

    # Count pods that actually match the selector. This runs inside the operator
    # pod and only touches the k8s API (pods list) — it must NOT call nvidia-smi,
    # which is unavailable in the driverless operator container. The GPU↔job
    # binding is done node-side by the agent's JobTracker; here we only resolve
    # how many workloads the policy currently applies to.
    matched_pod_objects = _list_matching_pods(namespace, match_labels)
    matched = len(matched_pod_objects)
    resolved = _resolved_assignments(
        matched_pod_objects,
        policy_uid=str(meta.get("uid") or ""),
        policy_generation=int(meta.get("generation") or 0),
        priority_class=priority,
        boost_pct=boost_pct,
    )

    # Publish the boost policy to a ConfigMap the controller reads, so a busy
    # GPU running a matched job gets its cap boosted (consumed by JobAwarePRSBrain).
    _write_boost_registry(
        namespace,
        name,
        {
            "matchLabels": match_labels,
            "gpu_boost_pct": boost_pct,
            "priority": priority,
            "matched_pods": matched,
            "assignments": resolved,
            "schemaVersion": BOOST_SCHEMA_VERSION,
        },
    )

    active = matched if boost_pct > 0.0 else 0
    log.info(
        "JobPowerPolicy %s/%s: labels=%s boost=%.0f%% priority=%s matchedPods=%d activeBoosts=%d",
        namespace,
        name,
        match_labels,
        boost_pct,
        priority,
        matched,
        active,
    )
    return matched, active


@kopf.on.delete("opendps.io", "v1alpha1", "jobpowerpolicies")
def on_jobpowerpolicy_delete(name, namespace, **kwargs):
    """Remove the deleted policy without disturbing sibling registry entries."""
    _invalidate_policy_selector_cache(namespace)
    _write_boost_registry(namespace, name, None)


def _pod_assignment_digest(
    body: dict[str, Any],
) -> tuple[str | None, tuple[tuple[str, str], ...], str | None]:
    metadata = body.get("metadata") or {}
    annotations = metadata.get("annotations") or {}
    labels = metadata.get("labels") or {}
    spec = body.get("spec") or {}
    return (
        annotations.get(GPU_INDICES_ANNOTATION),
        tuple(sorted((str(key), str(value)) for key, value in labels.items())),
        spec.get("nodeName"),
    )


def _pod_matches_assignment_policy(body: dict[str, Any], namespace: str) -> bool:
    annotation, label_items, _node_name = _pod_assignment_digest(body)
    if annotation is not None:
        return True
    labels = dict(label_items)
    if not labels:
        return False
    for selector_items in _policy_selectors(namespace):
        if all(labels.get(key) == value for key, value in selector_items):
            return True
    return False


def _invalidate_policy_selector_cache(namespace: str) -> None:
    _POLICY_SELECTOR_CACHE.pop(namespace, None)


def _policy_selectors(namespace: str) -> tuple[tuple[tuple[str, str], ...], ...]:
    now = time.monotonic()
    cached = _POLICY_SELECTOR_CACHE.get(namespace)
    if cached is not None and cached[0] > now:
        return cached[1]
    custom = kubernetes.client.CustomObjectsApi()
    policies = custom.list_namespaced_custom_object(
        "opendps.io",
        "v1alpha1",
        namespace,
        "jobpowerpolicies",
    )
    selectors = tuple(
        tuple(sorted((str(key), str(value)) for key, value in selector.items()))
        for policy in policies.get("items", [])
        if (selector := policy.get("spec", {}).get("matchLabels", {}))
    )
    _POLICY_SELECTOR_CACHE[namespace] = (
        now + POLICY_SELECTOR_CACHE_TTL_S,
        selectors,
    )
    return selectors


def _prune_pod_assignment_digests(namespace: str) -> None:
    """Remove cached UIDs that disappeared while the Pod watch was disrupted."""
    v1 = kubernetes.client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace)
    live_uids = {
        str(uid)
        for pod in pods.items
        if (uid := getattr(getattr(pod, "metadata", None), "uid", None))
    }
    with _POD_ASSIGNMENT_DIGESTS_LOCK:
        stale = [
            pod_uid
            for pod_uid, (
                _digest,
                _relevant,
                cached_namespace,
            ) in _POD_ASSIGNMENT_DIGESTS.items()
            if cached_namespace == namespace and pod_uid not in live_uids
        ]
        for pod_uid in stale:
            _POD_ASSIGNMENT_DIGESTS.pop(pod_uid, None)


def _reconcile_jobpowerpolicies_for_namespace(namespace: str) -> None:
    custom = kubernetes.client.CustomObjectsApi()
    policies = custom.list_namespaced_custom_object(
        "opendps.io",
        "v1alpha1",
        namespace,
        "jobpowerpolicies",
    )
    for policy in sorted(
        policies.get("items", []),
        key=lambda item: item.get("metadata", {}).get("name", ""),
    ):
        metadata = policy.get("metadata", {})
        name = metadata.get("name")
        if not name:
            continue
        _reconcile_jobpowerpolicy(
            policy.get("spec", {}),
            name,
            namespace,
            meta=metadata,
        )


@kopf.on.event("", "v1", "pods")
def on_pod_event(event, namespace, **kwargs):
    """Silently re-resolve policies after assignment-relevant Pod events."""
    event = event if isinstance(event, dict) else {}
    event_type = str(event.get("type") or "").upper()
    body = event.get("object") or event.get("body") or {}
    if not isinstance(body, dict):
        return
    metadata = body.get("metadata") or {}
    pod_uid = metadata.get("uid")
    if not pod_uid or not namespace:
        return
    pod_uid = str(pod_uid)
    with _POD_ASSIGNMENT_DIGESTS_LOCK:
        previous = _POD_ASSIGNMENT_DIGESTS.get(pod_uid)

    if event_type == "DELETED":
        try:
            currently_relevant = _pod_matches_assignment_policy(body, namespace)
            relevant = currently_relevant or (previous is not None and previous[1])
            if relevant:
                _reconcile_jobpowerpolicies_for_namespace(namespace)
        finally:
            with _POD_ASSIGNMENT_DIGESTS_LOCK:
                _POD_ASSIGNMENT_DIGESTS.pop(pod_uid, None)
        return

    digest = _pod_assignment_digest(body)
    if previous is not None and previous[0] == digest:
        return
    relevant = _pod_matches_assignment_policy(body, namespace)
    if relevant or (previous is not None and previous[1]):
        _reconcile_jobpowerpolicies_for_namespace(namespace)
    with _POD_ASSIGNMENT_DIGESTS_LOCK:
        _POD_ASSIGNMENT_DIGESTS[pod_uid] = (digest, relevant, namespace)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_topology(
    domain_name: str, gpu_indices: list[int], budget_w: float, overhead_w: float
) -> dict:
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


def _list_matching_pods(namespace: str, match_labels: dict) -> list[Any]:
    """List pods in the namespace matching the given label selector.

    In-pod safe: only calls the k8s API (pods list), never nvidia-smi. Returns []
    when no labels are given. API and connection failures are propagated so a
    transient outage can never be published as an empty assignment set.
    """
    if not match_labels:
        return []
    selector = ",".join(f"{key}={value}" for key, value in sorted(match_labels.items()))
    try:
        v1 = kubernetes.client.CoreV1Api()
        pods = v1.list_namespaced_pod(namespace, label_selector=selector)
        return list(pods.items)
    except kubernetes.client.exceptions.ApiException as e:
        if e.status in (401, 403, 404, 422):  # auth/permission/bad-request: surface it
            raise
        raise kopf.TemporaryError(
            f"pod list failed for selector {selector!r}: {e}", delay=15
        ) from e
    except Exception as e:
        raise kopf.TemporaryError(
            f"pod list connection error for selector {selector!r}: {e}", delay=15
        ) from e


def _count_matching_pods(namespace: str, match_labels: dict) -> int:
    """Return the number of matching pods for legacy callers and tests."""
    return len(_list_matching_pods(namespace, match_labels))


def _aggregate_resolved_assignments(data: dict[str, str]) -> str:
    assignments: list[dict[str, Any]] = []
    for key, raw in sorted(data.items()):
        if key == RESOLVED_ASSIGNMENTS_KEY or not key.endswith(".json"):
            continue
        try:
            policy = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            log.warning("Ignoring malformed boost registry entry %s", key)
            continue
        values = policy.get("assignments", [])
        if isinstance(values, list):
            assignments.extend(value for value in values if isinstance(value, dict))
    assignments.sort(
        key=lambda value: (
            str(value.get("nodeName", "")),
            (
                value.get("gpuIndex", -1)
                if isinstance(value.get("gpuIndex"), int)
                and not isinstance(value.get("gpuIndex"), bool)
                else -1
            ),
            str(value.get("podUid", "")),
            str(value.get("policyUid", "")),
        )
    )
    return json.dumps(
        {"assignments": assignments, "schemaVersion": BOOST_SCHEMA_VERSION},
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    )


def _write_boost_registry(namespace: str, policy_name: str, entry: dict | None) -> None:
    """Publish one policy entry with a resource-versioned full replacement.

    Each attempt reads the shared ConfigMap, updates the policy-named key and
    aggregate assignment payload, then replaces the complete object using its
    ``resourceVersion``. A missing ConfigMap is created. Conflicts are retried
    with jitter and exhausted retries are rescheduled by kopf.
    """
    v1 = kubernetes.client.CoreV1Api()
    policy_key = f"{policy_name}.json"
    for attempt in range(3):
        try:
            current = v1.read_namespaced_config_map(BOOST_CONFIG_MAP_NAME, namespace)
        except kubernetes.client.exceptions.ApiException as exc:
            if exc.status != 404:
                raise
            data: dict[str, str] = {}
            if entry is not None:
                data[policy_key] = json.dumps(
                    entry, indent=2, sort_keys=True, separators=(",", ": ")
                )
            data[RESOLVED_ASSIGNMENTS_KEY] = _aggregate_resolved_assignments(data)
            body = kubernetes.client.V1ConfigMap(
                metadata=kubernetes.client.V1ObjectMeta(
                    name=BOOST_CONFIG_MAP_NAME, namespace=namespace
                ),
                data=data,
            )
            try:
                v1.create_namespaced_config_map(namespace, body)
                return
            except kubernetes.client.exceptions.ApiException as create_exc:
                if create_exc.status == 409 and attempt < 2:
                    time.sleep(random.uniform(0.01, 0.05))
                    continue
                if create_exc.status == 409:
                    raise kopf.TemporaryError(
                        f"boost registry {namespace}/{BOOST_CONFIG_MAP_NAME} "
                        "remained conflicted after 3 attempts",
                        delay=1,
                    ) from create_exc
                raise

        original_data = dict(current.data or {})
        data = dict(original_data)
        if entry is None:
            data.pop(policy_key, None)
        else:
            data[policy_key] = json.dumps(entry, indent=2, sort_keys=True, separators=(",", ": "))
        data[RESOLVED_ASSIGNMENTS_KEY] = _aggregate_resolved_assignments(data)
        if data == original_data:
            return
        body = kubernetes.client.V1ConfigMap(
            metadata=kubernetes.client.V1ObjectMeta(
                name=BOOST_CONFIG_MAP_NAME,
                namespace=namespace,
                resource_version=current.metadata.resource_version,
            ),
            data=data,
        )
        try:
            v1.replace_namespaced_config_map(BOOST_CONFIG_MAP_NAME, namespace, body)
            return
        except kubernetes.client.exceptions.ApiException as exc:
            if exc.status == 409 and attempt < 2:
                time.sleep(random.uniform(0.01, 0.05))
                continue
            if exc.status == 409:
                raise kopf.TemporaryError(
                    f"boost registry {namespace}/{BOOST_CONFIG_MAP_NAME} "
                    "remained conflicted after 3 attempts",
                    delay=1,
                ) from exc
            raise


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


def _annotate_domain_configmap(
    namespace: str, domain_name: str, annotations: dict[str, str]
) -> None:
    v1 = kubernetes.client.CoreV1Api()
    cm_name = f"{CONFIG_MAP_NAME}-{domain_name}"
    try:
        v1.patch_namespaced_config_map(
            cm_name, namespace, {"metadata": {"annotations": annotations}}
        )
    except kubernetes.client.exceptions.ApiException as e:
        if e.status == 404:
            log.warning("ConfigMap %s not found for annotation; create PowerDomain first", cm_name)
        else:
            raise
