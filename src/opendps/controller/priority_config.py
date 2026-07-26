"""Kubernetes-backed priority-tier configuration for the controller."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

log = logging.getLogger(__name__)

CONFIG_MAP_NAME = "opendps-job-boosts"
ASSIGNMENTS_KEY = "resolved-assignments.json"
SCHEMA_VERSION = 1
VALID_TIERS = frozenset({"low", "normal", "high", "critical"})
TIER_RANK = {"low": 0, "normal": 1, "high": 2, "critical": 3}


def dynamic_priority_config_from_env(
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str | None]:
    """Return the explicit dynamic-config switch and local node name.

    Only the literals ``true`` and ``false`` are accepted so unrelated or
    misspelled environment values cannot silently weaken CLI validation.
    """
    env = os.environ if environ is None else environ
    raw = env.get("OPENDPS_PRIORITY_CONFIG_ENABLED", "false")
    normalized = raw.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError("OPENDPS_PRIORITY_CONFIG_ENABLED must be 'true' or 'false'")
    enabled = normalized == "true"
    if not enabled:
        return False, None
    node_name = env.get("OPENDPS_NODE_NAME", "").strip()
    if not node_name:
        raise ValueError("OPENDPS_NODE_NAME is required when dynamic priority config is enabled")
    return True, node_name


class PriorityConfigSource:
    """Poll a resolved-assignment ConfigMap and expose an immutable snapshot.

    The Kubernetes dependency is loaded lazily so non-Kubernetes deployments
    retain their existing CLI-only behaviour.
    """

    def __init__(
        self,
        *,
        namespace: str,
        node_name: str,
        baseline: Mapping[int, str] | None = None,
        api: Any | None = None,
    ) -> None:
        self._namespace = namespace
        self._node_name = node_name
        self._baseline = dict(baseline or {})
        self._snapshot: Mapping[int, str] = MappingProxyType(dict(self._baseline))
        self._resource_version: str | None = None
        self._api = api
        self._disabled = False
        self._reported_unavailable = False

    def snapshot(self) -> Mapping[int, str]:
        """Return the current immutable priority-tier snapshot."""
        return self._snapshot

    def poll(self) -> Mapping[int, str]:
        """Refresh once, preserving the last-known-good snapshot on bad data."""
        api = self._get_api()
        if api is None:
            return self._snapshot

        try:
            config_map = api.read_namespaced_config_map(
                name=CONFIG_MAP_NAME,
                namespace=self._namespace,
            )
        except Exception as exc:
            status = getattr(exc, "status", None)
            if status == 404:
                self._resource_version = None
                self._snapshot = MappingProxyType(dict(self._baseline))
                return self._snapshot
            if status in (401, 403):
                log.warning(
                    "Cannot read priority ConfigMap %s/%s (HTTP %s); "
                    "keeping the last-known-good configuration",
                    self._namespace,
                    CONFIG_MAP_NAME,
                    status,
                )
            else:
                log.debug("Priority ConfigMap refresh failed: %s", exc)
            return self._snapshot

        metadata = getattr(config_map, "metadata", None)
        resource_version = getattr(metadata, "resource_version", None)
        if resource_version and resource_version == self._resource_version:
            return self._snapshot

        data = getattr(config_map, "data", None) or {}
        raw = data.get(ASSIGNMENTS_KEY)
        try:
            dynamic = self._parse(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            log.warning(
                "Ignoring malformed %s in ConfigMap %s/%s: %s",
                ASSIGNMENTS_KEY,
                self._namespace,
                CONFIG_MAP_NAME,
                exc,
            )
            return self._snapshot

        merged = dict(self._baseline)
        merged.update(dynamic)
        self._snapshot = MappingProxyType(merged)
        self._resource_version = resource_version
        return self._snapshot

    def _get_api(self) -> Any | None:
        if self._disabled:
            return None
        if self._api is not None:
            return self._api
        try:
            from kubernetes import client, config

            config.load_incluster_config()
            self._api = client.CoreV1Api()
        except Exception as exc:
            self._disabled = True
            if not self._reported_unavailable:
                log.info(
                    "Kubernetes priority configuration unavailable; using CLI priority tiers: %s",
                    exc,
                )
                self._reported_unavailable = True
        return self._api

    def _parse(self, raw: str | None) -> dict[int, str]:
        if raw is None:
            raise ValueError(f"missing data key {ASSIGNMENTS_KEY!r}")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        if payload.get("schemaVersion") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schemaVersion {payload.get('schemaVersion')!r}")
        assignments = payload.get("assignments")
        if not isinstance(assignments, list):
            raise ValueError("assignments must be a list")

        selected: dict[int, tuple[int, float, str]] = {}
        for assignment in assignments:
            if not isinstance(assignment, dict):
                raise ValueError("each assignment must be an object")
            if assignment.get("nodeName") != self._node_name:
                continue
            gpu_index = assignment.get("gpuIndex")
            tier = assignment.get("priorityClass")
            boost = assignment.get("gpuBoostPct")
            if isinstance(gpu_index, bool) or not isinstance(gpu_index, int):
                raise ValueError("gpuIndex must be an integer")
            if gpu_index < 0:
                raise ValueError("gpuIndex must be non-negative")
            if tier not in VALID_TIERS:
                raise ValueError(f"invalid priorityClass {tier!r}")
            if isinstance(boost, bool) or not isinstance(boost, (int, float)):
                raise ValueError("gpuBoostPct must be a number")

            candidate = (TIER_RANK[tier], float(boost), tier)
            current = selected.get(gpu_index)
            if current is None or candidate[:2] > current[:2]:
                selected[gpu_index] = candidate

        return {gpu: candidate[2] for gpu, candidate in selected.items()}
