from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

from opendps.telemetry.model import GpuSample, NodeSample


class PromClient:
    def __init__(self, url: str = "http://localhost:9090") -> None:
        self._base = url.rstrip("/")

    def query(self, promql: str) -> list[dict[str, Any]]:
        """Instant vector query. Returns [{metric: {labels}, value: float}]."""
        params = urllib.parse.urlencode({"query": promql})
        with urllib.request.urlopen(f"{self._base}/api/v1/query?{params}") as resp:
            data = json.loads(resp.read())
        return _parse_vector(data)

    def query_range(
        self,
        promql: str,
        start: float,
        end: float,
        step: str = "5s",
    ) -> list[dict[str, Any]]:
        """Range matrix query. Returns [{metric: {labels}, values: [(ts, float)]}]."""
        params = urllib.parse.urlencode({"query": promql, "start": start, "end": end, "step": step})
        with urllib.request.urlopen(f"{self._base}/api/v1/query_range?{params}") as resp:
            data = json.loads(resp.read())
        return _parse_matrix(data)


def _parse_vector(data: dict) -> list[dict[str, Any]]:
    return [
        {"metric": r["metric"], "value": float(r["value"][1])}
        for r in data.get("data", {}).get("result", [])
    ]


def _parse_matrix(data: dict) -> list[dict[str, Any]]:
    return [
        {
            "metric": r["metric"],
            "values": [(float(ts), float(v)) for ts, v in r["values"]],
        }
        for r in data.get("data", {}).get("result", [])
    ]


def NodeSampleFromProm(client: PromClient, hostname: str | None = None) -> NodeSample:
    """Query 5 DCGM metrics and build a NodeSample. Uses hostname label to filter
    when multiple nodes export to the same Prometheus."""
    power_rows = client.query("DCGM_FI_DEV_POWER_USAGE")
    cap_rows = client.query("DCGM_FI_DEV_POWER_CAP")
    clock_rows = client.query("DCGM_FI_DEV_SM_CLOCK")
    util_rows = client.query("DCGM_FI_DEV_GPU_UTIL")
    temp_rows = client.query("DCGM_FI_DEV_GPU_TEMP")  # N16 — thermal signal

    def _index(rows: list[dict], hn_filter: str | None) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in rows:
            if hn_filter is not None and row["metric"].get("hostname") != hn_filter:
                continue
            out[row["metric"].get("gpu", "0")] = row["value"]
        return out

    power_draw = _index(power_rows, hostname)
    power_cap = _index(cap_rows, hostname)
    sm_clocks = _index(clock_rows, hostname)
    util = _index(util_rows, hostname)
    temps = _index(temp_rows, hostname)

    resolved_hostname = hostname or "unknown"
    model_name = "unknown"
    for row in power_rows:
        if hostname is None or row["metric"].get("hostname") == hostname:
            resolved_hostname = row["metric"].get("hostname", "unknown")
            model_name = row["metric"].get("modelName", "unknown")
            break

    all_gpus = sorted(
        set(power_draw) | set(power_cap) | set(sm_clocks) | set(util) | set(temps),
        key=lambda x: int(x),
    )

    gpus = [
        GpuSample(
            index=int(gpu_idx),
            name=model_name,
            power_draw_w=power_draw.get(gpu_idx),
            power_limit_w=power_cap.get(gpu_idx),
            sm_clock_mhz=int(sm_clocks[gpu_idx]) if gpu_idx in sm_clocks else None,
            gpu_util_pct=int(util[gpu_idx]) if gpu_idx in util else None,
            temperature_c=int(temps[gpu_idx]) if gpu_idx in temps else None,
        )
        for gpu_idx in all_gpus
    ]

    return NodeSample(ts=time.time(), hostname=resolved_hostname, driver_version="dcgm", gpus=gpus)
