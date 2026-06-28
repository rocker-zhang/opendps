"""Unit tests for PromClient — no real Prometheus needed."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from opendps.telemetry.model import NodeSample
from opendps.telemetry.prom_client import NodeSampleFromProm, PromClient


def _mock_resp(payload: dict):
    m = MagicMock()
    m.read.return_value = json.dumps(payload).encode()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    return m


_VECTOR = {
    "status": "success",
    "data": {
        "resultType": "vector",
        "result": [
            {
                "metric": {"gpu": "0", "modelName": "NVIDIA-SIM-GPU", "hostname": "sim-host-0"},
                "value": [1700000000.0, "350.5"],
            },
            {
                "metric": {"gpu": "1", "modelName": "NVIDIA-SIM-GPU", "hostname": "sim-host-0"},
                "value": [1700000000.0, "220.0"],
            },
        ],
    },
}

_EMPTY = {"status": "success", "data": {"resultType": "vector", "result": []}}

_MATRIX = {
    "status": "success",
    "data": {
        "resultType": "matrix",
        "result": [
            {
                "metric": {"gpu": "0"},
                "values": [[1700000000.0, "300.0"], [1700000005.0, "310.0"]],
            }
        ],
    },
}


def test_query_parses_vector():
    client = PromClient("http://fake:9090")
    with patch("urllib.request.urlopen", return_value=_mock_resp(_VECTOR)):
        results = client.query("DCGM_FI_DEV_POWER_USAGE")
    assert len(results) == 2
    assert results[0]["metric"]["gpu"] == "0"
    assert results[0]["value"] == pytest.approx(350.5)
    assert results[1]["value"] == pytest.approx(220.0)


def test_query_empty_result():
    client = PromClient("http://fake:9090")
    with patch("urllib.request.urlopen", return_value=_mock_resp(_EMPTY)):
        results = client.query("nonexistent")
    assert results == []


def test_query_range_parses_matrix():
    client = PromClient("http://fake:9090")
    with patch("urllib.request.urlopen", return_value=_mock_resp(_MATRIX)):
        results = client.query_range("DCGM_FI_DEV_POWER_USAGE", 1700000000.0, 1700000010.0)
    assert len(results) == 1
    assert results[0]["metric"]["gpu"] == "0"
    assert results[0]["values"][0] == (pytest.approx(1700000000.0), pytest.approx(300.0))
    assert results[0]["values"][1] == (pytest.approx(1700000005.0), pytest.approx(310.0))


def _fake_query_full(promql: str) -> list[dict]:
    _data: dict[str, list[dict]] = {
        "DCGM_FI_DEV_POWER_USAGE": [
            {"metric": {"gpu": "0", "modelName": "NVIDIA-SIM-GPU", "hostname": "sim-host-0"}, "value": 350.5},
            {"metric": {"gpu": "1", "modelName": "NVIDIA-SIM-GPU", "hostname": "sim-host-0"}, "value": 220.0},
        ],
        "DCGM_FI_DEV_POWER_CAP": [
            {"metric": {"gpu": "0", "modelName": "NVIDIA-SIM-GPU", "hostname": "sim-host-0"}, "value": 1000.0},
            {"metric": {"gpu": "1", "modelName": "NVIDIA-SIM-GPU", "hostname": "sim-host-0"}, "value": 1000.0},
        ],
        "DCGM_FI_DEV_SM_CLOCK": [
            {"metric": {"gpu": "0", "modelName": "NVIDIA-SIM-GPU", "hostname": "sim-host-0"}, "value": 1980.0},
            {"metric": {"gpu": "1", "modelName": "NVIDIA-SIM-GPU", "hostname": "sim-host-0"}, "value": 1980.0},
        ],
        "DCGM_FI_DEV_GPU_UTIL": [
            {"metric": {"gpu": "0", "modelName": "NVIDIA-SIM-GPU", "hostname": "sim-host-0"}, "value": 75.0},
            {"metric": {"gpu": "1", "modelName": "NVIDIA-SIM-GPU", "hostname": "sim-host-0"}, "value": 50.0},
        ],
    }
    return _data.get(promql, [])


def test_node_sample_from_prom_builds_correctly():
    client = MagicMock(spec=PromClient)
    client.query.side_effect = _fake_query_full

    node = NodeSampleFromProm(client)

    assert isinstance(node, NodeSample)
    assert node.hostname == "sim-host-0"
    assert len(node.gpus) == 2
    assert node.gpus[0].index == 0
    assert node.gpus[0].power_draw_w == pytest.approx(350.5)
    assert node.gpus[0].power_limit_w == pytest.approx(1000.0)
    assert node.gpus[0].sm_clock_mhz == 1980
    assert node.gpus[0].gpu_util_pct == 75
    assert node.gpus[1].index == 1
    assert node.gpus[1].power_draw_w == pytest.approx(220.0)


def test_node_sample_from_prom_none_when_metric_absent():
    def _sparse(promql: str) -> list[dict]:
        if promql == "DCGM_FI_DEV_POWER_USAGE":
            return [
                {"metric": {"gpu": "0", "modelName": "NVIDIA-SIM-GPU", "hostname": "sim-host-0"}, "value": 300.0},
            ]
        return []

    client = MagicMock(spec=PromClient)
    client.query.side_effect = _sparse

    node = NodeSampleFromProm(client)

    assert len(node.gpus) == 1
    assert node.gpus[0].power_draw_w == pytest.approx(300.0)
    assert node.gpus[0].power_limit_w is None
    assert node.gpus[0].sm_clock_mhz is None
    assert node.gpus[0].gpu_util_pct is None
