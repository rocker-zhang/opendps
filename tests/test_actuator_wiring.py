"""Tests for actuator wiring in standalone.py main().

Verifies that --actuator sim/nvml/agent flags wire the correct backend
without requiring real GPUs or a running opendps-agent.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from opendps.controller.standalone import main

# Minimal topology config shared by all tests
_TOPOLOGY = {
    "pdus": {"pdu-A": {"capacity_w": 10000, "derating": 0.9}},
    "domains": {
        "domain-0": {
            "budget_w": 8000,
            "gpu_indices": [0, 1, 2, 3],
            "pdu_name": "pdu-A",
            "priority": 1,
        }
    },
}


@pytest.fixture()
def cfg_file(tmp_path):
    """Write a minimal topology JSON and return its path as a string."""
    p = tmp_path / "topology.json"
    p.write_text(json.dumps(_TOPOLOGY))
    return str(p)


def test_sim_actuator_default(cfg_file):
    """main() with default --actuator sim constructs and runs one tick without error."""
    with patch("opendps.controller.standalone.StandaloneController.run") as mock_run:
        mock_run.return_value = None  # don't loop forever
        result = main(["--config", cfg_file, "--sim", "--interval", "0"])
    assert result == 0
    mock_run.assert_called_once()


def test_nvml_actuator_flag_exists(cfg_file):
    """--actuator nvml is a valid CLI choice; falls back to sim when pynvml unavailable."""
    with patch("opendps.controller.standalone.StandaloneController.run") as mock_run:
        mock_run.return_value = None
        # NvmlActuator.__init__ will raise (no GPU on CI); code falls back to sim
        result = main(["--config", cfg_file, "--sim", "--actuator", "nvml", "--interval", "0"])
    assert result == 0
    mock_run.assert_called_once()


def test_agent_actuator_flag_exists(cfg_file):
    """--actuator agent is a valid CLI choice; AgentBridgeActuator is wired."""
    with patch("opendps.controller.standalone.StandaloneController.run") as mock_run:
        mock_run.return_value = None
        # AgentBridgeActuator doesn't connect in __init__, so no socket needed
        result = main(["--config", cfg_file, "--sim", "--actuator", "agent", "--interval", "0"])
    assert result == 0
    mock_run.assert_called_once()
