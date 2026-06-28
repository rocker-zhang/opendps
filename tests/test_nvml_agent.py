"""Unit tests for NvmlActuator — pynvml fully mocked, no GPU hardware required."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers: build a fully mocked pynvml module
# ---------------------------------------------------------------------------

def _make_pynvml_mock(n_gpus: int = 2):
    mock = MagicMock()
    mock.nvmlDeviceGetCount.return_value = n_gpus
    mock.nvmlDeviceGetHandleByIndex.side_effect = lambda i: MagicMock(name=f"handle-{i}")
    mock.NVMLError = Exception
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_calls_nvml_init():
    pynvml = _make_pynvml_mock()
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        from importlib import reload
        import opendps.agent.nvml_agent as mod
        reload(mod)
        mod.NvmlActuator()
    pynvml.nvmlInit.assert_called_once()


def test_gpu_count_returns_device_count():
    pynvml = _make_pynvml_mock(n_gpus=4)
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        import opendps.agent.nvml_agent as mod
        from importlib import reload
        reload(mod)
        actuator = mod.NvmlActuator()
    assert actuator.gpu_count() == 4


def test_set_power_cap_converts_to_milliwatts():
    pynvml = _make_pynvml_mock()
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        import opendps.agent.nvml_agent as mod
        from importlib import reload
        reload(mod)
        actuator = mod.NvmlActuator()
        actuator.set_power_cap(0, 850.5)
    pynvml.nvmlDeviceSetPowerManagementLimit.assert_called_once_with(
        actuator._handles[0], 850500
    )


def test_get_power_cap_converts_from_milliwatts():
    pynvml = _make_pynvml_mock()
    pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 1_000_000  # 1000 W in mW
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        import opendps.agent.nvml_agent as mod
        from importlib import reload
        reload(mod)
        actuator = mod.NvmlActuator()
        result = actuator.get_power_cap(0)
    assert result == 1000.0


def test_get_power_draw_converts_from_milliwatts():
    pynvml = _make_pynvml_mock()
    pynvml.nvmlDeviceGetPowerUsage.return_value = 850_000  # 850 W in mW
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        import opendps.agent.nvml_agent as mod
        from importlib import reload
        reload(mod)
        actuator = mod.NvmlActuator()
        result = actuator.get_power_draw(0)
    assert result == 850.0


def test_get_util_pct():
    pynvml = _make_pynvml_mock()
    util_mock = MagicMock()
    util_mock.gpu = 75
    pynvml.nvmlDeviceGetUtilizationRates.return_value = util_mock
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        import opendps.agent.nvml_agent as mod
        from importlib import reload
        reload(mod)
        actuator = mod.NvmlActuator()
        result = actuator.get_util_pct(0)
    assert result == 75.0


def test_get_max_cap_w():
    pynvml = _make_pynvml_mock()
    pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (200_000, 1_200_000)
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        import opendps.agent.nvml_agent as mod
        from importlib import reload
        reload(mod)
        actuator = mod.NvmlActuator()
        result = actuator.get_max_cap_w(0)
    assert result == 1200.0


def test_get_name_bytes():
    pynvml = _make_pynvml_mock()
    pynvml.nvmlDeviceGetName.return_value = b"NVIDIA H100"
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        import opendps.agent.nvml_agent as mod
        from importlib import reload
        reload(mod)
        actuator = mod.NvmlActuator()
        result = actuator.get_name(0)
    assert result == "NVIDIA H100"


def test_get_name_string():
    pynvml = _make_pynvml_mock()
    pynvml.nvmlDeviceGetName.return_value = "NVIDIA B300"
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        import opendps.agent.nvml_agent as mod
        from importlib import reload
        reload(mod)
        actuator = mod.NvmlActuator()
        result = actuator.get_name(0)
    assert result == "NVIDIA B300"


def test_nvml_error_in_set_power_cap_does_not_raise():
    """NVMLError during set_power_cap must be caught and logged, not re-raised."""
    pynvml = _make_pynvml_mock()
    pynvml.NVMLError = RuntimeError
    pynvml.nvmlDeviceSetPowerManagementLimit.side_effect = RuntimeError("permission denied")
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        import opendps.agent.nvml_agent as mod
        from importlib import reload
        reload(mod)
        actuator = mod.NvmlActuator()
        # Must not raise
        actuator.set_power_cap(0, 800.0)


def test_nvml_error_in_get_power_draw_returns_zero():
    pynvml = _make_pynvml_mock()
    pynvml.NVMLError = RuntimeError
    pynvml.nvmlDeviceGetPowerUsage.side_effect = RuntimeError("not supported")
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        import opendps.agent.nvml_agent as mod
        from importlib import reload
        reload(mod)
        actuator = mod.NvmlActuator()
        result = actuator.get_power_draw(0)
    assert result == 0.0


def test_shutdown_calls_nvml_shutdown():
    pynvml = _make_pynvml_mock()
    with patch.dict("sys.modules", {"pynvml": pynvml}):
        import opendps.agent.nvml_agent as mod
        from importlib import reload
        reload(mod)
        actuator = mod.NvmlActuator()
        actuator.shutdown()
    pynvml.nvmlShutdown.assert_called_once()
