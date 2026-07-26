import pytest

from opendps.controller.priority_config import dynamic_priority_config_from_env


def test_dynamic_priority_config_env_is_explicit_and_strict():
    assert dynamic_priority_config_from_env({}) == (False, None)
    assert dynamic_priority_config_from_env({"UNRELATED_NODE_NAME": "node-a"}) == (False, None)
    assert dynamic_priority_config_from_env(
        {
            "OPENDPS_PRIORITY_CONFIG_ENABLED": "TrUe",
            "OPENDPS_NODE_NAME": " node-a ",
        }
    ) == (True, "node-a")

    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        dynamic_priority_config_from_env({"OPENDPS_PRIORITY_CONFIG_ENABLED": "1"})
    with pytest.raises(ValueError, match="OPENDPS_NODE_NAME is required"):
        dynamic_priority_config_from_env({"OPENDPS_PRIORITY_CONFIG_ENABLED": "true"})
