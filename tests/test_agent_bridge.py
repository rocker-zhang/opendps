from unittest.mock import patch, MagicMock
import socket
from opendps.controller.agent_bridge import AgentBridge


def test_push_caps_returns_false_when_unreachable():
    """Should return False and not raise when agent is down."""
    bridge = AgentBridge("127.0.0.1", 19999)
    result = bridge.push_caps({0: 850.0, 1: 900.0})
    assert result is False
    assert not bridge.is_connected


def test_push_caps_sends_json_messages():
    """Mock socket to verify JSON messages are sent for each GPU."""
    mock_sock = MagicMock()
    mock_sock.__enter__ = lambda s: s
    mock_sock.__exit__ = MagicMock(return_value=False)

    with patch("socket.create_connection", return_value=mock_sock):
        bridge = AgentBridge()
        result = bridge.push_caps({0: 850.0})

    assert result is True
    sent = b"".join(call[0][0] for call in mock_sock.sendall.call_args_list)
    import json
    msg = json.loads(sent.strip())
    assert msg["cmd"] == "set_cap"
    assert msg["gpu"] == 0
    assert msg["watts"] == 850.0
