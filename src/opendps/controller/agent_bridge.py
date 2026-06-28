"""
Controller → opendps-agent IPC bridge.

Protocol: newline-delimited JSON over Unix domain socket or TCP.
The Rust agent listens on a socket and accepts cap commands:

  {"cmd": "set_cap", "gpu": 0, "watts": 850.0}
  {"cmd": "get_caps"}  → {"caps": {0: 1100.0, 1: 900.0, ...}}
  {"cmd": "get_draws"} → {"draws": {0: 237.4, ...}}

This module is a CLIENT — it connects to a running opendps-agent process.
"""
from __future__ import annotations
import json
import logging
import socket
import threading
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_AGENT_HOST = "127.0.0.1"
DEFAULT_AGENT_PORT = 9500  # future: Rust agent listens here for IPC


class AgentBridge:
    """
    Sends cap decisions from the Python brain to the Rust opendps-agent.

    Falls back to no-op if agent is unreachable (dry-run mode).
    """

    def __init__(self, host: str = DEFAULT_AGENT_HOST, port: int = DEFAULT_AGENT_PORT,
                 timeout_s: float = 1.0):
        self._host = host
        self._port = port
        self._timeout = timeout_s
        self._lock = threading.Lock()
        self._connected = False

    def push_caps(self, caps: dict[int, float]) -> bool:
        """Push cap decisions to the agent. Returns True if sent successfully."""
        try:
            with self._lock:
                sock = socket.create_connection((self._host, self._port),
                                                timeout=self._timeout)
                with sock:
                    for gpu, watts in caps.items():
                        msg = json.dumps({"cmd": "set_cap", "gpu": gpu, "watts": watts})
                        sock.sendall((msg + "\n").encode())
                    sock.shutdown(socket.SHUT_WR)
            self._connected = True
            return True
        except (OSError, ConnectionRefusedError) as e:
            if self._connected:
                logger.warning("Agent bridge disconnected: %s", e)
            self._connected = False
            return False

    def get_draws(self) -> Optional[dict[int, float]]:
        """Query current GPU power draws from agent."""
        try:
            with self._lock:
                sock = socket.create_connection((self._host, self._port),
                                                timeout=self._timeout)
                with sock:
                    sock.sendall(b'{"cmd": "get_draws"}\n')
                    sock.shutdown(socket.SHUT_WR)
                    data = b""
                    while chunk := sock.recv(4096):
                        data += chunk
            resp = json.loads(data.decode())
            return {int(k): float(v) for k, v in resp.get("draws", {}).items()}
        except Exception:
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected


class AgentBridgeActuator:
    """
    Actuator that delegates set_power_cap to the Rust opendps-agent via TCP.
    Telemetry reads (get_power_draw, get_power_cap) return 0.0 — in --prom mode
    the controller reads telemetry from Prometheus, not the actuator.
    """

    def __init__(self, host: str = DEFAULT_AGENT_HOST, port: int = DEFAULT_AGENT_PORT):
        self._bridge = AgentBridge(host=host, port=port)
        self._last_caps: dict[int, float] = {}

    def set_power_cap(self, gpu_index: int, watts: float) -> None:
        self._last_caps[gpu_index] = watts
        # Batch send happens via push_all_caps() called at end of tick
        # For now, send immediately (one connection per cap, acceptable for 5s interval)
        self._bridge.push_caps({gpu_index: watts})

    def push_all_caps(self, caps: dict[int, float]) -> bool:
        """Efficient batch send — call once per tick instead of set_power_cap per GPU."""
        self._last_caps = caps
        return self._bridge.push_caps(caps)

    def get_power_cap(self, gpu_index: int) -> float:
        return self._last_caps.get(gpu_index, 0.0)

    def get_power_draw(self, gpu_index: int) -> float:
        draws = self._bridge.get_draws()
        if draws:
            return draws.get(gpu_index, 0.0)
        return 0.0

    def get_util_pct(self, gpu_index: int) -> float:
        return 0.0  # not available via bridge; use Prometheus

    def gpu_count(self) -> int:
        return 0  # not needed in --prom mode (topology drives this)

    @property
    def is_connected(self) -> bool:
        return self._bridge.is_connected
