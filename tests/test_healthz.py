import socket
import time
import urllib.request

from opendps.telemetry.metrics import start_healthz_server


def _free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_healthz_returns_200():
    port = _free_port()
    start_healthz_server(port)
    time.sleep(0.1)
    with urllib.request.urlopen(f"http://localhost:{port}/", timeout=2) as r:
        assert r.status == 200
        assert r.read() == b"ok"
