"""Prometheus metrics exporter for the standalone controller.

Exposes opendps_* gauges that Grafana queries to show PRS state:
  - opendps_idle_stranded_watts{domain}    — idle budget not utilized
  - opendps_prs_active{domain}            — 1 if PRS is active, 0 if DPM
  - opendps_domain_power_draw_watts{domain}
  - opendps_domain_power_cap_watts{domain}
  - opendps_hot_gpu_count{domain}
  - opendps_idle_gpu_count{domain}

Usage:
    from opendps.telemetry.metrics import start_metrics_server, update_decision_metrics
    start_metrics_server(9402)
    # ... each tick:
    update_decision_metrics(domain, decision, state, prs_active, prs_metrics)
"""

from __future__ import annotations

import threading

from prometheus_client import Gauge, start_http_server

_STRANDED = Gauge(
    "opendps_idle_stranded_watts",
    "Idle GPU stranded power allocation (W): allocated but unused",
    ["domain"],
)
_PRS_ACTIVE = Gauge(
    "opendps_prs_active",
    "1 = PRS brain active (oversubscription reclaim on), 0 = DPM only",
    ["domain"],
)
_DOMAIN_DRAW = Gauge(
    "opendps_domain_power_draw_watts",
    "Total domain GPU power draw (W)",
    ["domain"],
)
_DOMAIN_CAP = Gauge(
    "opendps_domain_power_cap_watts",
    "Total domain allocated power cap (W)",
    ["domain"],
)
_HOT_COUNT = Gauge(
    "opendps_hot_gpu_count",
    "Number of hot GPUs in domain (draw/cap >= PRS threshold)",
    ["domain"],
)
_IDLE_COUNT = Gauge(
    "opendps_idle_gpu_count",
    "Number of idle GPUs in domain (candidates for reclaim)",
    ["domain"],
)
_GPU_DRAW = Gauge(
    "opendps_gpu_power_draw_watts",
    "Per-GPU power draw (W)",
    ["domain", "gpu"],
)
_GPU_CAP = Gauge(
    "opendps_gpu_power_cap_watts",
    "Per-GPU allocated power cap (W)",
    ["domain", "gpu"],
)
_FAILSAFE_TRIPS = Gauge(
    "opendps_failsafe_trip_total",
    "Cumulative number of failsafe emergency cap trips",
    ["domain"],
)


def start_metrics_server(port: int) -> None:
    """Start the Prometheus HTTP /metrics endpoint on the given port."""
    start_http_server(port)


def update_decision_metrics(
    domain: str,
    *,
    prs_active: bool,
    domain_draw_w: float,
    domain_cap_w: float,
    idle_stranded_w: float = 0.0,
    hot_count: int = 0,
    idle_count: int = 0,
    gpu_draws: dict[int, float] | None = None,
    gpu_caps: dict[int, float] | None = None,
    failsafe_trips: int | None = None,
) -> None:
    """Update all Prometheus gauges after a brain decision."""
    _PRS_ACTIVE.labels(domain=domain).set(1 if prs_active else 0)
    _DOMAIN_DRAW.labels(domain=domain).set(domain_draw_w)
    _DOMAIN_CAP.labels(domain=domain).set(domain_cap_w)
    _STRANDED.labels(domain=domain).set(idle_stranded_w)
    _HOT_COUNT.labels(domain=domain).set(hot_count)
    _IDLE_COUNT.labels(domain=domain).set(idle_count)
    if gpu_draws:
        for gpu, w in gpu_draws.items():
            _GPU_DRAW.labels(domain=domain, gpu=str(gpu)).set(w)
    if gpu_caps:
        for gpu, w in gpu_caps.items():
            _GPU_CAP.labels(domain=domain, gpu=str(gpu)).set(w)
    if failsafe_trips is not None:
        _FAILSAFE_TRIPS.labels(domain=domain).set(failsafe_trips)


def start_healthz_server(port: int) -> None:
    """Minimal /healthz endpoint on a separate port."""
    import http.server

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args): pass

    srv = http.server.HTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True, name="healthz").start()
