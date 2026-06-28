"""Redfish chassis power scraper for BMC-level power telemetry."""
from __future__ import annotations
import base64
import dataclasses
import json
import logging
import ssl
import threading
import time
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ChassisPowerReading:
    ts: float
    chassis_total_w: float
    gpu_aggregate_w: float
    nvswitch_w: float
    cpu_package_w: float
    psu_input_watts: list[float]
    psu_efficiency: list[float]


class RedfishScraper:
    """
    Polls Redfish /Chassis/{id}/Power for node-level power breakdown.

    On DGX/HGX systems: bmc_url = "https://169.254.0.17" (BMC link-local).
    Requires the host to have a management NIC configured at 169.254.0.18.
    """

    def __init__(
        self,
        bmc_url: str,
        username: str = "admin",
        password: str = "",  # must be set by caller; never hard-code BMC credentials
        chassis_id: str = "1",
        poll_interval_s: float = 10.0,
        verify_ssl: bool = False,
    ):
        self._url = bmc_url.rstrip("/")
        self._auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        self._chassis_id = chassis_id
        self._interval = poll_interval_s
        self._verify_ssl = verify_ssl
        self._latest: Optional[ChassisPowerReading] = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="redfish-scraper"
        )

    def start(self) -> None:
        self._thread.start()

    def get_latest(self) -> Optional[ChassisPowerReading]:
        with self._lock:
            return self._latest

    def _fetch(self) -> ChassisPowerReading:
        url = f"{self._url}/redfish/v1/Chassis/{self._chassis_id}/Power"
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {self._auth}"})
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE if not self._verify_ssl else ssl.CERT_REQUIRED
        with urllib.request.urlopen(req, context=ctx, timeout=5) as r:
            data = json.loads(r.read())

        ctrl = data.get("PowerControl", [{}])[0]
        chassis_w = float(ctrl.get("PowerConsumedWatts", 0.0))
        oem = ctrl.get("Oem", {}).get("Nvidia", {})

        psus = data.get("PowerSupplies", [])
        psu_in  = [float(p.get("PowerInputWatts", 0.0)) for p in psus]
        psu_out = [float(p.get("PowerOutputWatts", 0.0)) for p in psus]
        eff = [o / i if i > 0 else 0.0 for o, i in zip(psu_out, psu_in)]

        return ChassisPowerReading(
            ts=time.time(),
            chassis_total_w=chassis_w,
            gpu_aggregate_w=float(oem.get("GpuPowerWatts", 0.0)),
            nvswitch_w=float(oem.get("NVSwitchPowerWatts", 0.0)),
            cpu_package_w=float(oem.get("CpuPackagePower", 0.0)),
            psu_input_watts=psu_in,
            psu_efficiency=eff,
        )

    def _loop(self) -> None:
        while True:
            try:
                reading = self._fetch()
                with self._lock:
                    self._latest = reading
            except Exception as e:
                logger.debug("Redfish poll failed: %s", e)
            time.sleep(self._interval)
