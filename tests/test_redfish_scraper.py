import json
from unittest.mock import patch, MagicMock
from opendps.telemetry.redfish_scraper import RedfishScraper

_REDFISH_RESPONSE = {
    "PowerControl": [{
        "PowerConsumedWatts": 5200.0,
        "Oem": {"Nvidia": {"GpuPowerWatts": 4800.0, "NVSwitchPowerWatts": 250.0, "CpuPackagePower": 150.0}}
    }],
    "PowerSupplies": [
        {"PowerInputWatts": 3000.0, "PowerOutputWatts": 2700.0},
        {"PowerInputWatts": 3000.0, "PowerOutputWatts": 2700.0},
    ]
}

def _mock_urlopen(url, **kwargs):
    m = MagicMock()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    m.read.return_value = json.dumps(_REDFISH_RESPONSE).encode()
    return m

def test_parses_chassis_total():
    s = RedfishScraper("https://169.254.0.17")
    with patch("urllib.request.urlopen", _mock_urlopen):
        r = s._fetch()
    assert r.chassis_total_w == 5200.0

def test_parses_nvswitch_power():
    s = RedfishScraper("https://169.254.0.17")
    with patch("urllib.request.urlopen", _mock_urlopen):
        r = s._fetch()
    assert r.nvswitch_w == 250.0

def test_get_latest_none_before_start():
    s = RedfishScraper("https://169.254.0.17")
    assert s.get_latest() is None
