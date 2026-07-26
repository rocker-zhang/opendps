import json
import sys
from types import SimpleNamespace

import pytest

from opendps.controller.priority_config import (
    ASSIGNMENTS_KEY,
    PriorityConfigSource,
)
from opendps.controller.standalone import StandaloneController
from opendps.controller.standalone import ControllerConfig
from opendps.pdn.presets import demo_single_domain
from opendps.sim.presets import oversub_scenario


class FakeConfigMapApi:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def read_namespaced_config_map(self, name, namespace, _request_timeout):
        self.calls.append((name, namespace, _request_timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeApiError(Exception):
    def __init__(self, status):
        super().__init__(f"API status {status}")
        self.status = status


def _config_map(assignments, resource_version="1"):
    return SimpleNamespace(
        metadata=SimpleNamespace(resource_version=resource_version),
        data={ASSIGNMENTS_KEY: json.dumps({"schemaVersion": 1, "assignments": assignments})},
    )


def _assignment(*, node="node-a", gpu=0, tier="high", generation=1):
    return {
        "policyUid": "policy-a",
        "policyGeneration": generation,
        "podUid": "pod-a",
        "nodeName": node,
        "gpuIndex": gpu,
        "priorityClass": tier,
        "gpuBoostPct": 20.0,
    }


def test_startup_load_filters_assignments_to_local_node():
    api = FakeConfigMapApi(
        _config_map(
            [
                _assignment(node="node-b", gpu=1, tier="critical"),
                _assignment(node="node-a", gpu=2, tier="high"),
            ]
        )
    )
    source = PriorityConfigSource(
        namespace="power-system",
        node_name="node-a",
        baseline={7: "low"},
        api=api,
    )

    assert source.snapshot() == {7: "low"}
    assert source.poll() == {2: "high", 7: "low"}
    assert source.snapshot() == {2: "high", 7: "low"}
    assert api.calls == [("opendps-job-boosts", "power-system", 5.0)]


def test_update_replaces_dynamic_snapshot_without_retaining_removed_gpus():
    api = FakeConfigMapApi(
        _config_map([_assignment(gpu=0, tier="normal")], "1"),
        _config_map([_assignment(gpu=3, tier="critical", generation=2)], "2"),
    )
    source = PriorityConfigSource(
        namespace="default", node_name="node-a", baseline={9: "low"}, api=api
    )

    assert source.poll() == {0: "normal", 9: "low"}
    assert source.poll() == {3: "critical", 9: "low"}


def test_incluster_initialization_failure_is_retried(monkeypatch):
    api = FakeConfigMapApi(_config_map([_assignment(gpu=2, tier="high")], "7"))

    class FakeConfig:
        calls = 0

        @classmethod
        def load_incluster_config(cls):
            cls.calls += 1
            if cls.calls == 1:
                raise RuntimeError("service account not ready")

    fake_kubernetes = SimpleNamespace(
        client=SimpleNamespace(CoreV1Api=lambda: api),
        config=FakeConfig,
    )
    monkeypatch.setitem(sys.modules, "kubernetes", fake_kubernetes)
    source = PriorityConfigSource(namespace="default", node_name="node-a", baseline={0: "low"})

    assert source.poll() == {0: "low"}
    assert source.poll() == {0: "low", 2: "high"}
    assert FakeConfig.calls == 2


def test_successful_new_resource_version_logs_stable_applied_marker(caplog):
    api = FakeConfigMapApi(
        _config_map(
            [
                _assignment(gpu=2, tier="normal"),
                _assignment(gpu=0, tier="high"),
            ],
            "42",
        ),
    )
    source = PriorityConfigSource(namespace="default", node_name="node-a", api=api)

    with caplog.at_level("INFO"):
        source.poll()

    assert (
        "Priority configuration applied: resourceVersion=42 "
        "node=node-a tiers=0=high,2=normal"
    ) in caplog.text


def test_missing_assignment_key_keeps_last_known_good_snapshot():
    empty = SimpleNamespace(
        metadata=SimpleNamespace(resource_version="2"),
        data={},
    )
    api = FakeConfigMapApi(
        _config_map([_assignment(gpu=0, tier="high")], "1"),
        empty,
    )
    source = PriorityConfigSource(
        namespace="default", node_name="node-a", baseline={0: "low"}, api=api
    )

    assert source.poll() == {0: "high"}
    assert source.poll() == {0: "high"}


def test_not_found_restores_cli_baseline():
    api = FakeConfigMapApi(
        _config_map([_assignment(gpu=0, tier="high")], "1"),
        FakeApiError(404),
    )
    source = PriorityConfigSource(
        namespace="default", node_name="node-a", baseline={0: "low"}, api=api
    )

    assert source.poll() == {0: "high"}
    assert source.poll() == {0: "low"}


def test_valid_empty_assignments_restore_cli_baseline():
    api = FakeConfigMapApi(
        _config_map([_assignment(gpu=0, tier="high")], "1"),
        _config_map([], "2"),
    )
    source = PriorityConfigSource(
        namespace="default", node_name="node-a", baseline={0: "low"}, api=api
    )

    assert source.poll() == {0: "high"}
    assert source.poll() == {0: "low"}


def test_malformed_update_keeps_last_known_good_snapshot():
    malformed = SimpleNamespace(
        metadata=SimpleNamespace(resource_version="2"),
        data={ASSIGNMENTS_KEY: "{not-json"},
    )
    api = FakeConfigMapApi(
        _config_map([_assignment(gpu=1, tier="high")], "1"),
        malformed,
    )
    source = PriorityConfigSource(
        namespace="default", node_name="node-a", baseline={1: "low"}, api=api
    )

    assert source.poll() == {1: "high"}
    assert source.poll() == {1: "high"}
    assert source.snapshot() == {1: "high"}


def test_persistent_failure_warns_once_and_success_resets_warning(caplog):
    malformed = SimpleNamespace(
        metadata=SimpleNamespace(resource_version="2"),
        data={ASSIGNMENTS_KEY: "{not-json"},
    )
    api = FakeConfigMapApi(
        malformed,
        malformed,
        _config_map([_assignment(gpu=1, tier="high")], "3"),
        malformed,
    )
    source = PriorityConfigSource(namespace="default", node_name="node-a", api=api)

    with caplog.at_level("WARNING"):
        source.poll()
        source.poll()
        source.poll()
        source.poll()

    assert sum("Ignoring malformed" in record.message for record in caplog.records) == 2


def test_forbidden_update_keeps_last_known_good_snapshot():
    api = FakeConfigMapApi(
        _config_map([_assignment(gpu=1, tier="high")], "1"),
        FakeApiError(403),
    )
    source = PriorityConfigSource(
        namespace="default", node_name="node-a", baseline={1: "low"}, api=api
    )

    assert source.poll() == {1: "high"}
    assert source.poll() == {1: "high"}


def test_poll_passes_configured_kubernetes_request_timeout():
    api = FakeConfigMapApi(_config_map([]))
    source = PriorityConfigSource(
        namespace="default",
        node_name="node-a",
        api=api,
        request_timeout_s=2.5,
    )

    source.poll()

    assert api.calls == [("opendps-job-boosts", "default", 2.5)]


def test_conflicting_assignments_choose_highest_tier_deterministically():
    assignments = [
        _assignment(gpu=4, tier="normal"),
        {
            **_assignment(gpu=4, tier="critical"),
            "policyUid": "policy-b",
            "podUid": "pod-b",
        },
        {
            **_assignment(gpu=4, tier="high"),
            "policyUid": "policy-c",
            "podUid": "pod-c",
        },
    ]
    source = PriorityConfigSource(
        namespace="default",
        node_name="node-a",
        api=FakeConfigMapApi(_config_map(assignments)),
    )

    assert source.poll() == {4: "critical"}


def test_standalone_run_applies_dynamic_tiers_before_next_tick(monkeypatch):
    class Source:
        def __init__(self):
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return {0: "critical", 1: "normal"}

    source = Source()
    controller = object.__new__(StandaloneController)
    controller._config = SimpleNamespace(
        priority_config_source=source,
        interval_s=0,
        dry_run=False,
        sim_mode=True,
    )

    class Brain:
        def __init__(self):
            self.tiers = {0: "low"}

        def set_tiers(self, tiers):
            self.tiers = dict(tiers)

    controller._brain = Brain()
    controller._managed_domains = []

    def verify_tick():
        assert controller._brain.tiers == {
            0: "critical",
            1: "normal",
        }

    controller.run_once = verify_tick
    monkeypatch.setattr(
        "opendps.controller.standalone.time.sleep",
        lambda _interval: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        controller.run()
    assert source.poll_count == 1


def test_standalone_run_reports_incompatible_dynamic_brain(monkeypatch, caplog):
    controller = object.__new__(StandaloneController)
    controller._config = SimpleNamespace(
        priority_config_source=SimpleNamespace(poll=lambda: {0: "high"}),
        interval_s=0,
        dry_run=False,
        sim_mode=True,
    )
    controller._brain = SimpleNamespace()
    controller._managed_domains = []
    monkeypatch.setattr(
        "opendps.controller.standalone.time.sleep",
        lambda _interval: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt), caplog.at_level("ERROR"):
        controller.run()

    assert "requires a brain with set_tiers()" in caplog.text


def test_priority_controller_allows_empty_cli_baseline_with_dynamic_source():
    source = SimpleNamespace(poll=lambda: {0: "high"}, snapshot=lambda: {})
    config = ControllerConfig(
        topology=demo_single_domain(n_gpus=1),
        actuator=oversub_scenario(n_gpus=1),
        gpu_priority_tiers={},
        priority_config_source=source,
        brain_type="priority-prs",
    )

    controller = StandaloneController(config)

    assert controller._brain._tiers == {}


def test_priority_controller_still_rejects_empty_cli_without_dynamic_source():
    config = ControllerConfig(
        topology=demo_single_domain(n_gpus=1),
        actuator=oversub_scenario(n_gpus=1),
        gpu_priority_tiers={},
        priority_config_source=None,
        brain_type="priority-prs",
    )

    with pytest.raises(ValueError, match="gpu-priority-tiers"):
        StandaloneController(config)
