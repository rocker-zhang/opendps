"""opendps.controller — standalone (zero-k8s) control loop."""

from .standalone import ControllerConfig, StandaloneController

__all__ = ["ControllerConfig", "StandaloneController"]
