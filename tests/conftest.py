"""Optional dependency stubs used by the default development test suite.

The Kubernetes operator dependencies live behind the ``operator`` extra.  The
default ``dev`` extra must still be able to collect and exercise the operator's
unit tests, so provide small semantic stubs when those packages are absent.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock


def _identity_handler(*_args, **_kwargs):
    def decorate(function):
        return function

    return decorate


if "kopf" not in sys.modules:
    try:
        import kopf  # noqa: F401
    except ImportError:
        kopf_stub = ModuleType("kopf")

        class PermanentError(Exception):
            pass

        class TemporaryError(Exception):
            def __init__(self, message, *, delay=None):
                super().__init__(message)
                self.delay = delay

        kopf_stub.PermanentError = PermanentError
        kopf_stub.TemporaryError = TemporaryError
        kopf_stub.on = SimpleNamespace(
            create=_identity_handler,
            update=_identity_handler,
            delete=_identity_handler,
            resume=_identity_handler,
            event=_identity_handler,
            timer=_identity_handler,
        )
        sys.modules["kopf"] = kopf_stub


if "kubernetes" not in sys.modules:
    try:
        import kubernetes  # noqa: F401
    except ImportError:
        kubernetes_stub = ModuleType("kubernetes")
        client_stub = ModuleType("kubernetes.client")
        exceptions_stub = ModuleType("kubernetes.client.exceptions")

        class ApiException(Exception):
            def __init__(self, status=None):
                super().__init__(f"Kubernetes API status {status}")
                self.status = status

        exceptions_stub.ApiException = ApiException
        client_stub.exceptions = exceptions_stub
        client_stub.CoreV1Api = MagicMock()
        client_stub.CustomObjectsApi = MagicMock()
        client_stub.V1ConfigMap = MagicMock()
        client_stub.V1ObjectMeta = MagicMock()
        kubernetes_stub.client = client_stub
        sys.modules["kubernetes"] = kubernetes_stub
        sys.modules["kubernetes.client"] = client_stub
        sys.modules["kubernetes.client.exceptions"] = exceptions_stub
