"""Entry point: python -m opendps.operator"""
import logging

import kopf

# Importing handlers registers the @kopf.on.* callbacks. Without this import
# kopf.run() starts with zero handlers and the operator reconciles nothing.
from opendps.operator import handlers  # noqa: F401

logging.basicConfig(level=logging.INFO)


@kopf.on.startup()
def _configure(settings: kopf.OperatorSettings, **_):
    # Run without peering so the operator does not require KopfPeering CRDs or
    # coordination.k8s.io Leases. A single replica owns reconciliation.
    settings.peering.standalone = True


def main() -> None:
    # clusterwide=True: watch CRs across all namespaces (explicit, silences the
    # kopf deprecation warning about defaulting to cluster-wide mode).
    kopf.run(clusterwide=True)


if __name__ == "__main__":
    main()
