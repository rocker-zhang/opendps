"""Demo scenario factories for the sim backend.

Each factory returns a fully initialised SimBackend ready for use by the
control brain or the CLI demo.  All scenarios use per-GPU seeded RNGs derived
from the master seed so the fleet is reproducible while each GPU evolves
independently.
"""

from __future__ import annotations

import random

from .backend import LoadProfile, SimBackend, SimGpu


def oversub_scenario(
    n_gpus: int = 10,
    cap_w: float = 1000.0,
    hot_fraction: float = 0.6,
    seed: int = 42,
) -> SimBackend:
    """The headline demo: n_gpus GPUs, mixed hot/idle load.

    hot_fraction of GPUs are "hot" (~90 % utilisation); the remainder are
    "idle" (~10 % utilisation).  All GPUs start capped at cap_w, meaning the
    fleet's raw demand exceeds any reasonable PDN budget — exactly the scenario
    the controller must manage.

    Parameters
    ----------
    n_gpus:       total number of simulated GPUs (default 10)
    cap_w:        initial per-GPU power cap in watts (default 1000 W)
    hot_fraction: fraction of GPUs that are hot (default 0.6 → 6 out of 10)
    seed:         master RNG seed for reproducibility (default 42)
    """
    n_hot = round(n_gpus * hot_fraction)
    gpus: list[SimGpu] = []
    for i in range(n_gpus):
        rng = random.Random(seed + i)
        if i < n_hot:
            load = LoadProfile(base_util_pct=90.0, util_noise=5.0)
        else:
            load = LoadProfile(base_util_pct=10.0, util_noise=3.0)
        gpus.append(SimGpu(index=i, cap_w=cap_w, max_cap_w=cap_w, load=load, _rng=rng))
    return SimBackend(gpus)


def uniform_load(
    n_gpus: int = 8,
    cap_w: float = 1000.0,
    util_pct: float = 70.0,
    seed: int = 42,
) -> SimBackend:
    """Uniform load, baseline scenario.

    All GPUs run at the same utilisation level.  Useful as a regression
    baseline and for verifying that the controller respects budgets when demand
    is well-spread.

    Parameters
    ----------
    n_gpus:   number of GPUs (default 8)
    cap_w:    per-GPU power cap in watts (default 1000 W)
    util_pct: target utilisation percentage for all GPUs (default 70.0)
    seed:     master RNG seed for reproducibility (default 42)
    """
    gpus: list[SimGpu] = []
    for i in range(n_gpus):
        rng = random.Random(seed + i)
        load = LoadProfile(base_util_pct=util_pct, util_noise=5.0)
        gpus.append(SimGpu(index=i, cap_w=cap_w, max_cap_w=cap_w, load=load, _rng=rng))
    return SimBackend(gpus)
