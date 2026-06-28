"""FailsafeLoop — brain-independent, cap-lower-only GPU protection.

Runs in a dedicated daemon thread.  Each poll cycle it reads the power draw
of every GPU and, if any exceeds emergency_threshold_w, immediately lowers
that GPU's cap to emergency_cap_w without consulting the brain.

Design properties
-----------------
- Cap-lower-only: the failsafe only ever *lowers* caps.  Raising caps back to
  normal is exclusively the brain's job (on the next regular control tick).
- Brain-independent: no shared lock with the brain thread.  The brain may be
  solving a CVXPY problem for 50 ms; the failsafe must not block on that.
- Configurable: threshold, emergency cap, and poll interval are all runtime
  parameters so the same code works across GPU families (A10, B300, GB200).

Phase 2 note
------------
This Python loop is replaced by a Rust SCHED_FIFO thread in P2-M2.  The
Python version has ~20–50 ms response latency because of GIL scheduling jitter.
The Rust version targets <1 ms P99 and is benchmarked by bench_failsafe.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


class FailsafeLoop:
    """Background thread that enforces a hard power draw ceiling."""

    def __init__(
        self,
        actuator,
        emergency_threshold_w: float,
        emergency_cap_w: float,
        poll_interval_s: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        actuator:
            Any object satisfying the Actuator Protocol.
        emergency_threshold_w:
            If any GPU's instantaneous draw exceeds this, trip immediately.
        emergency_cap_w:
            The cap to push on a tripped GPU.  Must be < emergency_threshold_w
            to prevent oscillation; typically 70–80% of TDP.
        poll_interval_s:
            Seconds between full-fleet scans.  Default 100 ms ≈ 10 Hz.
        """
        if emergency_cap_w >= emergency_threshold_w:
            raise ValueError(
                f"emergency_cap_w ({emergency_cap_w}) must be less than "
                f"emergency_threshold_w ({emergency_threshold_w}) to prevent oscillation"
            )
        self._actuator = actuator
        self._threshold = emergency_threshold_w
        self._emergency_cap = emergency_cap_w
        self._poll_interval = poll_interval_s
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="opendps-failsafe"
        )
        self._trip_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to exit (max 2 s)."""
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trip_count(self) -> int:
        """Number of failsafe trips since start() was called."""
        return self._trip_count

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        n = self._actuator.gpu_count()
        while not self._stop_event.is_set():
            for i in range(n):
                try:
                    draw = self._actuator.get_power_draw(i)
                    if draw > self._threshold:
                        self._actuator.set_power_cap(i, self._emergency_cap)
                        self._trip_count += 1
                        log.warning(
                            "FAILSAFE TRIP gpu=%d draw=%.1fW > threshold=%.1fW"
                            " → cap set to %.1fW",
                            i,
                            draw,
                            self._threshold,
                            self._emergency_cap,
                        )
                except Exception as exc:  # noqa: BLE001
                    log.error("failsafe poll error gpu=%d: %s", i, exc)
            self._stop_event.wait(self._poll_interval)
