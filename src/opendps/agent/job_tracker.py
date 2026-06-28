"""Tracks running compute processes per GPU via nvidia-smi."""
from __future__ import annotations
import subprocess, threading, time
from dataclasses import dataclass, field


@dataclass
class GPUJob:
    pid: int
    gpu_uuid: str
    gpu_index: int
    used_memory_mib: int


class JobTracker:
    """Polls nvidia-smi to discover active compute processes on each GPU."""

    def __init__(self, poll_interval_s: float = 5.0):
        self._interval = poll_interval_s
        self._jobs: dict[int, list[GPUJob]] = {}
        self._uuid_to_index: dict[str, int] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="job-tracker"
        )

    def start(self) -> None:
        self._refresh_uuid_map()
        self._thread.start()

    def get_jobs(self, gpu_index: int) -> list[GPUJob]:
        with self._lock:
            return list(self._jobs.get(gpu_index, []))

    def is_gpu_busy(self, gpu_index: int) -> bool:
        return bool(self.get_jobs(gpu_index))

    # ------------------------------------------------------------------ private
    def _refresh_uuid_map(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            for line in out.stdout.strip().splitlines():
                idx_s, uuid = [p.strip() for p in line.split(",", 1)]
                self._uuid_to_index[uuid] = int(idx_s)
        except Exception:
            pass

    def _poll(self) -> dict[int, list[GPUJob]]:
        result: dict[int, list[GPUJob]] = {}
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            for line in out.stdout.strip().splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                pid, uuid, mem = int(parts[0]), parts[1], int(parts[2])
                gpu_idx = self._uuid_to_index.get(uuid, -1)
                if gpu_idx < 0:
                    continue
                result.setdefault(gpu_idx, []).append(
                    GPUJob(pid=pid, gpu_uuid=uuid, gpu_index=gpu_idx, used_memory_mib=mem)
                )
        except Exception:
            pass
        return result

    def _loop(self) -> None:
        while True:
            jobs = self._poll()
            with self._lock:
                self._jobs = jobs
            time.sleep(self._interval)
