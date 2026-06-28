from unittest.mock import patch
from opendps.agent.job_tracker import JobTracker

_NVIDIA_SMI_APPS = "1234, GPU-aaaa-bbbb, 512\n5678, GPU-cccc-dddd, 256\n"
_NVIDIA_SMI_GPUS = "0, GPU-aaaa-bbbb\n1, GPU-cccc-dddd\n"

def _make_tracker():
    t = JobTracker(poll_interval_s=60)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = _NVIDIA_SMI_GPUS
        t._refresh_uuid_map()
    return t

def test_poll_parses_jobs():
    t = _make_tracker()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = _NVIDIA_SMI_APPS
        jobs = t._poll()
    assert 0 in jobs and 1 in jobs
    assert jobs[0][0].pid == 1234
    assert jobs[1][0].pid == 5678

def test_is_gpu_busy_true_when_jobs():
    t = _make_tracker()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = _NVIDIA_SMI_APPS
        with t._lock:
            t._jobs = t._poll()
    assert t.is_gpu_busy(0) is True

def test_is_gpu_busy_false_when_no_jobs():
    t = _make_tracker()
    assert t.is_gpu_busy(0) is False
