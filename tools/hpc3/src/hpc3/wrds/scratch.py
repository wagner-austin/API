"""Observe named job scratch on its original node without reading plaintext."""

from __future__ import annotations

import pathlib
import re
import socket

from platform_core.json_utils import JSONObject

from hpc3.wrds import _test_hooks


def observe(
    expected_host: str, job_id: str, private: pathlib.Path, scratch_root: pathlib.Path
) -> JSONObject:
    """List directory names to distinguish absence from inaccessible storage.

    Args:
        expected_host: Exact hostname from terminal scheduler accounting.
        job_id: Terminated job whose scratch is being observed.
        private: Exact authenticated plaintext directory from its opened receipt.
        scratch_root: Scheduler scratch root, explicitly /tmp on WRDS.

    Returns:
        Metadata only, including the observation time and both path presences.

    Raises:
        ValueError: Wrong host, invalid paths or an observation of the current job.
        OSError: Directory listing fails; access failures never mean absence.
    """
    node = socket.gethostname()
    observer_job = _test_hooks.environment()["JOB_ID"]
    if node != expected_host:
        raise ValueError("WRDS_SCRATCH_HOST: observer is not on the original execution node")
    if not job_id.isdecimal() or not observer_job.isdecimal() or job_id == observer_job:
        raise ValueError("WRDS_SCRATCH_JOB: distinct numeric job identifiers are required")
    scratch = private.parent
    if (
        scratch.parent != scratch_root
        or re.fullmatch(re.escape(job_id) + r"\.1\.[a-zA-Z0-9_.-]+", scratch.name) is None
        or re.fullmatch(r"research-plaintext-[a-zA-Z0-9_-]+", private.name) is None
    ):
        raise ValueError("WRDS_SCRATCH_PATH: private path does not belong to the named job")
    scratch_present = scratch.name in {path.name for path in scratch_root.iterdir()}
    private_present = scratch_present and private.name in {path.name for path in scratch.iterdir()}
    return {
        "node": node,
        "job_id": job_id,
        "observer_job_id": observer_job,
        "observed_at": _test_hooks.now_iso(),
        "scratch": str(scratch),
        "private": str(private),
        "scratch_present": scratch_present,
        "private_present": private_present,
    }


__all__ = ["observe"]
