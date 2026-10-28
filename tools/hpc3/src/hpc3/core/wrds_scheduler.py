"""Grid Engine admission, queue state and accounting, with no Slurm emulation."""

from __future__ import annotations

import pathlib
import shlex
import xml.etree.ElementTree as ET

from hpc3.contracts.wrds_run import WrdsRun
from hpc3.core import _test_hooks, remote


def job_argv(run: WrdsRun, directory: str, attempt: str, runner_sha: str) -> list[str]:
    """Build the bounded scheduler request for one immutable attempt.

    Args:
        run: Validated resources and runtime.
        directory: Attempt directory.
        attempt: Attempt label.
        runner_sha: Expected worker archive digest.

    Returns:
        qsub argv without admission/submission mode flags.
    """
    path = pathlib.PurePosixPath(directory)
    runner = str(path / "worker.pyz")
    check = shlex.join(["printf", "%s  %s\\n", runner_sha, runner]) + " | sha256sum -c -"
    execute = shlex.join(
        [run["runtime"]["python"], "-s", runner, "run", str(path / "run.json"), attempt]
    )
    return [
        "qsub",
        "-terse",
        "-b",
        "y",
        "-shell",
        "n",
        "-N",
        "wrds_" + runner_sha[:12] + "_" + attempt[:16],
        "-pe",
        "onenode",
        str(run["slots"]),
        "-l",
        f"h_rt={run['seconds']},m_mem_free={run['memory_gb_per_slot']}G",
        "-wd",
        directory,
        "-o",
        str(path / "scheduler.out"),
        "-e",
        str(path / "scheduler.err"),
        "-v",
        "PGHOST,PGPORT,PGDATABASE",
        "/bin/sh",
        "-c",
        check + " && exec " + execute,
    ]


def parse_job_id(text: str) -> str:
    """Decode a terse, non-array Grid Engine receipt.

    Args:
        text: Scheduler stdout.

    Returns:
        Numeric job id.

    Raises:
        ValueError: When the receipt is empty, ambiguous or not numeric.
    """
    value = text.strip()
    if not value.isdecimal():
        raise ValueError(f"WRDS_JOB_ID: invalid receipt {text!r}")
    return value


def queue_state(xml: str, job_id: str) -> str:
    """Read the exact job from qstat XML.

    Args:
        xml: qstat -xml output.
        job_id: Numeric id.

    Returns:
        Grid Engine's state or 'absent'; absence is never success.

    Raises:
        ValueError: For an unexpected document or duplicate/malformed job.
    """
    root = ET.fromstring(xml)
    if root.tag != "job_info":
        raise ValueError("WRDS_QUEUE: unexpected qstat XML root")
    matches = [node for node in root.iter("job_list") if node.findtext("JB_job_number") == job_id]
    if not matches:
        return "absent"
    if len(matches) != 1 or not matches[0].findtext("state"):
        raise ValueError("WRDS_QUEUE: ambiguous or missing job state")
    return str(matches[0].findtext("state"))


def account_state(text: str, job_id: str) -> str:
    """Require a matching completed qacct record.

    Args:
        text: qacct stdout.
        job_id: Expected job number.

    Returns:
        'completed' only for scheduler failure=0 and payload exit_status=0.

    Raises:
        ValueError: For missing, duplicate or mismatched accounting fields.
    """
    rows: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2 and fields[0] in {"jobnumber", "failed", "exit_status"}:
            if fields[0] in rows:
                raise ValueError("WRDS_ACCOUNTING: duplicate accounting record")
            rows[fields[0]] = fields[1].split()[0]
    if rows.get("jobnumber") != job_id or not {"failed", "exit_status"} <= rows.keys():
        raise ValueError("WRDS_ACCOUNTING: incomplete or mismatched record")
    return "completed" if rows["failed"] == "0" and rows["exit_status"] == "0" else "failed"


def status(host: str, job_id: str) -> str:
    """Consult the queue, then terminal accounting.

    Args:
        host: SSH alias.
        job_id: Numeric scheduler id.

    Returns:
        Active scheduler state, 'completed', 'failed' or 'accounting-pending'.

    Raises:
        ValueError: When accounting fails for any reason other than its
            documented not-found interval after a job leaves the queue.
    """
    parse_job_id(job_id)
    active = queue_state(remote.run_remote(host, "qstat -xml"), job_id)
    if active != "absent":
        return active
    answer = _test_hooks.run(remote.ssh_argv(host, shlex.join(["qacct", "-j", job_id])))
    if answer["returncode"] != 0:
        if "not found" in answer["stderr"] and job_id in answer["stderr"]:
            return "accounting-pending"
        raise ValueError(f"WRDS_ACCOUNTING: {answer['stderr']}")
    return account_state(answer["stdout"], job_id)


__all__ = ["account_state", "job_argv", "parse_job_id", "queue_state", "status"]
