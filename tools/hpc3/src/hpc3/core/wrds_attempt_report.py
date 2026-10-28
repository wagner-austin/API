"""Measure one WRDS attempt's outcome on WRDS, for the program's board report.

A session that copies job numbers and digests into a note restates them; the
reading an audit accepts is one the measuring program writes. This module
renders that reading for one attempt: the experiment's code, input and runtime
digests, the scheduler's own accounting of where and how the job ended, and
either the attested outputs and the run-record digest collect would write, or
the refusal collect gives and whether the worker ever wrote completion.json.
Nothing here reads payload logs or dataset rows.
"""

from __future__ import annotations

import hashlib
import shlex

from platform_core.json_utils import dump_json_str, narrow_json_to_str, require_dict

from hpc3.contracts.wrds_run import WrdsRun, encode_wrds_runtime, run_identity
from hpc3.core import remote, wrds_jobs, wrds_results, wrds_scheduler

_PRESENCE = {"present", "absent"}


def completion_presence(host: str, directory: str) -> str:
    """Report whether the worker wrote its completion attestation.

    Args:
        host: SSH alias.
        directory: Remote attempt directory.

    Returns:
        ``present`` or ``absent``.

    Raises:
        AppError: When the remote test cannot run.
        ValueError: ``WRDS_PRESENCE`` for any other reply.
    """
    path = shlex.quote(directory + "/completion.json")
    answer = remote.run_remote(host, f"test -e {path} && echo present || echo absent").strip()
    if answer not in _PRESENCE:
        raise ValueError(f"WRDS_PRESENCE: unexpected reply {answer!r}")
    return answer


def _identity_lines(run: WrdsRun, attempt: str, identifier: str) -> list[str]:
    """Name the attempt and the code, input and runtime it was submitted with.

    Args:
        run: Experiment definition.
        attempt: Attempt label.
        identifier: Scheduler job id.

    Returns:
        Report lines.
    """
    runtime = run["runtime"]
    pin = hashlib.sha256(dump_json_str(encode_wrds_runtime(runtime)).encode()).hexdigest()
    return [
        f"WRDS ATTEMPT LOAD-REPORT {run['project']}/{run['name']} attempt {attempt}: "
        f"job {identifier} on {run['host']}",
        f"identity {run_identity(run)}",
        *(f"code {item['name']} sha256={item['sha256']}" for item in run["code"]),
        *(f"input {item['name']} sha256={item['sha256']}" for item in run["inputs"]),
        f"runtime python={runtime['python']} sha256={runtime['sha256']} "
        f"version={runtime['version']} packages={len(runtime['packages'])} pin_sha256={pin}",
    ]


def render(run: WrdsRun, attempt: str) -> str:
    """Measure one attempt on WRDS and render its report.

    Args:
        run: Experiment definition.
        attempt: Attempt label.

    Returns:
        One report, a line per measured fact.

    Raises:
        AppError: When WRDS cannot be reached or a remote read fails.
        ValueError: When the remote experiment drifted from the definition,
            accounting is malformed, or a completed job's attestation does not
            match (the same refusals collect gives).
    """
    identifier = wrds_jobs.job_id(run, attempt)
    lines = _identity_lines(run, attempt, identifier)
    state = wrds_scheduler.status(run["host"], identifier)
    lines.append(f"scheduler state={state}")
    if state in {"completed", "failed"}:
        record = wrds_scheduler.accounting(run["host"], identifier)
        lines.append(
            f"accounting job {identifier} hostname={record['hostname']} "
            f"failed={record['failed']} exit_status={record['exit_status']}"
        )
    if state == "completed":
        result = wrds_results.attested_completion(run, attempt, identifier)
        outputs = require_dict(result, "outputs")
        lines.extend(
            f"output {name} sha256={narrow_json_to_str(outputs[name])}" for name in run["outputs"]
        )
        record_digest = hashlib.sha256(wrds_results.run_record_bytes(result)).hexdigest()
        lines.append(f"run-record sha256={record_digest}")
    else:
        lines.append("collect refused: " + wrds_results.not_complete(identifier, state))
        directory = wrds_jobs.attempt_directory(run, attempt)
        lines.append("completion.json " + completion_presence(run["host"], directory))
    return "\n".join(lines)


__all__ = ["completion_presence", "render"]
