"""Validating a job against the live scheduler without running it.

Three checks, cheapest first, each catching a failure the previous one cannot
see:

1. The environment exists. A job whose ``env_path`` is absent starts, fails
   on the first import, and returns the node -- having consumed a queue slot
   to learn something a one-line test would have said instantly.
2. ``sbatch --test-only`` on the REAL rendered script. Not a reconstruction
   of it from flags: a dry run that exercises different bytes than the real
   submission is a dry run that can pass while the real thing fails.
3. The scheduler's verdict is parsed rather than pattern-matched loosely, so
   a changed output format fails loudly instead of being read as success.

A rejection raises. It is not returned as a result to inspect, because a
rejected job is a failure the caller must handle, and handing back a
"would_run: false" object invites treating it as advisory.
"""

from __future__ import annotations

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.dependency import describe_dependency
from hpc3.contracts.job import JobSpec
from hpc3.contracts.layout import qualified_name
from hpc3.contracts.preflight import PreflightResult, decode_preflight_result
from hpc3.core import env_probe, image_exec, remote, sbatch

_START_ANCHOR = " to start at "
_USING_ANCHOR = " using "
_PROCESSORS_ANCHOR = " processors on nodes "
_PARTITION_ANCHOR = " in partition "


def check_env_path(host: str, spec: JobSpec) -> str:
    """Verify the job's Python environment exists where the job will look.

    Returns the verified path rather than None, matching the rest of this
    package: ``read_and_verify`` returns the bytes it verified and
    ``check_remote_digest`` returns the digest. A validator returning nothing
    can only be tested by asserting it did not raise, which asserts almost
    nothing.

    THE PROBE FOLLOWS THE JOB. For a host run ``env_path`` is a cluster
    directory and the probe runs on the cluster. For an image run it is a
    CONTAINER path -- ``/opt/env`` exists only inside the ``.sif`` and
    nowhere on the cluster filesystem -- so probing the host would refuse
    every image job for a directory that was never meant to be there.

    Args:
        host: SSH destination.
        spec: The spec whose ``env_path`` is checked, inside its ``image``
            when it declares one.

    Returns:
        The verified interpreter directory, ``<env_path>/bin``.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.ENV_PATH_MISSING` if
            the directory is absent. Checked as ``<env_path>/bin`` rather
            than the root, because an empty directory of the right name would
            otherwise pass and then fail at the first interpreter lookup. The
            message names the filesystem that was actually searched, so a
            reader is not sent to look on the host for a container path.
    """
    bin_dir = f"{spec['env_path']}/bin"
    image = spec["image"]
    probe = f"test -d '{bin_dir}' && echo PRESENT || echo ABSENT"
    if image is not None:
        probe = image_exec.run_inside_image(image, probe)
    if remote.run_remote(host, probe).strip() != "PRESENT":
        raise AppError(
            Hpc3ErrorCode.ENV_PATH_MISSING,
            f"{bin_dir} does not exist {image_exec.describe_location(image, host)}. "
            "The job would start, fail on its first import, and return the node "
            "having learned nothing.",
        )
    return bin_dir


def _take(text: str, cursor: int, anchor: str, field: str) -> tuple[str, int]:
    """Read from the cursor up to the next anchor, and step past it.

    A forward walk rather than independent searches: each field's starting
    point is where the previous field ended, so there is exactly one way for
    each step to fail and no unreachable "start not found" branch.

    Args:
        text: Line being parsed.
        cursor: Index to read from.
        anchor: Text that terminates this field.
        field: Name of the field, used in the error message.

    Returns:
        The field's stripped text, and the index just past the anchor.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.PREFLIGHT_UNPARSABLE`
            if the anchor does not appear. Slurm's phrasing is not a stable
            API, so a changed format must fail loudly rather than yield a
            plausible empty string.
    """
    tail = text.find(anchor, cursor)
    if tail == -1:
        raise AppError(
            Hpc3ErrorCode.PREFLIGHT_UNPARSABLE,
            f"sbatch --test-only output has no {field!r} terminator ({anchor.strip()!r}): {text!r}",
        )
    return text[cursor:tail].strip(), tail + len(anchor)


def parse_test_only(output: str, cluster: ClusterFacts) -> PreflightResult:
    """Parse ``sbatch --test-only`` output into a verdict.

    Args:
        output: The command's standard output, of the form
            ``sbatch: Job N to start at T using P processors on nodes L in
            partition Q``.

    Returns:
        The scheduler's verdict.

    Raises:
        AppError: With ``PREFLIGHT_UNPARSABLE`` if no line carries the
            expected shape, or a field cannot be read from the line that
            does.
        JSONTypeError: If a parsed field is not a value the contract accepts
            -- a non-numeric processor count or an unrecognised partition.
    """
    for line in output.splitlines():
        head = line.find(_START_ANCHOR)
        if head == -1:
            continue

        cursor = head + len(_START_ANCHOR)
        # ``... to start at 2026-08-22T03:23:00 a using 4 processors ...``
        # The trailing token before " using " is not part of the timestamp.
        start_field, cursor = _take(line, cursor, _USING_ANCHOR, "start estimate")
        processors, cursor = _take(line, cursor, _PROCESSORS_ANCHOR, "processors")
        node_list, cursor = _take(line, cursor, _PARTITION_ANCHOR, "node list")

        if not processors.isdigit():
            raise AppError(
                Hpc3ErrorCode.PREFLIGHT_UNPARSABLE,
                f"sbatch --test-only reported a non-numeric processor count "
                f"{processors!r}: {line!r}",
            )
        return decode_preflight_result(
            {
                "start_estimate": start_field.split()[0],
                "processors": int(processors),
                "node_list": node_list,
                "partition": line[cursor:].strip(),
            },
            cluster,
        )
    raise AppError(
        Hpc3ErrorCode.PREFLIGHT_UNPARSABLE,
        f"sbatch --test-only announced no start estimate; got {output.strip()!r}.",
    )


_DEPENDENCY_REFUSALS = ("presently disabled", "Job dependency problem")
"""What Slurm says when ``--test-only`` cannot evaluate a dependency.

Measured on HPC3 2026-08-23, both against a real refusal:

* ``afterok`` on a job that has already FAILED gives ``Requested operation is
  presently disabled``.
* ``afterok`` on a job the controller no longer holds -- anything past
  ``MinJobAge``, 300 seconds here -- gives ``Job dependency problem``.

Neither says anything about a dependency, and the first says nothing at all.
"""


def _dependency_hint(spec: JobSpec, output: str) -> str:
    """Explain a refusal that is really about the job this one waits on.

    Args:
        spec: The spec that was refused.
        output: Slurm's own output, already captured.

    Returns:
        A sentence naming the dependency, or an empty string when the refusal
        was about something else. The hint is appended rather than replacing
        Slurm's text: its wording is still the ground truth, and a translator
        that swallowed it would hide any refusal this table guesses wrong.
    """
    depends_on = spec["depends_on"]
    if depends_on is None:
        return ""
    if not any(phrase in output for phrase in _DEPENDENCY_REFUSALS):
        return ""
    return (
        f" This job waits on {describe_dependency(depends_on)}, and Slurm cannot "
        "evaluate that: the job it names has already failed, or is old enough that "
        "the controller no longer holds it. In a chain that means an earlier stage "
        "failed, and nothing after it was submitted."
    )


def preflight(
    spec: JobSpec,
    *,
    host: str,
    script_dir: str,
    log_dir: str,
    cluster: ClusterFacts,
    charge_account: str,
) -> PreflightResult:
    """Validate a job against the live scheduler without running it.

    The rendered script is uploaded and tested by path, so the bytes the
    scheduler admits are the bytes a later submission will run. Nothing is
    queued and no allocation is made.

    Args:
        spec: A spec already validated by
            :func:`~hpc3.contracts.job.decode_job_spec`.
        host: SSH destination.
        script_dir: Absolute cluster directory to hold the batch script.
        log_dir: Absolute cluster directory the script will name for output.
        charge_account: Slurm account to bill, or empty for none. Carried
            because the script the scheduler tests must be the script that
            later runs -- a preflight that omitted the directive would pass
            a job the real submission is refused for.
        cluster: The cluster the workspace selected. The scheduler's echoed
            partition is checked against it, so a workspace pointed at the
            wrong machine fails here rather than after the job is queued.

    Returns:
        The scheduler's verdict, including an estimated start.

    Raises:
        AppError: With ``ENV_PATH_MISSING`` if the environment is absent,
            ``ENV_PACKAGE_MISMATCH`` if it exists but does not contain what
            the project pinned, ``ENV_PROBE_UNREADABLE`` if it cannot say what
            it contains, ``PREFLIGHT_REJECTED`` if Slurm refuses the job --
            carrying its own reason, which is the diagnostic -- or
            ``PREFLIGHT_UNPARSABLE`` if the verdict cannot be read.
    """
    check_env_path(host, spec)
    # Existence, then identity. The path check catches a typo; this catches
    # the more expensive mistake of a real environment that is the wrong one.
    env_probe.verify_env_packages(
        host, spec["env_path"], spec["pinned_packages"], image=spec["image"]
    )

    remote.make_directory(host, script_dir)
    remote.make_directory(host, log_dir)
    script = sbatch.render_sbatch(spec, log_dir=log_dir, charge_account=charge_account)
    label = qualified_name(spec["project"], spec["name"])
    script_path = f"{script_dir}/{label}.sbatch"
    remote.put_bytes(host, script_path, script.encode("utf-8"))

    probe = f'cd {script_dir} && sbatch --test-only {label}.sbatch 2>&1; echo "rc=$?"'
    output = remote.run_remote(host, probe)
    if "rc=0" not in output:
        raise AppError(
            Hpc3ErrorCode.PREFLIGHT_REJECTED,
            f"Slurm would refuse {spec['name']!r}: {output.strip()}"
            + _dependency_hint(spec, output),
        )
    return parse_test_only(output, cluster)


__all__ = ["check_env_path", "parse_test_only", "preflight"]
