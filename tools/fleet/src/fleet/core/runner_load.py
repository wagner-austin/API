"""Sampling what the CI hosts are actually running, one process tree at a time.

Built to the spec @opus-ci-billing-0905 posted 2026-09-08 23:20Z, after a
night in which three sessions tuned worker literals against a box none of
them could see. The measured state that motivated it: load 78 on 16 cores,
with 31 PyTorch DataLoader children alive beside 33 pythons -- processes the
`-n` literals never counted, because a worker cap bounds what xdist SPAWNS
and not what those workers spawn in turn.

THE FOUR RULES OF THE SPEC, and where each lives here:

  1. SAMPLE, DO NOT AGGREGATE. One invocation records one sample: a
     timestamp and, per runner install, whether its Runner.Worker is live
     and how many DESCENDANT processes it carries. Totals are computed at
     report time, never stored -- "32 right now" and "16 twice" are the
     same total and different facts.
  2. THE ANSWER IS A DISTRIBUTION, NOT A MAX. The report prints the
     fraction of samples whose total exceeds the core count and the longest
     consecutive excursion, beside the minimum.
  3. IT MUST BE ABLE TO SAY NO. The report always prints the minimum total
     and the fraction of samples UNDER the core count; a reading that can
     only confirm oversubscription is not a measurement.
  4. NO THRESHOLD, NO ALERT. The report judges nothing. The moment it
     judged, somebody would tune the number instead of reading the
     distribution.

DESCENDANTS, NOT PATTERN MATCHES. A sample runs one ``ps -eo pid=,ppid=,args=``
inside the distro and parses the forest LOCALLY: each install's
``Runner.Worker`` is found by its install directory in the args, and
everything beneath it in the parent tree is counted -- xdist workers, vitest
workers, DataLoader children, whatever exists. A name-based count would
report 8 where the box carries 40, which is the exact blindness this exists
to end.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    require_bool,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import remote


class InstallLoad(TypedDict):
    """One runner install's share of a sample.

    Attributes:
        repo: The install's repository, from the roster.
        runner_name: The install's runner name, from the roster.
        busy: Whether a ``Runner.Worker`` for this install was alive -- the
            process GitHub starts per job, so its absence means idle.
        descendants: Live processes beneath that worker, transitively. Zero
            when idle.
    """

    repo: str
    runner_name: str
    busy: bool
    descendants: int


class LoadSample(TypedDict):
    """One reading of one host.

    Attributes:
        at: Whole seconds since the epoch, from the clock hook.
        host: The roster host name.
        installs: Every install's share, in roster order.
        total: Sum of descendants across installs. Stored as well as
            derivable because the report reads many samples and the sum is
            the column it reasons over; the per-install rows stay so the sum
            can always be re-derived and challenged.
    """

    at: int
    host: str
    installs: list[InstallLoad]
    total: int


def parse_process_forest(ps_output: str) -> dict[int, tuple[int, str]]:
    """Parse ``ps -eo pid=,ppid=,args=`` output.

    Args:
        ps_output: The command's standard output.

    Returns:
        Each pid mapped to its parent pid and its args string. Lines that do
        not begin with two integers are refused rather than skipped -- a
        mangled forest miscounts silently, and the one transport known to
        mangle here (UTF-16 out of wsl.exe) mangles every line.

    Raises:
        AppError: ``RUNNER_AUDIT_UNPARSABLE`` on a malformed line.
    """
    forest: dict[int, tuple[int, str]] = {}
    for line in ps_output.splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 2)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise AppError(
                FleetErrorCode.RUNNER_AUDIT_UNPARSABLE,
                f"ps output line is not 'pid ppid args': {line!r}",
            )
        args = parts[2] if len(parts) == 3 else ""
        forest[int(parts[0])] = (int(parts[1]), args)
    return forest


def _descendant_count(forest: dict[int, tuple[int, str]], root: int) -> int:
    """Count every process beneath one, transitively.

    Args:
        forest: The process forest.
        root: The ancestor pid.

    Returns:
        The number of descendants, excluding the root itself.
    """
    children: dict[int, list[int]] = {}
    for pid, (ppid, _) in forest.items():
        children.setdefault(ppid, []).append(pid)
    count = 0
    pending = list(children.get(root, []))
    while pending:
        current = pending.pop()
        count += 1
        pending.extend(children.get(current, []))
    return count


def _install_marker(workdir: str, side: str) -> str:
    """The Runner.Worker path fragment that identifies an install's worker.

    Args:
        workdir: The install's ``_work`` tree.
        side: The install's execution environment.

    Returns:
        For ``wsl``, the POSIX worker path (``<root>/bin/Runner.Worker``);
        for ``windows``, the backslashed form a Win32 command line carries
        (``<root>\\bin\\Runner.Worker``), since the roster records Windows
        workdirs with forward slashes but process command lines do not.
    """
    root = workdir.rsplit("/", 1)[0]
    if side == "windows":
        return root.replace("/", "\\") + "\\bin\\Runner.Worker"
    return f"{root}/bin/Runner.Worker"


#: The PowerShell payload that snapshots the WINDOWS process forest in the
#: same three-column shape ``ps -eo pid=,ppid=,args=`` gives for the distro,
#: so one parser scores both. Sent and run by path per the remote layer's
#: rule; a null CommandLine formats as empty, which the parser accepts as a
#: process with no args.
WINDOWS_FOREST_SCRIPT = (
    "Get-CimInstance Win32_Process | ForEach-Object {\n"
    "  '{0} {1} {2}' -f $_.ProcessId, $_.ParentProcessId, $_.CommandLine\n"
    "}\n"
)

#: File name the Windows forest script lands under in the host's scratch_dir.
FOREST_SCRIPT_NAME = "fleet-runner-forest.ps1"


def take_sample(
    spec: HostRunnerSpec, wsl_ps_output: str, windows_ps_output: str | None, *, at: int
) -> LoadSample:
    """Score the host's process forests against the roster.

    The two forests are scored SEPARATELY -- WSL and Windows are distinct
    pid namespaces, so merging them could alias a distro pid onto a host
    process and miscount both.

    Args:
        spec: The host's roster entry.
        wsl_ps_output: One ``ps -eo pid=,ppid=,args=`` reading from inside
            the distro.
        windows_ps_output: One Win32_Process reading in the same shape, or
            None when the roster declares no windows-side installs.
        at: The reading's timestamp, whole epoch seconds.

    Returns:
        The sample.

    Raises:
        AppError: ``RUNNER_AUDIT_UNPARSABLE`` from the forest parser.
        ValueError: When the roster declares a windows-side install and no
            Windows forest was provided -- scoring it as zero would be the
            healthiest possible reading taken while looking away, the exact
            blind spot the ``side`` field was added to close.
    """
    wsl_forest = parse_process_forest(wsl_ps_output)
    windows_forest = (
        parse_process_forest(windows_ps_output) if windows_ps_output is not None else None
    )
    installs: list[InstallLoad] = []
    total = 0
    for install in spec["installs"]:
        if install["side"] == "windows":
            if windows_forest is None:
                raise ValueError(
                    f"install {install['repo']}:{install['runner_name']} is windows-side "
                    "but no Windows process forest was provided; a zero here would be a "
                    "reading taken while looking away"
                )
            forest = windows_forest
        else:
            forest = wsl_forest
        marker = _install_marker(install["workdir"], install["side"])
        workers = [pid for pid, (_, args) in forest.items() if marker in args]
        descendants = sum(_descendant_count(forest, pid) for pid in workers)
        installs.append(
            InstallLoad(
                repo=install["repo"],
                runner_name=install["runner_name"],
                busy=bool(workers),
                descendants=descendants,
            )
        )
        total += descendants
    return LoadSample(at=at, host=spec["name"], installs=installs, total=total)


def sample_host(spec: HostRunnerSpec, *, at: int) -> LoadSample:
    """Take one live sample of one host.

    Args:
        spec: The host's roster entry.
        at: The reading's timestamp, whole epoch seconds.

    Returns:
        The sample.

    Raises:
        AppError: ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            remote layer -- a sampler that recorded zeros for a host it
            could not reach would write the healthiest possible reading at
            the exact moment it knows nothing -- and
            ``RUNNER_AUDIT_UNPARSABLE`` for a forest it cannot parse.
    """
    wsl_output = remote.run_ssh(
        spec["host"],
        ("wsl", "-d", spec["wsl_distro"], "--", "ps", "-eo", "pid=,ppid=,args="),
    )
    windows_output: str | None = None
    if any(install["side"] == "windows" for install in spec["installs"]):
        windows_output = remote.run_script(
            spec["host"],
            f"{spec['scratch_dir']}/{FOREST_SCRIPT_NAME}",
            WINDOWS_FOREST_SCRIPT,
        )
    return take_sample(spec, wsl_output, windows_output, at=at)


def encode_load_sample(sample: LoadSample) -> JSONObject:
    """Encode one sample for the record file.

    Args:
        sample: The sample.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "at": sample["at"],
        "host": sample["host"],
        "installs": [
            {
                "repo": install["repo"],
                "runner_name": install["runner_name"],
                "busy": install["busy"],
                "descendants": install["descendants"],
            }
            for install in sample["installs"]
        ],
        "total": sample["total"],
    }


def decode_load_sample(value: JSONValue) -> LoadSample:
    """Decode one recorded sample.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated sample.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or the stored total does not equal the sum of the
            per-install rows -- a record whose summary disagrees with its
            own detail was corrupted, and reporting over it would launder
            the corruption into a distribution.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"load sample must be a JSON object, got {type(value).__name__}")
    installs: list[InstallLoad] = []
    for entry in require_list(value, "installs"):
        if not isinstance(entry, dict):
            raise JSONTypeError(f"install load must be a JSON object, got {type(entry).__name__}")
        installs.append(
            InstallLoad(
                repo=require_str(entry, "repo"),
                runner_name=require_str(entry, "runner_name"),
                busy=require_bool(entry, "busy"),
                descendants=require_int(entry, "descendants"),
            )
        )
    total = require_int(value, "total")
    derived = sum(install["descendants"] for install in installs)
    if total != derived:
        raise JSONTypeError(
            f"sample total {total} does not equal the sum of its installs {derived}; "
            "a summary that disagrees with its own detail is a corrupt record"
        )
    return LoadSample(
        at=require_int(value, "at"),
        host=require_str(value, "host"),
        installs=installs,
        total=total,
    )


def render_sample_line(sample: LoadSample) -> str:
    """One JSONL line for the record file.

    Args:
        sample: The sample.

    Returns:
        The line, newline-free; the appender owns the newline.
    """
    return dump_json_str(encode_load_sample(sample))


class LoadReport(TypedDict):
    """The distribution over a sample file, judged by nobody.

    Attributes:
        samples: How many samples were read.
        minimum: The smallest total seen -- the field that lets the report
            say NO, the box was not always loaded.
        maximum: The largest total seen.
        over_fraction: Fraction of samples whose total exceeds the core
            count.
        under_fraction: Fraction at or under it. Printed as well as
            derivable, so the reading that can exonerate is always beside
            the one that can accuse.
        longest_over_streak: The most CONSECUTIVE samples over the core
            count -- a single excursion is noise; a long streak is a
            starved box.
    """

    samples: int
    minimum: int
    maximum: int
    over_fraction: float
    under_fraction: float
    longest_over_streak: int


def report_over_cores(samples: list[LoadSample], *, cores: int) -> LoadReport:
    """The distribution of recorded totals against a core count.

    Args:
        samples: Every recorded sample, in file order (which is time order,
            because the sampler appends).
        cores: The host's logical core count -- a reference line, not a
            threshold: nothing here judges.

    Returns:
        The distribution.

    Raises:
        ValueError: On an empty sample list or a non-positive core count.
            An empty file has measured nothing, and reporting zeros over it
            would read exactly like an idle box.
    """
    if cores <= 0:
        raise ValueError(f"cores must be positive, got {cores}")
    if not samples:
        raise ValueError(
            "no samples to report over; an empty record has measured nothing and "
            "must not read like an idle box"
        )
    totals = [sample["total"] for sample in samples]
    over = [total > cores for total in totals]
    streak = 0
    longest = 0
    for is_over in over:
        streak = streak + 1 if is_over else 0
        longest = max(longest, streak)
    over_count = sum(1 for is_over in over if is_over)
    return LoadReport(
        samples=len(totals),
        minimum=min(totals),
        maximum=max(totals),
        over_fraction=over_count / len(totals),
        under_fraction=(len(totals) - over_count) / len(totals),
        longest_over_streak=longest,
    )


def decode_sample_file(raw: str) -> list[LoadSample]:
    """Decode a whole JSONL record file.

    Args:
        raw: The file's contents.

    Returns:
        Every sample, in file order.

    Raises:
        JSONTypeError: From the per-sample decoder.
    """
    return [decode_load_sample(load_json_str(line)) for line in raw.splitlines() if line.strip()]


__all__ = [
    "FOREST_SCRIPT_NAME",
    "WINDOWS_FOREST_SCRIPT",
    "InstallLoad",
    "LoadReport",
    "LoadSample",
    "decode_load_sample",
    "decode_sample_file",
    "encode_load_sample",
    "parse_process_forest",
    "render_sample_line",
    "report_over_cores",
    "sample_host",
    "take_sample",
]
