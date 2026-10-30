"""A runner host's daily hygiene, read on it (MCPs board task 46c7b11b).

The unit suite pins what :data:`~fleet.core.ci_clean.CI_CLEAN_SCRIPT`
says, and nothing read whether the host runs it: lavender's ci-clean
reclaimed 0 B a day while its daemon's build cache grew to 103.7 GB
(2026-10-18T07:32Z), because no pass in it reached that cache, and the
provision that installs it is rendered for an operator to run, so a host can
keep an old copy. This case reads, on the runner host, that the installed
script is the render byte for byte, that its timer is enabled and due within a
day, that its service started within a day, by the timer or the provision's
first pass, and that its last run succeeded, which under the script's ``set -euo
pipefail`` means every docker prune in it did. The run's own report then says
that the render is what ran, after it was installed, that every pass printed
its closing line, and that the private build cache the size pass left is
within its bound (MCPs board tasks 46c7b11b and ecb13d3d, A2): read now, the
cache holds a day of builds since. It records the daemon's build cache as
BuildKit counts it now.

The fleet project ``tools/fleet-execution-runners``, pinned to the runner
host's node, runs it as the fleet's own user, whose ``docker`` group reaches
the host's daemon on its socket.
"""

from __future__ import annotations

import hashlib
import pathlib
import time
from collections.abc import Callable
from typing import Final, NamedTuple

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_int,
    require_list,
)

from fleet.contracts.runner_base import BUILD_CACHE_BOUND_GIB
from fleet.contracts.runners import HostRunnerSpec
from fleet.core.ci_clean import (
    CI_CLEAN_REPORT,
    CI_CLEAN_SCRIPT,
    REPORT_BUILD_CACHE,
    REPORT_BUILD_CACHE_BEFORE,
    REPORT_DIGEST,
    REPORT_HELD,
    REPORT_RUNNER_UPDATE,
    REPORT_RUNNING,
    REPORT_STARTED,
    REPORT_WORK_TREE,
)
from fleet.core.runner_job_scope_render import (
    HOOK_DROP_IN,
    HOOK_DROP_IN_NAME,
    JOB_HOOK_PATH,
    JOB_HOOK_SCRIPT,
)
from fleet.core.runner_unit_files import drop_in_path
from fleet.core.runner_work_clear import JOB_COMPLETED_HOOK_PATH, JOB_COMPLETED_SCRIPT
from fleet.core.tool_caches import REMOVAL_GRACE_SECONDS
from tests._host import READING_PROPERTY, command_output, unit_properties

#: Where the rendered provision installs the script.
CI_CLEAN_PATH: Final[pathlib.Path] = pathlib.Path("/usr/local/bin/ci-clean")

#: The timer that starts it daily, and the service it starts.
CI_CLEAN_TIMER: Final[str] = "ci-clean.timer"
CI_CLEAN_SERVICE: Final[str] = "ci-clean.service"

#: The longest since the service last started: the timer's daily start, plus
#: the hours a run may be late by (``Persistent=true`` catches a missed start
#: at boot). The run is the service's, not the timer's: the provision starts
#: the first pass itself, which the timer's LastTriggerUSec never records, and
#: lavender's first check, the day it joined, failed on "ci-clean.timer has
#: never fired" with that pass's report beside it.
TRIGGER_WITHIN_SECONDS: Final[int] = 26 * 3600

#: The timer is daily, so its next start is never further off than a day.
DAY_SECONDS: Final[int] = 24 * 3600

#: The host daemon's socket, named so the case never reads another daemon a
#: ``DOCKER_HOST`` in the run's environment points at.
HOST_DOCKER: Final[str] = "unix:///var/run/docker.sock"

#: How ``docker buildx du`` opens the lines of its totals. Its
#: ``printSummary`` prints ``Shared:`` and ``Private:`` only when some record
#: is shared (buildx v0.38.1 ``commands/diskusage.go``): on lavender on
#: 2026-10-30T19:57Z none was, and du printed ``Reclaimable:`` and ``Total:``
#: alone (MCPs board task 9ab83f4b).
CACHE_TOTALS: Final[tuple[str, ...]] = ("Shared:", "Private:", "Reclaimable:", "Total:")

#: The totals du prints when no record is shared.
UNSHARED_TOTALS: Final[tuple[str, ...]] = ("Reclaimable:", "Total:")

#: The size pass's bound in bytes: docker's ``gb`` counts 1024**3.
BOUND_BYTES: Final[int] = BUILD_CACHE_BOUND_GIB * 1024**3

#: How each pass's closing line opens in the report, as docker 29.1.3 prints
#: its prunes' (read on lavender's journal on 2026-10-18) and the cache half
#: its own.
PASS_LINES: Final[tuple[str, ...]] = (
    "Total reclaimed space:",
    "Total:",
    "venv-sweep:",
    "cache-generations:",
)

#: The passes in the order the script runs them: the container, volume and
#: image prunes, the build cache's age and size passes, then the cache half.
PASS_ORDER: Final[tuple[str, ...]] = (
    *("Total reclaimed space",) * 3,
    *("Total",) * 2,
    "venv-sweep",
    "cache-generations",
)

#: The caches the hook names inside a generation, by their directories.
#: Which of them a generation holds is which tools its jobs ran: the MCPs
#: TypeScript jobs run npm, its Python jobs pip and then poetry.
GENERATION_TOOLS: Final[tuple[str, ...]] = ("npm", "pip", "pypoetry")


@pytest.mark.host_runner_units
def test_ci_clean_is_the_render_and_its_daily_run_succeeded(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    assert CI_CLEAN_TIMER in runner_host["systemd_timers"]
    assert CI_CLEAN_PATH.read_text(encoding="utf-8") == CI_CLEAN_SCRIPT

    timer = unit_properties(
        CI_CLEAN_TIMER, ("UnitFileState", "ActiveState", "NextElapseUSecRealtime")
    )
    assert (timer["UnitFileState"], timer["ActiveState"]) == ("enabled", "active")
    due = _epoch(timer["NextElapseUSecRealtime"], f"{CI_CLEAN_TIMER} has no next start")
    assert due - int(time.time()) <= DAY_SECONDS, f"{CI_CLEAN_TIMER} next starts at {due}"

    service = unit_properties(
        CI_CLEAN_SERVICE, ("Result", "ExecMainStatus", "ExecMainStartTimestamp")
    )
    since = int(time.time()) - _epoch(
        service.pop("ExecMainStartTimestamp"), f"{CI_CLEAN_SERVICE} has never run"
    )
    assert since <= TRIGGER_WITHIN_SECONDS, f"{CI_CLEAN_SERVICE} last started {since} s ago"
    assert service == {"Result": "success", "ExecMainStatus": "0"}

    report = _last_run()
    du = command_output(["docker", "-H", HOST_DOCKER, "buildx", "du"], timeout=300)
    totals = [line.split() for line in du.splitlines() if line.startswith(CACHE_TOTALS)]
    opened = tuple(total[0] for total in totals)
    assert opened in {CACHE_TOTALS, UNSHARED_TOTALS}, f"buildx du printed totals {opened}"
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: {CI_CLEAN_PATH} is the render of CI_CLEAN_SCRIPT, "
        f"{CI_CLEAN_TIMER} enabled and due within a day, {CI_CLEAN_SERVICE} started {since} s "
        f"ago, its last run {service['Result']} "
        f"with status {service['ExecMainStatus']}; {CI_CLEAN_REPORT} says the render ran "
        f"{report.after_install} s after it was installed and took the private build cache "
        f"from {report.before} bytes to {report.private}, {report.records} records, against "
        f"{BOUND_BYTES}; it reported "
        + "; ".join(report.passes)
        + "; the host daemon's build cache now reads "
        + ", ".join(" ".join(total) for total in totals),
    )


def _epoch(instant: str, unset: str) -> int:
    """An instant systemctl show printed, in seconds since the epoch.

    Args:
        instant: The property's value.
        unset: What the assertion says when systemd recorded none.

    Returns:
        The instant.
    """
    assert instant not in {"", "n/a"}, f"{unset}: {instant!r}"
    # systemd 255 prints the instant in the host's zone, which the host's own
    # date reads back; its --timestamp=unix does not reach systemctl show.
    return int(command_output(["date", "-d", instant, "+%s"]).strip())


class LastRun(NamedTuple):
    """What :data:`CI_CLEAN_REPORT` says of the last run, read against the render."""

    after_install: int
    before: int
    private: int
    records: int
    passes: tuple[str, ...]


def _last_run() -> LastRun:
    """Read the last run's report and hold it to the render and the bound.

    The report names the sha256 of the script that ran, which must be the
    render's, and its start, which must follow the installed script's mtime,
    so an old copy's success never reads as the render's. The build cache
    records it holds are the daemon's own count in bytes before the prunes
    and after the size pass, which bounds the private ones, those no other
    record shares.

    Returns:
        The reading.
    """
    lines = pathlib.Path(CI_CLEAN_REPORT).read_text(encoding="utf-8").splitlines()
    assert _fields(lines, REPORT_DIGEST) == [
        hashlib.sha256(CI_CLEAN_SCRIPT.encode("utf-8")).hexdigest()
    ]
    started = _fields(lines, REPORT_STARTED)
    assert len(started) == 1
    after_install = int(started[0]) - int(CI_CLEAN_PATH.stat().st_mtime)
    assert after_install >= 0, f"the last run started {-after_install} s before the install"

    passes = tuple(line for line in lines if line.startswith(PASS_LINES))
    assert [line.split(":")[0] for line in passes] == list(PASS_ORDER)

    before, _ = _private_build_cache(lines, REPORT_BUILD_CACHE_BEFORE)
    private, records = _private_build_cache(lines, REPORT_BUILD_CACHE)
    assert private <= BOUND_BYTES, f"{private} bytes of private build cache after the prune"
    return LastRun(after_install, before, private, records, passes)


def _private_build_cache(lines: list[str], prefix: str) -> tuple[int, int]:
    """The private bytes of one build cache reading in the report, and its records.

    Args:
        lines: The report's lines.
        prefix: How the reading's one line opens.

    Returns:
        The bytes no other record shares, and how many records there are.
    """
    reading = _fields(lines, prefix)
    assert len(reading) == 1, f"{len(reading)} lines open with {prefix!r}"
    body = narrow_json_to_dict(load_json_str(reading[0]))
    records = [narrow_json_to_dict(record) for record in require_list(body, "BuildCache")]
    private = sum(
        require_int(record, "Size") for record in records if not require_bool(record, "Shared")
    )
    return private, len(records)


@pytest.mark.host_runner_units
def test_the_jobs_read_their_caches_from_the_generation_the_hook_chose(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    """MCPs board task 4b405239, A1: the runner account's home is closed to
    the fleet's user, so ci-clean, running as that account, reports which
    generation each runner install's last job read and what the current one
    holds. A job's tools write into a generation only when the hook named it
    to them, so what it holds is the jobs' own use of it."""
    lines = pathlib.Path(CI_CLEAN_REPORT).read_text(encoding="utf-8").splitlines()
    current = [
        sweep.split(": generation ")[1].split(" ")[0]
        for sweep in _fields(lines, "cache-generations: ")
    ]
    assert len(current) == 1, f"no current generation in {CI_CLEAN_REPORT}: {current}"
    readers: dict[str, str] = {}
    for reader in _fields(lines, "cache-reader: "):
        install, generation = reader.split(" read generation ")
        readers[install] = generation
    held: dict[str, int] = {}
    for holding in _fields(lines, f"cache-tool: generation {current[0]} holds "):
        tool, size = holding.removesuffix(" bytes of files)").split(" (")
        held[tool] = int(size)
    assert len(held) == len(_fields(lines, "cache-tool: ")), "a holding of another generation"

    assert current[0] in readers.values(), f"no runner install read generation {current[0]}"
    assert held, f"generation {current[0]} was read and holds nothing"
    assert {tool.split("/")[0] for tool in held} <= set(GENERATION_TOOLS)
    for tool in held:
        if tool.startswith("npm/"):
            assert readers[tool.removeprefix("npm/")] == current[0]
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: generation {current[0]} is current, read last by "
        + ", ".join(f"{install} ({generation})" for install, generation in sorted(readers.items()))
        + "; it holds "
        + ", ".join(f"{tool} {size} bytes" for tool, size in sorted(held.items())),
    )


@pytest.mark.host_runner_units
def test_the_runner_cache_holds_no_generation_a_running_job_does_not_read(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    """MCPs board task 9ab83f4b: after ci-clean's sweep, every generation
    left is the current one, one a job running as the sweep began read, or
    one inside the removal's grace at the run's start. On lavender at
    2026-10-30T19:46Z generation 1792891106 was none of these and stayed,
    kept by the markers of idle and retired installs."""
    lines = pathlib.Path(CI_CLEAN_REPORT).read_text(encoding="utf-8").splitlines()
    started = int(_fields(lines, REPORT_STARTED)[0])
    sweeps = _fields(lines, "cache-generations: ")
    assert len(sweeps) == 1, f"{len(sweeps)} sweep lines in {CI_CLEAN_REPORT}"
    current = sweeps[0].split(": generation ")[1].split(" ")[0]
    running = set(_fields(lines, REPORT_RUNNING))
    held = _fields(lines, REPORT_HELD)
    unread = [
        generation
        for generation in held
        if generation != current
        and generation not in running
        and started - max(int(generation), int(current)) > REMOVAL_GRACE_SECONDS
    ]

    assert current in held, f"the current generation {current} is not among {held}"
    assert unread == [], f"generations no running job read, past the grace: {unread}"
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: the sweep left {len(held)} generation(s), "
        f"{', '.join(held)}; {current} is current, running jobs read "
        f"{', '.join(sorted(running)) or 'none'}, and none is left unread past "
        f"{REMOVAL_GRACE_SECONDS} s",
    )


def _fields(lines: list[str], prefix: str) -> list[str]:
    """The rest of every report line that opens with ``prefix``.

    Args:
        lines: The report's lines.
        prefix: How the wanted lines open.

    Returns:
        Each such line without its prefix, in order.
    """
    return [line.removeprefix(prefix) for line in lines if line.startswith(prefix)]


@pytest.mark.host_runner_units
def test_the_job_started_hook_that_hands_each_job_its_cache_generation_is_the_render(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    """MCPs board task 4b405239: ci-clean removes the download caches only
    by whole generations, which is safe only while every job reads the
    generation the hook names, so the installed hook must be the render.
    The generations themselves are read from ci-clean's report, below."""
    hook = pathlib.Path(JOB_HOOK_PATH)

    assert hook.read_text(encoding="utf-8") == JOB_HOOK_SCRIPT
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: {JOB_HOOK_PATH} is the render of JOB_HOOK_SCRIPT, "
        f"{len(JOB_HOOK_SCRIPT.splitlines())} lines",
    )


class WorkTree(NamedTuple):
    """One install's job trees as ci-clean's report reads them."""

    state: str
    started: str
    cleared: str
    files: int
    megabytes: int


@pytest.mark.host_runner_units
def test_each_idle_runner_install_s_job_trees_are_empty_once_its_hook_has_run(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    """MCPs board task 621902c3: on lavender on 2026-10-30 the API installs
    each kept the 6 GB torch venv their last job built until their next job.
    The job-completed hook empties a job's trees as it ends, so every unit's
    drop-in must name it and the installed hook be the render, and ci-clean's
    report, read here since the runner account's home is closed to the
    fleet's user, must find no file in the job trees of an idle install its
    hook cleared after its last job started."""
    assert pathlib.Path(JOB_COMPLETED_HOOK_PATH).read_text(encoding="utf-8") == (
        JOB_COMPLETED_SCRIPT
    )
    for install in runner_host["installs"]:
        drop_in = pathlib.Path(drop_in_path(install, HOOK_DROP_IN_NAME))
        assert drop_in.read_text(encoding="utf-8") == HOOK_DROP_IN, f"{drop_in} is not the render"
    lines = pathlib.Path(CI_CLEAN_REPORT).read_text(encoding="utf-8").splitlines()
    trees: dict[str, WorkTree] = {}
    for line in _fields(lines, REPORT_WORK_TREE):
        install_name, rest = line.split(" ", 1)
        state, started, cleared, held = rest.split(", ", 3)
        files, megabytes = held.removesuffix(" MB) in its job trees").split(" files (")
        trees[install_name] = WorkTree(
            state,
            started.removeprefix("started "),
            cleared.removeprefix("cleared "),
            int(files),
            int(megabytes),
        )
    roster = sorted(
        pathlib.PurePosixPath(install["workdir"]).parent.name for install in runner_host["installs"]
    )
    owed = [
        name
        for name, tree in sorted(trees.items())
        if tree.state == "idle"
        and tree.cleared != "never"
        and (tree.started == "never" or int(tree.cleared) >= int(tree.started))
    ]

    assert sorted(trees) == roster, f"{CI_CLEAN_REPORT} reports the installs {sorted(trees)}"
    assert [f"{name}: {trees[name].files} files" for name in owed if trees[name].files] == []
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: {JOB_COMPLETED_HOOK_PATH} is the render and "
        f"{len(runner_host['installs'])} units' {HOOK_DROP_IN_NAME} name it; ci-clean read "
        f"{len(trees)} install(s), {len(owed)} idle and cleared after their last job, each "
        "0 files; "
        + "; ".join(
            f"{name} {tree.state}, started {tree.started}, cleared {tree.cleared}, "
            f"{tree.files} files ({tree.megabytes} MB)"
            for name, tree in sorted(trees.items())
        )
        + "; "
        + ("; ".join(_fields(lines, REPORT_RUNNER_UPDATE)) or "no runner update on disk"),
    )
