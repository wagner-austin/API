"""A runner host's daily hygiene, read on it (MCPs board task 46c7b11b).

The unit suite pins what :data:`~fleet.core.ci_clean.CI_CLEAN_SCRIPT`
says, and nothing read whether the host runs it: lavender's ci-clean
reclaimed 0 B a day while its daemon's build cache grew to 103.7 GB
(2026-10-18T07:32Z), because no pass in it reached that cache, and the
provision that installs it is rendered for an operator to run, so a host can
keep an old copy. This case reads, on the runner host, that the installed
script is the render byte for byte, that its timer is enabled and fired within
a day, and that its last run succeeded, which under the script's ``set -euo
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

from fleet.contracts.runners import HostRunnerSpec
from fleet.core.ci_clean import (
    BUILD_CACHE_BOUND_GIB,
    CI_CLEAN_REPORT,
    CI_CLEAN_SCRIPT,
    REPORT_BUILD_CACHE,
    REPORT_DIGEST,
    REPORT_STARTED,
)
from fleet.core.runner_job_scope_render import JOB_HOOK_PATH, JOB_HOOK_SCRIPT
from tests._host import READING_PROPERTY, command_output, unit_properties

#: Where the rendered provision installs the script.
CI_CLEAN_PATH: Final[pathlib.Path] = pathlib.Path("/usr/local/bin/ci-clean")

#: The timer that starts it daily, and the service it starts.
CI_CLEAN_TIMER: Final[str] = "ci-clean.timer"
CI_CLEAN_SERVICE: Final[str] = "ci-clean.service"

#: The longest since the timer last fired: its daily start, plus the hours a
#: run may be late by (``Persistent=true`` catches a missed start at boot).
TRIGGER_WITHIN_SECONDS: Final[int] = 26 * 3600

#: The host daemon's socket, named so the case never reads another daemon a
#: ``DOCKER_HOST`` in the run's environment points at.
HOST_DOCKER: Final[str] = "unix:///var/run/docker.sock"

#: How ``docker buildx du`` opens the lines of its totals.
CACHE_TOTALS: Final[tuple[str, ...]] = ("Private:", "Reclaimable:", "Total:")

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


@pytest.mark.host_runner_units
def test_ci_clean_is_the_render_and_its_daily_run_succeeded(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    assert CI_CLEAN_TIMER in runner_host["systemd_timers"]
    assert CI_CLEAN_PATH.read_text(encoding="utf-8") == CI_CLEAN_SCRIPT

    timer = unit_properties(CI_CLEAN_TIMER, ("UnitFileState", "LastTriggerUSec"))
    assert timer["UnitFileState"] == "enabled"
    fired = timer["LastTriggerUSec"]
    assert fired not in {"", "n/a"}, f"{CI_CLEAN_TIMER} has never fired: {fired!r}"
    # systemd 255 prints the instant in the host's zone, which the host's own
    # date reads back; its --timestamp=unix does not reach systemctl show.
    epoch = int(command_output(["date", "-d", fired, "+%s"]).strip())
    since = int(time.time()) - epoch
    assert since <= TRIGGER_WITHIN_SECONDS, f"{CI_CLEAN_TIMER} last fired {since} s ago"

    service = unit_properties(CI_CLEAN_SERVICE, ("Result", "ExecMainStatus"))
    assert service == {"Result": "success", "ExecMainStatus": "0"}

    report = _last_run()
    du = command_output(["docker", "-H", HOST_DOCKER, "buildx", "du"], timeout=300)
    totals = [line.split() for line in du.splitlines() if line.startswith(CACHE_TOTALS)]
    assert [total[0] for total in totals] == list(CACHE_TOTALS)
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: {CI_CLEAN_PATH} is the render of CI_CLEAN_SCRIPT, "
        f"{CI_CLEAN_TIMER} enabled and fired {since} s ago, its last run {service['Result']} "
        f"with status {service['ExecMainStatus']}; {CI_CLEAN_REPORT} says the render ran "
        f"{report.after_install} s after it was installed and left {report.private} bytes of "
        f"private build cache, {report.records} records, against {BOUND_BYTES}; it reported "
        + "; ".join(report.passes)
        + "; the host daemon's build cache now reads "
        + ", ".join(" ".join(total) for total in totals),
    )


class LastRun(NamedTuple):
    """What :data:`CI_CLEAN_REPORT` says of the last run, read against the render."""

    after_install: int
    private: int
    records: int
    passes: tuple[str, ...]


def _last_run() -> LastRun:
    """Read the last run's report and hold it to the render and the bound.

    The report names the sha256 of the script that ran, which must be the
    render's, and its start, which must follow the installed script's mtime,
    so an old copy's success never reads as the render's. The build cache
    records it holds are the daemon's own count in bytes after the size
    pass, which bounds the private ones, those no other record shares.

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

    cache = _fields(lines, REPORT_BUILD_CACHE)
    assert len(cache) == 1
    body = narrow_json_to_dict(load_json_str(cache[0]))
    records = [narrow_json_to_dict(record) for record in require_list(body, "BuildCache")]
    private = sum(
        require_int(record, "Size") for record in records if not require_bool(record, "Shared")
    )
    assert private <= BOUND_BYTES, f"{private} bytes of private build cache after the prune"
    return LastRun(after_install, private, len(records), passes)


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
    The runner account's home is closed to the fleet's user, so the
    generations themselves are measured as that account by the closure."""
    hook = pathlib.Path(JOB_HOOK_PATH)

    assert hook.read_text(encoding="utf-8") == JOB_HOOK_SCRIPT
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: {JOB_HOOK_PATH} is the render of JOB_HOOK_SCRIPT, "
        f"{len(JOB_HOOK_SCRIPT.splitlines())} lines",
    )
