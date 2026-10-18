"""A runner host's daily hygiene, read on it (MCPs board task 46c7b11b).

The unit suite pins what :data:`~fleet.core.runner_render.CI_CLEAN_SCRIPT`
says, and nothing read whether the host runs it: lavender's ci-clean
reclaimed 0 B a day while its daemon's build cache grew to 103.7 GB
(2026-10-18T07:32Z), because no pass in it reached that cache, and the
provision that installs it is rendered for an operator to run, so a host can
keep an old copy. This case reads, on the runner host, that the installed
script is the render byte for byte, that its timer is enabled and fired within
a day, and that its last run succeeded, which under the script's ``set -euo
pipefail`` means every docker prune in it did. It records the daemon's build
cache as BuildKit counts it, the figure the size pass bounds.

The fleet project ``tools/fleet-execution-runners``, pinned to the runner
host's node, runs it as the fleet's own user, whose ``docker`` group reaches
the host's daemon on its socket.
"""

from __future__ import annotations

import pathlib
import time
from collections.abc import Callable
from typing import Final

import pytest

from fleet.contracts.runners import HostRunnerSpec
from fleet.core.runner_render import CI_CLEAN_SCRIPT
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

    du = command_output(["docker", "-H", HOST_DOCKER, "buildx", "du"], timeout=300)
    totals = [line.split() for line in du.splitlines() if line.startswith(CACHE_TOTALS)]
    assert [total[0] for total in totals] == list(CACHE_TOTALS)
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: {CI_CLEAN_PATH} is the render of CI_CLEAN_SCRIPT, "
        f"{CI_CLEAN_TIMER} enabled and fired {since} s ago, its last run {service['Result']} "
        f"with status {service['ExecMainStatus']}; the host daemon's build cache reads "
        + ", ".join(" ".join(total) for total in totals),
    )
