"""How a WSL runner unit comes to read the action archive cache (MCPs board task ce2b92e0).

A unit takes an environment only when it starts, and a restart ends a
running job (the unit's ``KillMode`` is ``control-group``), so the drop-in
naming :data:`~fleet.core.runner_action_cache.CACHE_VARIABLE` reaches a
running unit only through a restart while it is idle.

THE RESTART WAS ONCE A PROVISION RUN'S, AND THAT NEVER CAUGHT THE BUSY ONES.
Provision restarted a unit only if it was idle at that moment and otherwise
said to run provision again. On lavender on 2026-10-15, three provision runs
at 12:41Z, 12:53Z and 13:02Z each found the four MCPs WSL runners running
jobs (``gh api repos/wagner-austin/MCPs/actions/runners`` read busy=true
for all four at each), while the API, corvis-stick and tree-bot units, idle,
took the cache at the first. So the restart is now the job of
``fleet-action-cache-adopt``, a root timer firing every minute: a unit whose
configured environment holds the setting and whose running main process
lacks it is restarted the first minute it is idle. Provision runs the same
script once per unit right after it writes the drop-in, so an idle unit
still takes the cache within the provision run.

A unit is restarted only when every one of these holds: it is running (a
stopped unit takes the drop-in whenever it next starts, and the timer
never starts one), systemd's configured environment for it holds the
setting (so a timer that runs before the drop-in exists, or after it is
removed, restarts nothing, rather than restarting an idle unit every
minute), its running main process lacks the setting, and no process in its
cgroup is a ``Runner.Worker``. Once every unit reads the cache, each firing
is two ``systemctl show`` calls and one read of ``/proc`` per unit.
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core.runner_action_cache import (
    CACHE_DROP_IN,
    CACHE_DROP_IN_NAME,
    CACHE_VARIABLE,
    WSL_CACHE_DIR,
)
from fleet.core.script_values import scriptable

#: Where the adopt script is installed.
ADOPT_PATH = "/usr/local/sbin/fleet-action-cache-adopt"

#: Its units, by file name under ``/etc/systemd/system``.
ADOPT_SERVICE_NAME = "fleet-action-cache-adopt.service"
ADOPT_TIMER_NAME = "fleet-action-cache-adopt.timer"

#: The script. Its arguments are the units it may restart.
ADOPT_SCRIPT = (
    r"""#!/usr/bin/env bash
# fleet-action-cache-adopt -- rendered by fleet-runners (MCPs board task ce2b92e0).
# Restarts each named runner unit that is configured with the action archive
# cache but running without it, and only while no job runs in it.
# Usage: fleet-action-cache-adopt UNIT...
set -euo pipefail
if [ $# -lt 1 ]; then
    echo "fleet-action-cache-adopt: usage: fleet-action-cache-adopt UNIT..." >&2
    exit 2
fi
"""
    + f"setting={CACHE_VARIABLE}={WSL_CACHE_DIR}\n"
    + r"""
# Each read is captured whole before it is matched: under pipefail, a grep -q
# that stops at its first match can kill the writer feeding it, and the
# pipeline would then read as no match.
for unit in "$@"; do
    main=$(systemctl show -p MainPID --value "$unit")
    if [ "$main" = 0 ]; then
        continue
    fi
    configured=$(systemctl show -p Environment --value "$unit")
    if [[ " $configured " != *" $setting "* ]]; then
        continue
    fi
    running=$(tr '\0' '\n' < "/proc/$main/environ" 2>/dev/null || true)
    if grep -qxF "$setting" <<<"$running"; then
        continue
    fi
    cgroup=$(systemctl show -p ControlGroup --value "$unit")
    if ! procs=$(cat "/sys/fs/cgroup$cgroup/cgroup.procs" 2>/dev/null); then
        echo "fleet-action-cache-adopt: $unit left running: its cgroup" \
            "/sys/fs/cgroup$cgroup cannot be read, so no job can be ruled out"
        continue
    fi
    busy=no
    for pid in $procs; do
        if grep -qs Runner.Worker "/proc/$pid/cmdline"; then
            busy=yes
        fi
    done
    if [ "$busy" = no ]; then
        systemctl restart "$unit"
        echo "fleet-action-cache-adopt: $unit restarted while idle, to read the action cache"
    else
        echo "fleet-action-cache-adopt: $unit is running a job;" \
            "$ADOPT_TIMER restarts it the first minute it is idle"
    fi
done
"""
).replace("$ADOPT_TIMER", ADOPT_TIMER_NAME)

#: Every minute, so a runner between two jobs of a busy hour is caught.
ADOPT_TIMER = """[Unit]
Description=Restart a runner unit to read the action cache once it is idle

[Timer]
OnBootSec=1min
OnUnitActiveSec=1min
AccuracySec=5s

[Install]
WantedBy=timers.target
"""


def wsl_units(spec: HostRunnerSpec) -> list[str]:
    """The host's WSL runner units, as the timer passes them.

    Args:
        spec: The host's roster entry.

    Returns:
        Each wsl-side install's unit, in roster order.

    Raises:
        ValueError: When a unit name cannot be embedded verbatim.
    """
    return [
        scriptable(install["service"], label="service")
        for install in spec["installs"]
        if install["side"] == "wsl"
    ]


def render_adopt_lines(spec: HostRunnerSpec) -> list[str]:
    """Bash lines that install the adopt script and its units and start its timer.

    Args:
        spec: The host's roster entry.

    Returns:
        The lines, written whole on every run as the fill's are; none when
        the host has no WSL installs, since there is nothing to adopt.

    Raises:
        ValueError: When a unit name cannot be embedded verbatim.
    """
    units = wsl_units(spec)
    if not units:
        return []
    service = "\n".join(
        [
            "[Unit]",
            "Description=Restart each idle runner unit running without the action cache",
            "",
            "[Service]",
            "Type=oneshot",
            f"ExecStart={ADOPT_PATH} " + " ".join(units),
        ]
    )
    return [
        f"cat > {ADOPT_PATH} <<'ADOPT_EOF'",
        ADOPT_SCRIPT.rstrip("\n"),
        "ADOPT_EOF",
        f"chmod +x {ADOPT_PATH}",
        f"cat > /etc/systemd/system/{ADOPT_SERVICE_NAME} <<'UNIT_EOF'",
        service,
        "UNIT_EOF",
        f"cat > /etc/systemd/system/{ADOPT_TIMER_NAME} <<'TIMER_EOF'",
        ADOPT_TIMER.rstrip("\n"),
        "TIMER_EOF",
        "systemctl daemon-reload",
        f"systemctl enable --now {ADOPT_TIMER_NAME}",
    ]


def render_cache_environment_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that give a started runner unit the cache, restarting it now only if idle.

    The drop-in is written, and systemd reloaded, only when the file does
    not already hold exactly :data:`~fleet.core.runner_action_cache.CACHE_DROP_IN`;
    then the adopt script, which :func:`render_adopt_lines` installed earlier
    in the same run, restarts the unit if it is idle and says so either way.

    Args:
        install: A wsl-side install; its ``service`` is the unit's name.

    Returns:
        The lines.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    unit = scriptable(install["service"], label="service")
    directory = f"/etc/systemd/system/{unit}.d"
    path = f"{directory}/{CACHE_DROP_IN_NAME}"
    body = CACHE_DROP_IN.replace("\n", "\\n")
    return [
        f'if [ "$(cat {path} 2>/dev/null)" != "$(printf \'{body}\')" ]; then',
        f"    mkdir -p {directory}",
        f"    printf '{body}' > {path}",
        "    systemctl daemon-reload",
        "fi",
        f"{ADOPT_PATH} {unit}",
    ]
