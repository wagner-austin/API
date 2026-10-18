"""The scheduled entry point: source the untracked credentials, run one cycle.

Native-Python replacement for ``run-cycle.ps1``, written 2026-09-09 after the
Windows-Update reboot proved the PowerShell form has TWO independent ways to
die silently as a scheduled task on this box:

* registered ``LogonType=Interactive``, Task Scheduler MISSES every trigger
  while no desktop session exists (37 missed runs, ``Start-ScheduledTask`` a
  silent no-op, measured 2026-09-09 10:29Z-12:23Z);
* registered ``LogonType=S4U``, ``powershell.exe`` itself deadlocks before
  reaching the script (measured on this box by the Docker-autostart work,
  board 11:48Z post: stall at .NET assembly load, ~0.16s CPU across 157s,
  repeatable; a NATIVE binary action does not).

``python -m scripts.run_cycle --package-root <dir> --workspace <MCPs>
--python <interpreter>`` is the native-binary action that pattern calls
for. The root is an ARGUMENT, not derived from ``__file__``: the package
tree holds untracked state (``runs/env.ps1``, ``runs/cycle.log``) that
exists on the operating machine and not on a clean checkout, and an entry
point that reaches for its own tree is untestable without that state —
which is exactly how this job shipped red in CI while green locally (board,
2026-09-09 20:39Z: five CI runs, five failures, zero passes, invisible
because the job usually does not run).

THE MACHINE IS AN ARGUMENT TOO (MCPs board task c39ee056). The MCPs hub
runner's ``hpc-wake`` row runs this wherever the control duty is held,
filling ``--workspace`` with that machine's MCPs checkout and ``--python``
with its system interpreter, the one MCPs' maketools rows run under. The
table spelled austinpc's ``C:\\Users\\Test`` paths and Windows separators
until the duty moved to diphtheria on 2026-10-18, where its first tick
looked for ``..\\ci-wake`` as one directory name; every path here is now
built from those two arguments or written with ``/``, which both
platforms read.

Behaviour is the PowerShell script's, deliberately: source ``runs/env.ps1``
(still PowerShell syntax so interactive sessions can keep dot-sourcing it —
parsed strictly by :mod:`platform_core.env_assignments`, the one parser the
fleet ticks share, refusing any uncommented line that is not a plain
``$env:NAME = 'value'`` assignment), truncate ``runs/cycle.log`` past ~1 MB,
stamp a UTC header, append the cycle's output, exit with the cycle's own
status so the scheduler's task history stays the health record.
"""

from __future__ import annotations

import os
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Final, TextIO

from platform_core.env_assignments import parse_env_assignments
from typing_extensions import TypedDict

from scripts import _test_hooks, pump_health

PACKAGE_ROOT_FLAG: Final[str] = "--package-root"
WORKSPACE_FLAG: Final[str] = "--workspace"
PYTHON_FLAG: Final[str] = "--python"

#: Every flag the entry point takes, each exactly once and each required.
FLAGS: Final[tuple[str, ...]] = (PACKAGE_ROOT_FLAG, WORKSPACE_FLAG, PYTHON_FLAG)

#: The session host, where the operator's Claude sessions run their make
#: commands and so write their fleet-lock and action journals (MCPs board
#: task 701151b4), and the path of its MCPs checkout there.
SESSION_HOST: Final[str] = "ziggurat"
SESSION_HOST_CHECKOUT: Final[str] = "/home/corvis/PROJECTS/MCPs"

#: maketools' launcher, relative to the MCPs checkout.
MAKETOOLS_RUN: Final[str] = "packages/maketools/scripts/run.py"


class PumpArguments(TypedDict):
    """What one invocation names.

    Attributes:
        package_root: The hpc-wake package directory holding ``runs/``.
        workspace: This machine's MCPs checkout, where the hub's journals,
            the cursors of the remote ones and maketools live.
        python: The interpreter maketools' launcher runs under.
    """

    package_root: pathlib.Path
    workspace: pathlib.Path
    python: str


class Publisher(TypedDict):
    """One publisher the pump runs each tick.

    Attributes:
        name: The marker written to the cycle log before this publisher's
            output, so a red tick names its red half.
        args: The command, run to completion with captured output.
        cwd: Working directory, RELATIVE to the pump's package root or
            absolute — ``poetry run`` resolves its project from the cwd,
            which is how one pump drives entry points from several poetry
            projects.
    """

    name: str
    args: tuple[str, ...]
    cwd: str


def publishers(workspace: pathlib.Path, python: str) -> tuple[Publisher, ...]:
    """The pump's publishers, run IN ORDER each tick.

    The one scheduled job the board's event system rides (board task
    9406cfd9: publishers are added HERE, never as sibling scheduled tasks).
    Order is publication order only; each runs regardless of the previous
    one's exit status, and the tick's own status is the first nonzero exit
    so the scheduler's history stays the health record while the log names
    the failing half.

    Two rows the table held while austinpc ran it left it with the control
    duty (MCPs board task c39ee056): hub-tunnel-wake read the journal of
    austinpc's reverse tunnel, and rcs-pair-watch read the bridge log beside
    rcs-bridge, both files on the machine holding the devices duty, which
    the control duty's machine cannot read; rcs-pair-watch is a devices row
    of MCPs' hub runner, and the tunnel retires with austinpc's devices.

    Args:
        workspace: This machine's MCPs checkout.
        python: The interpreter maketools' launcher runs under.

    Returns:
        The rows, each command and directory built from the arguments.
    """
    journals = str(workspace)
    return (
        {
            "name": "hpc-wake",
            "args": ("poetry", "run", "hpc-wake", "--config", "../hpc3/runs/hpc3.json"),
            "cwd": ".",
        },
        # ci-wake (board 21:02Z, bridge-ci-wake-0909): GitHub Actions -> board,
        # with the pre-push enrolment ledger supplying the @mention target the
        # Actions API cannot know. Its standing task id rides runs/env.ps1 as
        # CI_WAKE_TASK_ID beside the pump's other credentials.
        {
            "name": "ci-wake",
            "args": ("poetry", "run", "ci-wake", "--enrolment", "runs/pushes.jsonl"),
            "cwd": "../ci-wake",
        },
        # lock-wake (board 9406cfd9 publisher b): the hub's fleet-lock journal
        # -> board, where every deploy takes its lock. One post per tick at
        # most, boundaries only, progress folded as counts; the journal's
        # 'agent' field (MCPs 66b85d32) supplies the mention target. Standing
        # task id: LOCK_WAKE_TASK_ID in runs/env.ps1. The check journal beside
        # it (MCPs board task ea2ea29c) carries the check lock's finished make
        # test runs, posted unaddressed in the same post; it is its own file
        # because older readers of the fleet journal refuse a kind they do not
        # declare. The action journal (MCPs board task b00ab472) carries every
        # MCPs command that changes state and takes no lock; an action that
        # failed or was refused is posted under ACTIONS and tags who ran it,
        # and its own file because the fleet journal's readers read that file
        # whole.
        {
            "name": "lock-wake",
            "args": (
                "poetry",
                "run",
                "lock-wake",
                "--journal",
                f"{journals}/.fleet-events.jsonl",
                "--check-journal",
                f"{journals}/.check-events.jsonl",
                "--action-journal",
                f"{journals}/.action-events.jsonl",
            ),
            "cwd": "../lock-wake",
        },
        # fleet-health-wake (MCPs board task ebc80a03): the fleet audit's health
        # journal -> board. The audit appends one line per run whose outcome
        # changed, the note already rendered; this row posts the unread ones
        # as one note, unaddressed. Standing task id:
        # FLEET_HEALTH_WAKE_TASK_ID in runs/env.ps1.
        {
            "name": "fleet-health-wake",
            "args": (
                "poetry",
                "run",
                "fleet-health-wake",
                "--journal",
                f"{journals}/fleet-mcp/state/health-events.jsonl",
            ),
            "cwd": "../fleet-health-wake",
        },
        # lock-wake-ziggurat (MCPs board tasks 03590bf9 and c39ee056): the
        # session host's fleet-lock journal -> board, through the same bridge
        # and standing task as the lock-wake row, since the sessions there take
        # their locks in its checkout. The bridge reads only the unread window
        # over one ssh per tick and keeps its cursor on this machine, in the
        # MCPs clone root beside the hub's own
        # (.fleet-events.jsonl.lock-wake-ziggurat-offset.json, under that
        # root's /.fleet-events.jsonl.* ignore).
        {
            "name": "lock-wake-ziggurat",
            "args": (
                "poetry",
                "run",
                "lock-wake",
                "--remote-journal",
                f"{SESSION_HOST}:{SESSION_HOST_CHECKOUT}/.fleet-events.jsonl",
                "--cursor-dir",
                journals,
            ),
            "cwd": "../lock-wake",
        },
        # lock-wake-ziggurat-actions (MCPs board tasks b00ab472 and c39ee056):
        # the session host's action journal -> board, the remote half of the
        # lock-wake row's --action-journal, on the same standing task; an
        # action that failed or was refused is posted under ACTIONS tagging
        # who ran it. Its cursor is
        # .action-events.jsonl.lock-wake-ziggurat-offset.json in the same
        # clone root, under that root's /.action-events.jsonl.* ignore.
        {
            "name": "lock-wake-ziggurat-actions",
            "args": (
                "poetry",
                "run",
                "lock-wake",
                "--remote-action-journal",
                f"{SESSION_HOST}:{SESSION_HOST_CHECKOUT}/.action-events.jsonl",
                "--cursor-dir",
                journals,
            ),
            "cwd": "../lock-wake",
        },
        # login-probe (MCPs board task 2ce5fc24): the login real-use check.
        # Every three hours MCPs' maketools drives a real Chromium through
        # corvis sign-in with the probe's own passkey, a connector's
        # authorization and token refresh, one tool call and sign-out,
        # against production, and records the judged run in the hub
        # checkout's fleet-mcp/state/login-probe.json; when the set of failing
        # checks changes it states one critical operator ask (texted) and
        # posts a note in room corvis. A tick inside the interval returns at
        # once, and a run whose checks fail still exits 0, since the
        # measurement is the job; it exits non-zero only when the probe could
        # not run or the board refused. One run measured 8 s on 2026-10-10,
        # well under PUBLISHER_WALL_SECONDS. The interpreter MCPs' maketools
        # rows run under; the enrolment lives outside every checkout.
        {
            "name": "login-probe",
            "args": (python, "-B", MAKETOOLS_RUN, "login-probe", "--workspace", journals),
            "cwd": journals,
        },
        # search-probe (MCPs board task d84d37e3): https://search.austinwagner.org
        # -> board. MCPs maketools' search-probe GETs the public search's
        # declared checks through Cloudflare once its last run is 540 s old
        # (every 9 to 12 minutes at this pump's 3), keeps the runs in the hub
        # checkout's fleet-mcp/state/search-probe.json for fleet_search_probe,
        # and when the set of failing checks changes states one repair ask and
        # posts a note in room civic. A run whose checks fail still exits 0; a
        # red row means the probe could not measure or post, never a slow
        # search. Last in the table, so a slow site delays no other row's
        # post. The same interpreter as the row above.
        {
            "name": "search-probe",
            "args": (python, "-B", MAKETOOLS_RUN, "search-probe", "--workspace", journals),
            "cwd": journals,
        },
    )


#: The row run after every publisher above, each tick: fleet's record of
#: which commit each of them ran and how it exited, through MCPs
#: ``dispatch_publisher_tick`` (MCPs board task 9f1367d5). Which commit the
#: hub runs was recorded only here, in files no audit check reads, so a
#: criterion that the hub runs a fix could not be decided; the row is the
#: database's copy. Its args gain one ``<publisher>=<exit code>=<directory>``
#: token per publisher this tick, which is why it is not a row of the table
#: above. It is a row like them in every other way: marked in the log and
#: given its own line in the health record, so a record that could not be
#: made reads as one red row and stops no publisher.
RECORDER: Final[Publisher] = {
    "name": "publisher-ticks",
    "args": ("poetry", "run", "--", "python", "-m", "fleet.cli.publisher_ticks"),
    "cwd": "../fleet",
}


def arguments_from(tokens: Sequence[str]) -> PumpArguments:
    """Parse the three flags this entry point takes, with stdlib only.

    STDLIB-ONLY IS A CONSTRAINT, NOT A STYLE CHOICE: the scheduled task
    runs this file under the SYSTEM python — the poetry venv begins
    inside the cycle's own subprocess — so a first-party import here
    resolves against whatever stale copy the system interpreter happens
    to hold, or nothing. Measured 2026-09-09 20:42Z: an import of
    ``platform_core.cli_args`` found an ancient site-packages install
    without the module and every scheduled cycle exited 1 before
    writing a log header.

    Args:
        tokens: Command-line arguments excluding the program name.

    Returns:
        The package root, the MCPs checkout and the interpreter.

    Raises:
        ValueError: For anything other than each of :data:`FLAGS` exactly
            once with a value — a defaulted root would reach for this
            file's own tree, which is the untestable dependence the flag
            removes, and a defaulted checkout or interpreter would name
            one machine's.
    """
    usage = (
        f"usage: python -m scripts.run_cycle {PACKAGE_ROOT_FLAG} <dir> "
        f"{WORKSPACE_FLAG} <MCPs checkout> {PYTHON_FLAG} <interpreter>; got {list(tokens)!r}"
    )
    names = list(tokens[0::2])
    values = list(tokens[1::2])
    if len(tokens) != 2 * len(FLAGS) or sorted(names) != sorted(FLAGS):
        raise ValueError(usage)
    given = dict(zip(names, values, strict=True))
    return {
        "package_root": pathlib.Path(given[PACKAGE_ROOT_FLAG]),
        "workspace": pathlib.Path(given[WORKSPACE_FLAG]),
        "python": given[PYTHON_FLAG],
    }


_LOG_LIMIT_BYTES = 1_000_000

#: Wall-clock bound on one publisher. Twenty minutes: twice the longest
#: deadline any publisher holds inside itself (hpc3's ten-minute cluster
#: command, which the hpc-wake publisher's ssh rides), so no honest run is
#: cut short, and still FINITE. Before it the pump had none, and a
#: publisher that never returned would have held every later tick forever,
#: the shape that drained the fleet queue for three days in September 2026.
PUBLISHER_WALL_SECONDS: Final[int] = 1200


def _run_row(
    handle: TextIO,
    name: str,
    args: Sequence[str],
    cwd: pathlib.Path,
    env: Mapping[str, str],
) -> int:
    """Run one row to completion, its output under its marker in the log.

    Args:
        handle: The cycle log, open for appending.
        name: The row's marker.
        args: Its command.
        cwd: Its resolved working directory.
        env: Its complete environment.

    Returns:
        Its exit code.

    Raises:
        OSError: When the command cannot be spawned.
        subprocess.TimeoutExpired: When it outlasts
            :data:`PUBLISHER_WALL_SECONDS`.
    """
    handle.write(f"-- {name}\n")
    completed = _test_hooks.run_process(
        list(args), cwd=cwd, env=env, timeout=PUBLISHER_WALL_SECONDS
    )
    handle.write(completed.stdout)
    handle.write(completed.stderr)
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bridge cycle and append its output to the cycle log.

    Args:
        argv: Command-line arguments excluding the program name — the three
            required flags of :func:`arguments_from`. Arguments rather than
            ``__file__`` derivations so the whole function is exercisable
            against a temporary tree. Defaults to the process arguments.

    Returns:
        The tick's status: 0 when every publisher and the :data:`RECORDER`
        row after them exited 0, otherwise the FIRST nonzero exit — every
        row runs regardless, and the log's per-row markers name which one
        went red.

    Raises:
        ValueError: Propagated from :func:`arguments_from`.
        AppError: ``CONFIG_ERROR`` from
            :func:`platform_core.env_assignments.parse_env_assignments` when
            ``runs/env.ps1`` holds a line that is not a plain assignment.
        OSError: When the log or the package tree is unwritable/unreadable,
            or a publisher's command cannot be spawned — a spawn failure is
            a broken pump, not a publisher outcome, and the header already
            written puts the evidence in the log before the scheduler
            records the crash.
        subprocess.TimeoutExpired: When a publisher outlasts
            :data:`PUBLISHER_WALL_SECONDS`; it is killed and the tick ends
            there, as a crash the scheduler records, with the health record
            left unwritten so its age says the pump stopped.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    arguments = arguments_from(tokens)
    package_root = arguments["package_root"]
    env_file = package_root / "runs" / "env.ps1"
    environment = dict(
        parse_env_assignments(env_file.read_text(encoding="utf-8"), source=str(env_file))
    )

    log = package_root / "runs" / "cycle.log"
    if log.exists() and log.stat().st_size > _LOG_LIMIT_BYTES:
        log.unlink()

    now = _test_hooks.now()
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    env = {**os.environ, **environment}
    outcomes: list[tuple[str, int]] = []
    ran: list[str] = []
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"== {stamp}\n")
        for publisher in publishers(arguments["workspace"], arguments["python"]):
            cwd = (package_root / publisher["cwd"]).resolve()
            code = _run_row(handle, publisher["name"], publisher["args"], cwd, env)
            outcomes.append((publisher["name"], code))
            ran.append(f"{publisher['name']}={code}={cwd}")
        recorded = _run_row(
            handle,
            RECORDER["name"],
            (*RECORDER["args"], *ran),
            (package_root / RECORDER["cwd"]).resolve(),
            env,
        )
        outcomes.append((RECORDER["name"], recorded))
    worst = next((code for _, code in outcomes if code != 0), 0)
    # THE HEALTH RECORD IS WRITTEN EVERY TICK, GREEN OR RED, and it is the
    # only part of this function anything outside the machine's scheduler
    # can read. The log above is where a five-day outage hid: it is
    # truncated past 1 MB, so by the time anyone looked for the 2026-09-16
    # start of it there were 80 minutes of ticks left. See scripts/pump_health.
    health_path = package_root / "runs" / pump_health.HEALTH_FILENAME
    pump_health.write_health(
        health_path,
        pump_health.next_health(pump_health.read_health(health_path), outcomes, now),
    )
    return worst


if __name__ == "__main__":
    sys.exit(main())
