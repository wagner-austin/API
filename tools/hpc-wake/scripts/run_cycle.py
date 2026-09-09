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

``python.exe -m scripts.run_cycle --package-root <dir>`` is the
native-binary action that pattern calls for. The root is an ARGUMENT, not
derived from ``__file__``: the package tree holds untracked state
(``runs/env.ps1``, ``runs/cycle.log``) that exists on the operating machine
and not on a clean checkout, and an entry point that reaches for its own
tree is untestable without that state — which is exactly how this job
shipped red in CI while green locally (board, 2026-09-09 20:39Z: five CI
runs, five failures, zero passes, invisible because the job usually does
not run). The task registration lives in the README's Scheduling section.

Behaviour is the PowerShell script's, deliberately: source ``runs/env.ps1``
(still PowerShell syntax so interactive sessions can keep dot-sourcing it —
parsed here strictly, refusing any uncommented line that is not a plain
``$env:NAME = 'value'`` assignment), truncate ``runs/cycle.log`` past ~1 MB,
stamp a UTC header, append the cycle's output, exit with the cycle's own
status so the scheduler's task history stays the health record.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import re
import sys
from collections.abc import Sequence
from typing import Final

from typing_extensions import TypedDict

from scripts import _test_hooks

PACKAGE_ROOT_FLAG = "--package-root"


class Publisher(TypedDict):
    """One publisher the pump runs each tick.

    Attributes:
        name: The marker written to the cycle log before this publisher's
            output, so a red tick names its red half.
        args: The command, run to completion with captured output.
        cwd: Working directory RELATIVE to the pump's package root —
            ``poetry run`` resolves its project from the cwd, which is how
            one pump drives entry points from several poetry projects.
    """

    name: str
    args: tuple[str, ...]
    cwd: str


#: The pump's publishers, run IN ORDER each tick — the one scheduled task
#: the board's event system rides (board task 9406cfd9: publishers are
#: added HERE, never as sibling scheduled tasks). Order is publication
#: order only; each runs regardless of the previous one's exit status,
#: and the tick's own status is the first nonzero exit so the scheduler's
#: history stays the health record while the log names the failing half.
PUBLISHERS: Final[tuple[Publisher, ...]] = (
    {
        "name": "hpc-wake",
        "args": ("poetry", "run", "hpc-wake", "--config", "..\\hpc3\\runs\\hpc3.json"),
        "cwd": ".",
    },
    # ci-wake (board 21:02Z, bridge-ci-wake-0909): GitHub Actions -> board,
    # with the pre-push enrolment ledger supplying the @mention target the
    # Actions API cannot know. Its standing task id rides runs/env.ps1 as
    # CI_WAKE_TASK_ID beside the pump's other credentials.
    {
        "name": "ci-wake",
        "args": ("poetry", "run", "ci-wake", "--enrolment", "runs\\pushes.jsonl"),
        "cwd": "..\\ci-wake",
    },
)


def package_root_from(tokens: Sequence[str]) -> pathlib.Path:
    """Parse the one flag this entry point takes, with stdlib only.

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
        The hpc-wake package directory named by ``--package-root``.

    Raises:
        ValueError: For anything other than exactly that one flag and
            its value — a defaulted root would reach for this file's own
            tree, which is the untestable dependence this flag removes.
    """
    if len(tokens) != 2 or tokens[0] != PACKAGE_ROOT_FLAG:
        raise ValueError(
            f"usage: python -m scripts.run_cycle {PACKAGE_ROOT_FLAG} <dir>; "
            f"got {list(tokens)!r} — the root is an argument precisely so "
            f"no invocation ever reaches for this file's own tree"
        )
    return pathlib.Path(tokens[1])


_ASSIGNMENT = re.compile(r"^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'([^']*)'\s*$")
_LOG_LIMIT_BYTES = 1_000_000


def load_env_assignments(env_file: pathlib.Path) -> dict[str, str]:
    """Parse the untracked credentials file's assignments.

    Args:
        env_file: ``runs/env.ps1``, holding ``$env:NAME = 'value'`` lines
            beside comments and blank lines.

    Returns:
        The assignments, in file order.

    Raises:
        ValueError: For any uncommented, non-blank line that is not a plain
            single-quoted assignment — a credential this parser silently
            skipped would surface later as an unauthenticated cycle, which
            is the failure mode this refusal exists to prevent.
    """
    assignments: dict[str, str] = {}
    for raw in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        matched = _ASSIGNMENT.match(line)
        if matched is None:
            raise ValueError(
                f"unparseable line in {env_file}: {line!r} — this loader accepts only "
                f"plain $env:NAME = 'value' assignments, and skipping one would run "
                f"the cycle with a credential missing"
            )
        assignments[matched.group(1)] = matched.group(2)
    return assignments


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bridge cycle and append its output to the cycle log.

    Args:
        argv: Command-line arguments excluding the program name — required
            flag ``--package-root``, the hpc-wake package directory holding
            ``runs/``. An argument rather than a ``__file__`` derivation so
            the whole function is exercisable against a temporary tree.
            Defaults to the process arguments.

    Returns:
        The tick's status: 0 when every publisher exited 0, otherwise the
        FIRST nonzero exit — every publisher runs regardless, and the log's
        per-publisher markers name which one went red.

    Raises:
        ValueError: Propagated from :func:`package_root_from` or
            :func:`load_env_assignments`.
        OSError: When the log or the package tree is unwritable/unreadable,
            or a publisher's command cannot be spawned — a spawn failure is
            a broken pump, not a publisher outcome, and the header already
            written puts the evidence in the log before the scheduler
            records the crash.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    package_root = package_root_from(tokens)
    environment = load_env_assignments(package_root / "runs" / "env.ps1")

    log = package_root / "runs" / "cycle.log"
    if log.exists() and log.stat().st_size > _LOG_LIMIT_BYTES:
        log.unlink()

    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    worst = 0
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"== {stamp}\n")
        for publisher in PUBLISHERS:
            handle.write(f"-- {publisher['name']}\n")
            completed = _test_hooks.run_process(
                list(publisher["args"]),
                cwd=(package_root / publisher["cwd"]).resolve(),
                env={**os.environ, **environment},
                capture_output=True,
                text=True,
            )
            handle.write(completed.stdout)
            handle.write(completed.stderr)
            if completed.returncode != 0 and worst == 0:
                worst = completed.returncode
    return worst


if __name__ == "__main__":
    sys.exit(main())
