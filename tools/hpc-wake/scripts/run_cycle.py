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

``python.exe <this file>`` is the native-binary action that pattern calls
for. The task registration lives in the README beside the old one.

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

from scripts import _test_hooks

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


def main(package_root: pathlib.Path) -> int:
    """Run one bridge cycle and append its output to the cycle log.

    Args:
        package_root: The hpc-wake package directory, holding ``runs/`` —
            passed rather than derived so the whole function is exercisable
            against a temporary tree.

    Returns:
        The cycle's own exit status.

    Raises:
        ValueError: Propagated from :func:`load_env_assignments`.
        OSError: When the log or the package tree is unwritable/unreadable.
    """
    environment = load_env_assignments(package_root / "runs" / "env.ps1")

    log = package_root / "runs" / "cycle.log"
    if log.exists() and log.stat().st_size > _LOG_LIMIT_BYTES:
        log.unlink()

    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    completed = _test_hooks.run_process(
        ["poetry", "run", "hpc-wake", "--config", "..\\hpc3\\runs\\hpc3.json"],
        cwd=package_root,
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"== {stamp}\n")
        handle.write(completed.stdout)
        handle.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(__file__).resolve().parent.parent))
