"""Execute the API hook's routing around MCPs' shared publication policy.

The policy itself is exercised against real git by MCPs maketools'
test_cli_push_published.py. Here its command boundary supplies each documented
exit status while the actual shell stanza decides whether to dispatch checks.
This catches an inverted shell status, swallowed policy failure, or early exit
that would skip the hook's later CI enrollment and mirror work.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final

import pytest

ROOT: Final = Path(__file__).resolve().parents[3]
TIP: Final = "a" * 40
OTHER: Final = "b" * 40
ZERO: Final = "0" * 40


def route(
    tmp_path: Path, policy_status: int, push_input: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the committed shell body with a recording command-boundary fixture.

    Args:
        tmp_path: Isolated scratch directory.
        policy_status: Documented shared-policy exit code to exercise.
        push_input: Git's real four-field pre-push input shape.

    Returns:
        Shell result and every command it actually dispatched.
    """
    hook = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
    start = hook.index("# PUBLISHED TIPS ALREADY PASSED")
    end = hook.index("# The tool's own virtualenv", start)
    runner = tmp_path / "command-boundary"
    runner.write_text(
        '#!/bin/sh\nset -eu\nprintf "%s\\n" "$*" >> "$CALLS"\n'
        'if [ "$2" = push-published ]; then exit "$POLICY_STATUS"; fi\n',
        encoding="utf-8",
    )
    runner.chmod(0o755)
    calls = tmp_path / "calls"
    result = subprocess.run(
        ["sh", "-c", "set -eu\n" + hook[start:end] + '\nprintf "continued\\n"\n'],
        env={
            **os.environ,
            "HOOK_PYTHON": str(runner),
            "CALLS": str(calls),
            "POLICY_STATUS": str(policy_status),
            "PUSH_INPUT": push_input,
            "REPO": str(tmp_path),
            "ZERO": ZERO,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return result, calls.read_text(encoding="utf-8").splitlines()


def test_published_roll_dispatches_no_checks_and_continues_the_hook(tmp_path: Path) -> None:
    result, calls = route(tmp_path, 0, f"HEAD {TIP} refs/fleet/rolled {OTHER}")
    assert result.returncode == 0
    assert result.stdout == "continued\n"
    assert calls == [f"{tmp_path}/.githooks/published_maketools.py push-published {tmp_path} {TIP}"]


def test_unpublished_tip_runs_the_existing_fleet_gate(tmp_path: Path) -> None:
    result, calls = route(tmp_path, 3, f"HEAD {TIP} refs/heads/main {OTHER}")
    assert result.returncode == 0
    assert result.stdout == "continued\n"
    prefix = f"{tmp_path}/.githooks/published_maketools.py"
    assert calls == [
        f"{prefix} push-published {tmp_path} {TIP}",
        f"{prefix} session-host {tmp_path}",
        f"{prefix} pushed-fleet-check {tmp_path} {TIP} {OTHER}..{TIP}",
    ]


@pytest.mark.parametrize("status", [1, 2, 7])
def test_unknown_publication_state_refuses_before_dispatch(tmp_path: Path, status: int) -> None:
    result, calls = route(tmp_path, status, f"HEAD {TIP} refs/fleet/rolled {OTHER}")
    assert result.returncode == 1
    assert result.stdout == (
        f"=== PRE-PUSH BLOCKED: push-published exited {status}, "
        "so publication state is unknown ===\n"
    )
    assert calls == [f"{tmp_path}/.githooks/published_maketools.py push-published {tmp_path} {TIP}"]


def test_every_live_tip_reaches_the_policy_and_deletions_do_not(tmp_path: Path) -> None:
    result, calls = route(
        tmp_path,
        0,
        f"HEAD {TIP} refs/fleet/rolled {OTHER}\n"
        f"other {OTHER} refs/heads/other {ZERO}\n(delete) {ZERO} refs/heads/old {TIP}",
    )
    assert result.returncode == 0
    assert result.stdout == "continued\n"
    assert calls == [
        f"{tmp_path}/.githooks/published_maketools.py push-published {tmp_path} {TIP} {OTHER}"
    ]
