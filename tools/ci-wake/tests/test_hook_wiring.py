"""This repository's ``pre-push`` hands the enrolment the remote ref.

MCPs board task 24547f50. The enrolment judges a push by the workflows whose
push trigger admits the ref it UPDATES, and refuses a ref outside ``refs/``,
so the hook must bind git's third field and pass that: git's local ref for
a push of ``HEAD`` is ``HEAD``. MCPs's ``.husky/pre-push`` is pinned the
same way by its own suite (``push-authorship-wiring.test.ts``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

#: The repository root: tools/ci-wake/tests is three levels below it.
ROOT: Final = Path(__file__).resolve().parents[3]

#: The first line of the hook's enrolment loop.
LOOP_START: Final = "printf '%s\\n' \"$PUSH_INPUT\" | while read -r _local_ref local_sha"

#: git's all-zero sha, as the hook names it.
ZERO: Final = "0" * 40


def test_every_checkout_enrols_into_the_ledger_the_pump_reads() -> None:
    """MCPs board task 88a2853f: the pump reads the hub checkout's ledger
    alone, so the tree is the hub's whichever checkout pushes, resolved with
    MCPs's own variable and default, and nothing names this checkout's."""
    hook = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
    executable = [line for line in hook.splitlines() if not line.lstrip().startswith("#")]

    assert 'TOOL="${CI_WAKE_HOME:-$HOME/PROJECTS/API/tools/ci-wake}"' in executable
    assert 'LEDGER="$TOOL/runs/pushes.jsonl"' in executable
    assert not any("$REPO/tools/ci-wake" in line for line in executable)


def test_the_enrolment_is_handed_the_ref_the_push_updates() -> None:
    hook = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
    loop = hook.index("while read -r _local_ref local_sha remote_ref _remote_sha; do")
    enrolment = hook.index("-m ci_wake.cli.enrol")
    call = hook[enrolment : hook.index("< /dev/null", enrolment)]

    assert loop < enrolment
    assert '--ref "$remote_ref"' in call
    assert "local_ref" not in call


def enrolments(tmp_path: Path, push_input: str) -> list[str]:
    """Run the hook's own enrolment loop under sh, with an interpreter that records its calls.

    Args:
        tmp_path: The test's directory.
        push_input: What git hands the hook on standard input.

    Returns:
        Each enrolment's arguments, one line per call.
    """
    hook = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
    start = hook.index(LOOP_START)
    loop = hook[start : hook.index("\ndone\n", start) + len("\ndone\n")]
    recorder = tmp_path / "python"
    calls = tmp_path / "calls.log"
    recorder.write_text(f'#!/bin/sh\necho "$*" >> {calls}\n', encoding="utf-8")
    recorder.chmod(0o755)
    environment = {
        "PATH": "/usr/bin:/bin",
        "PUSH_INPUT": push_input,
        "ZERO": ZERO,
        "PYTHON": str(recorder),
        "LEDGER": "pushes.jsonl",
        "NWO": "wagner-austin/API",
    }
    subprocess.run(("sh", "-c", loop), env=environment, check=True)
    return calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []


def test_a_push_of_nothing_enrols_nothing(tmp_path: Path) -> None:
    """MCPs board task b4005697: git runs the hook with empty input when every
    ref is up to date, and that empty line once reached the enrolment as
    --ref '', which refused it and failed the push."""
    assert enrolments(tmp_path, "") == []


def test_each_pushed_ref_enrols_once_and_a_deletion_none(tmp_path: Path) -> None:
    pushed = "HEAD " + "a" * 40 + " refs/heads/main " + "b" * 40
    deleted = "(delete) " + ZERO + " refs/heads/old " + "c" * 40

    assert enrolments(tmp_path, f"{pushed}\n{deleted}") == [
        "-m ci_wake.cli.enrol --enrolment pushes.jsonl --repo wagner-austin/API "
        f"--sha {'a' * 40} --ref refs/heads/main"
    ]
