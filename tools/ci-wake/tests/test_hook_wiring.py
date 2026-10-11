"""This repository's ``pre-push`` hands the enrolment the remote ref.

MCPs board task 24547f50. The enrolment judges a push by the workflows whose
push trigger admits the ref it UPDATES, and refuses a ref outside ``refs/``,
so the hook must bind git's third field and pass that: git's local ref for
a push of ``HEAD`` is ``HEAD``. MCPs's ``.husky/pre-push`` is pinned the
same way by its own suite (``push-authorship-wiring.test.ts``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

#: The repository root: tools/ci-wake/tests is three levels below it.
ROOT: Final = Path(__file__).resolve().parents[3]


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
