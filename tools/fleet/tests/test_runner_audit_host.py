"""The runner audit's host rows read on the runner host itself (MCPs board task faec98a5).

``fleet-runners --spec runners.json --host <host>`` sends
:func:`~fleet.core.runner_audit.render_audit_script` to the host over ssh,
runs it with ``sudo -n`` and scores its transcript, but it runs from a
session, and nothing the board can run again recorded what it read. On
2026-10-30 its only two DRIFT rows on lavender were the manual asset
``/opt/corvis/rw-game/game-lib.jar`` and its pin, a file nothing had read
since RustedWarfareBot's CI left the host and no machine held, until the
roster dropped it. These cases run the same script with the same rights on
the host ``tools/fleet-execution-runners`` is pinned to and score it with
the same scorer, so a row the roster and the host disagree on fails a run
the board can repeat. The dropped-jobs row and the registrations rows read
GitHub and the hub, not this host, and stay with the command.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pytest

from fleet.contracts.runners import FileAsset, HostRunnerSpec
from fleet.core.remote import ROOT_BASH
from fleet.core.runner_audit import AUDIT_TIMEOUT_SECONDS, expected_checks, render_audit_script
from fleet.core.runner_audit_types import AuditFinding
from fleet.core.runner_transcript import score_transcript
from tests._host import READING_PROPERTY, command_output

#: The asset the roster dropped, declared as it was, which no host holds.
DROPPED_ASSET: Final = FileAsset(
    path="/opt/corvis/rw-game/game-lib.jar",
    sha256="8a550a37e2d8a5430866090d4e7d5892f9010b47f52a5a09350fc66c620deec9",
    writable=False,
    reason="RustedWarfareBot provenance fingerprints digest the real licensed engine jar",
    manual=True,
    provision_command=None,
)


def _audit_here(spec: HostRunnerSpec) -> list[AuditFinding]:
    """Run a host's audit script on this host as the command runs it there, and score it.

    Args:
        spec: The roster entry whose script runs.

    Returns:
        One finding per check the script owes.
    """
    output = command_output(
        [*ROOT_BASH, "-c", render_audit_script(spec)], timeout=AUDIT_TIMEOUT_SECONDS
    )
    return score_transcript(spec["name"], expected_checks(spec), output)


def _drifted(findings: list[AuditFinding]) -> list[str]:
    return [f"{each['check_id']}: {each['detail']}" for each in findings if not each["ok"]]


@pytest.mark.host_runner_units
def test_the_host_reads_no_drift_against_its_roster_entry(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    findings = _audit_here(runner_host)
    assert _drifted(findings) == []
    checked = [each["check_id"] for each in findings]
    assert [check_id for check_id in checked if "/opt/corvis/rw-game" in check_id] == []
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: the runner audit's {len(findings)} host row(s) read OK, "
        f"0 DRIFT, against {len(runner_host['assets'])} declared asset(s)",
    )


@pytest.mark.host_runner_units
def test_an_asset_the_host_lacks_reads_as_drift(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    lacking = runner_host.copy()
    lacking["assets"] = [*runner_host["assets"], DROPPED_ASSET]
    drifted = [each["check_id"] for each in _audit_here(lacking) if not each["ok"]]
    assert drifted == [f"asset:{DROPPED_ASSET['path']}", f"sha256:{DROPPED_ASSET['path']}"]
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: with the dropped {DROPPED_ASSET['path']} declared again, "
        f"the audit reads exactly {len(drifted)} DRIFT row(s), {', '.join(drifted)}",
    )
