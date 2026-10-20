"""Scoring a runner audit's transcript, and every way one can fail to be scorable.

The checks are a Linux host's (:func:`fleet.core.runner_audit.expected_checks`),
the richest list either platform owes; loki's Windows audit is scored by the
same reader, and :mod:`tests.test_runner_windows_host` drives that path.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import runner_audit
from fleet.core.runner_audit_types import AuditFinding
from fleet.core.runner_transcript import score_transcript
from tests._runner_fixtures import a_host


def _clean(spec: HostRunnerSpec) -> str:
    """A transcript reporting every expected check OK, built from the source.

    Args:
        spec: The host.

    Returns:
        One OK line per check, in order.
    """
    expected = runner_audit.expected_checks(spec)
    return "\n".join(f"CHECK {check['check_id']} OK" for check in expected) + "\n"


def _score(spec: HostRunnerSpec, transcript: str) -> list[AuditFinding]:
    """Score a transcript against the host's own checks.

    Args:
        spec: The host.
        transcript: The output.

    Returns:
        The findings.
    """
    return score_transcript(spec["name"], runner_audit.expected_checks(spec), transcript)


def test_a_clean_transcript_scores_every_check_ok_with_its_reason() -> None:
    spec = a_host(gpu_required=True)
    findings = _score(spec, _clean(spec))
    assert [finding["ok"] for finding in findings] == [True] * 9
    assert findings[0]["reason"] == runner_audit.MEMORY_REASON


def test_a_drift_line_carries_its_detail_and_reason() -> None:
    spec = a_host(gpu_required=True)
    lines = _clean(spec).splitlines()
    lines[4] = "CHECK gpu:wagner-austin/API:lavender DRIFT nvidia-smi on the runner PATH said: "
    drifted = _score(spec, "\n".join(lines))[4]
    assert drifted == AuditFinding(
        check_id="gpu:wagner-austin/API:lavender",
        ok=False,
        detail="nvidia-smi on the runner PATH said: ",
        reason="runner jobs on this host digest a real GPU",
    )


def test_blank_lines_are_not_checks() -> None:
    spec = a_host()
    assert len(_score(spec, "\n\n" + _clean(spec) + "\n")) == 8


@pytest.mark.parametrize("line", ["some stray output", "CHECK x", "CHECK x MAYBE detail"])
def test_a_non_check_line_is_unparsable(line: str) -> None:
    with pytest.raises(AppError) as fault:
        _score(a_host(), f"{line}\n")
    assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE
    assert fault.value.message == (
        f"the audit transcript from lavender cannot be scored: line is not a CHECK line: {line!r}"
    )


def test_a_drift_line_with_no_detail_is_unparsable() -> None:
    spec = a_host()
    lines = _clean(spec).splitlines()
    lines[0] = "CHECK memory:31gb DRIFT"
    with pytest.raises(AppError) as fault:
        _score(spec, "\n".join(lines))
    assert fault.value.message == (
        "the audit transcript from lavender cannot be scored: DRIFT line for memory:31gb "
        "carries no detail"
    )


def test_a_truncated_transcript_is_a_script_that_died_midway() -> None:
    spec = a_host()
    truncated = "\n".join(_clean(spec).splitlines()[:4])
    with pytest.raises(AppError) as fault:
        _score(spec, truncated)
    assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE
    assert fault.value.message.endswith(
        "; a transcript that stops early is a script that died midway"
    )


def test_a_reordered_transcript_is_unparsable() -> None:
    spec = a_host()
    lines = _clean(spec).splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    with pytest.raises(AppError) as fault:
        _score(spec, "\n".join(lines))
    assert fault.value.code is FleetErrorCode.RUNNER_AUDIT_UNPARSABLE
