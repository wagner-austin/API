"""Cross-analyzer consistency: the digest and the issue report must agree.

Both post-run analyzers now count kills from the same signal --
killer-attributed ``tank_deactivated`` diagnostics (unified
2026-08-28, when the digest's free-text ``kill registered`` count was
caught missing coordinate-aimed kills: arterial 2026-08-26 had 44
wire kills but 43 registered lines). Before that, for six days
(2026-08-14 to 2026-08-20) the two disagreed on the same artifact
(digest 0, scorecard 2 for the wedged gatherer run) because the
scorecard's copy of the kill law missed the fleet attribution split,
and nothing compared them. This suite is the instrument that
comparison was missing: one stream shaped exactly like the live
emitters produce, both analyzers over it, shared facts diffed.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import FakeFileSystem
from tests.diagnostics._issue_report_fixtures import _emit_session_room

from tankpit_bot.diagnostics.issue_report import build_issue_report
from tankpit_bot.diagnostics.run_digest import build_run_digest
from tankpit_bot.runtime_logging import (
    configure_probe_runtime_logging,
    emit_ai,
    emit_diagnostic,
)


def test_digest_and_scorecard_agree_on_kills_in_a_fleet_room(
    fake_fs: FakeFileSystem,
) -> None:
    """Both analyzers count exactly the own-attributed kills.

    The stream mirrors the live emitters tick-for-tick: an own kill
    produces BOTH the tick layer's ``kill registered`` AI line
    (``tick_combat_feedback``) and a ``tank_deactivated`` diagnostic
    naming our tank as killer; a fleet sibling's kill produces only
    the diagnostic, with the sibling as killer. Any future change
    that moves one counter without the other breaks this pin instead
    of shipping a silent disagreement.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(diagnostic_kind="tank_identity", tank_id=601, name="artax")
    # A locked-target own kill: the wire's 0x41 names us AND the tick
    # layer logs its free-text line.
    emit_ai("kill registered (tank_id=%d)", 529)
    emit_diagnostic(diagnostic_kind="tank_deactivated", victim_id=529, killer_id=601)
    # A coordinate-aimed own kill: the 0x41 names us but the tick
    # layer never logs "kill registered" (arterial 2026-08-26, victim
    # 569 / orange-7). Both analyzers must still count it.
    emit_diagnostic(diagnostic_kind="tank_deactivated", victim_id=569, killer_id=601)
    # A fleet sibling's kill in the same room: diagnostic only, killer
    # is the sibling.
    emit_diagnostic(diagnostic_kind="tank_deactivated", victim_id=530, killer_id=1301)

    events_path = Path(artifacts["latest_events_path"])
    digest = build_run_digest(events_path)
    report = build_issue_report(events_path)

    assert digest["kills"] == 2
    assert report["scorecard"]["kills"] == 2
    assert digest["kills"] == report["scorecard"]["kills"]
    assert digest["self_tank_id"] == 601
