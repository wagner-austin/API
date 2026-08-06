"""Fight report CLI and the audit's human-episode findings."""

from __future__ import annotations

import datetime
from pathlib import Path

from platform_core.json_utils import dump_json_str
from tests.validate.builders import (
    aimed_shot_message,
    deactivation_message,
    identity_message,
    make_session,
    named_identity_message,
    sync_message,
)

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.capture_audit import _human_episode_findings
from tankpit_bot.diagnostics.fight_report import main, window_bounds_ms
from tankpit_bot.types import CaptureSession, encode_capture_session

_SELF = 1301
_HUMAN = 2678
_T0 = 1_785_000_000_000


def _turret_duel_session() -> CaptureSession:
    """Self trades four same-tile shots against a firing human."""
    messages = [
        identity_message(_T0, _SELF),
        named_identity_message(_T0 + 1, _HUMAN, "nope"),
        sync_message(_T0 + 3, _SELF, 1, 1100),
    ]
    for tick in range(4):
        at = _T0 + 2_000 * (tick + 1)
        messages.append(aimed_shot_message(at, _HUMAN, (200, 142), (201, 143), 0))
        messages.append(aimed_shot_message(at + 100, _SELF, (201, 143), (200, 142), 1))
    # The human's closing shot keeps all four same-tile replies inside
    # the episode window (the window is the human's first-to-last shot).
    messages.append(aimed_shot_message(_T0 + 10_000, _HUMAN, (200, 142), (201, 143), 1))
    messages.append(deactivation_message(_T0 + 10_500, _SELF, _HUMAN))
    return make_session(messages, start_timestamp_ms=_T0)


class TestHumanEpisodeFindings:
    def test_turret_duel_yields_episode_and_warning(self) -> None:
        """The episode INFO and the turret WARNING both surface."""
        findings = _human_episode_findings(_turret_duel_session())
        checks = [finding["check"] for finding in findings]
        assert checks == ["human_episode", "turret_exchange"]
        episode = findings[0]
        assert episode["severity"] == "info"
        assert "nope" in episode["summary"]
        assert "1 death(s)" in episode["summary"]
        turret = findings[1]
        assert turret["severity"] == "warning"
        assert turret["evidence"]["max_stationary_streak"] == 4

    def test_short_trade_yields_no_turret_warning(self) -> None:
        """Two shots each from moving tiles: episode only."""
        messages = [
            identity_message(_T0, _SELF),
            named_identity_message(_T0 + 1, _HUMAN, "nope"),
            aimed_shot_message(_T0 + 2_000, _HUMAN, (200, 142), (201, 143), 0),
            aimed_shot_message(_T0 + 2_100, _SELF, (201, 143), (200, 142), 1),
            aimed_shot_message(_T0 + 4_000, _HUMAN, (199, 141), (202, 144), 0),
            aimed_shot_message(_T0 + 4_100, _SELF, (202, 144), (199, 141), 1),
        ]
        findings = _human_episode_findings(make_session(messages))
        assert [finding["check"] for finding in findings] == ["human_episode"]


class TestFightReportCli:
    def test_cli_renders_episodes_and_window_rows(self, tmp_path: Path) -> None:
        """The CLI reads the capture through the hooks and exits 0."""
        capture_path = tmp_path / "duel.capture_session.json"
        payload = dump_json_str(encode_capture_session(_turret_duel_session()))
        start = datetime.datetime.fromtimestamp(_T0 / 1000).strftime("%H:%M:%S")
        end = datetime.datetime.fromtimestamp((_T0 + 11_000) / 1000).strftime("%H:%M:%S")
        original_argv = _test_hooks.get_argv
        original_read = _test_hooks.read_text

        def _read(path: Path) -> str:
            if path == capture_path:
                return payload
            return original_read(path)

        _test_hooks.get_argv = lambda: ["tankpit-fight", str(capture_path), start, end]
        _test_hooks.read_text = _read
        try:
            exit_code = main()
        finally:
            _test_hooks.get_argv = original_argv
            _test_hooks.read_text = original_read
        assert exit_code == 0

    def test_cli_without_args_uses_the_latest_capture(self) -> None:
        """No argv: the default capture path is read."""
        payload = dump_json_str(encode_capture_session(_turret_duel_session()))
        default_path = Path("runs/bot/latest.capture_session.json")
        seen: list[Path] = []
        original_argv = _test_hooks.get_argv
        original_read = _test_hooks.read_text

        def _read(path: Path) -> str:
            if path == default_path:
                seen.append(path)
                return payload
            return original_read(path)

        _test_hooks.get_argv = lambda: ["tankpit-fight"]
        _test_hooks.read_text = _read
        try:
            exit_code = main()
        finally:
            _test_hooks.get_argv = original_argv
            _test_hooks.read_text = original_read
        assert exit_code == 0
        assert seen == [default_path]


class TestWindowBounds:
    def test_anchor_round_trips_and_duration_holds(self) -> None:
        """Passing the anchor's own wall time returns the anchor
        (whole-second), and the window duration is exact."""
        anchor_hms = datetime.datetime.fromtimestamp(_T0 / 1000).strftime("%H:%M:%S")
        start_ms, end_ms = window_bounds_ms(_T0 + 250, anchor_hms, anchor_hms)
        assert start_ms == _T0
        assert end_ms == start_ms

    def test_offset_window_is_relative_to_the_anchor(self) -> None:
        """A window N seconds past the anchor lands N seconds later."""
        later = datetime.datetime.fromtimestamp((_T0 + 9_000) / 1000).strftime("%H:%M:%S")
        anchor_hms = datetime.datetime.fromtimestamp(_T0 / 1000).strftime("%H:%M:%S")
        start_ms, end_ms = window_bounds_ms(_T0, anchor_hms, later)
        assert (start_ms, end_ms) == (_T0, _T0 + 9_000)
