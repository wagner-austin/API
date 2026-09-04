"""Tests for :mod:`tankpit_bot.browser.flag_capture`.

Covers the ensure lifecycle (binding registered once per CDP session,
re-registered on a fresh session), the lead-up ring's size cap, and the
binding-event path: a click becomes a ``human_flag`` diagnostic on the
events JSONL carrying the click metadata and the ring snapshot; foreign
bindings are ignored.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
    narrow_json_to_str,
)

from tankpit_bot.browser.flag_capture import (
    FLAG_BINDING_NAME,
    FLAG_RING_SIZE,
    FlagCaptureService,
)
from tankpit_bot.browser.overlay import OverlayStateDict, encode_overlay_state
from tankpit_bot.runtime_logging import configure_bot_runtime_logging
from tests.browser.test_overlay import make_overlay
from tests.conftest import FakeFileSystem


class _RecordingCDP:
    """CDP-session fake that records sends and handler registrations."""

    def __init__(self) -> None:
        """Initialize with empty send/registration logs."""
        self.sent: list[tuple[str, JSONObject | None]] = []
        self.handlers: dict[str, Callable[[JSONObject], None]] = {}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Record one CDP send.

        Args:
            method: CDP method name.
            params: CDP call parameters.

        Returns:
            Empty CDP-style result.
        """
        self.sent.append((method, params))
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Record one handler registration.

        Args:
            event: CDP event name.
            handler: Registered callback.
        """
        self.handlers[event] = handler

    def detach(self) -> None:
        """Unused protocol member."""
        raise AssertionError("the flag service never detaches the session")


def _click_params(flag_seq: int, clicked_at_ms: int) -> JSONObject:
    """Build one flag-click ``Runtime.bindingCalled`` event.

    Args:
        flag_seq: Click sequence number the HUD assigned.
        clicked_at_ms: Click wall-clock timestamp.

    Returns:
        CDP event parameters.
    """
    return {
        "name": FLAG_BINDING_NAME,
        "payload": dump_json_str({"flag_seq": flag_seq, "clicked_at_ms": clicked_at_ms}),
    }


class TestEnsure:
    """Binding registration lifecycle."""

    def test_first_ensure_registers_binding_once(self) -> None:
        """The first ensure registers the handler and adds the binding."""
        service = FlagCaptureService()
        cdp = _RecordingCDP()

        service.ensure(cdp)
        service.ensure(cdp)

        assert cdp.sent == [("Runtime.addBinding", {"name": FLAG_BINDING_NAME})]
        assert list(cdp.handlers) == ["Runtime.bindingCalled"]

    def test_fresh_session_re_registers(self) -> None:
        """A new CDP session after a restart gets its own registration."""
        service = FlagCaptureService()
        first = _RecordingCDP()
        second = _RecordingCDP()

        service.ensure(first)
        service.ensure(second)

        assert second.sent == [("Runtime.addBinding", {"name": FLAG_BINDING_NAME})]


class TestRecordTick:
    """Lead-up ring behavior."""

    def test_ring_keeps_only_the_newest_payloads(self, fake_fs: FakeFileSystem) -> None:
        """The ring caps at FLAG_RING_SIZE, dropping the oldest."""
        artifacts = configure_bot_runtime_logging("20260729-000000")
        service = FlagCaptureService()
        cdp = _RecordingCDP()
        service.ensure(cdp)
        for fuel in range(FLAG_RING_SIZE + 3):
            service.record_tick(OverlayStateDict(**{**make_overlay(), "fuel": fuel}))

        cdp.handlers["Runtime.bindingCalled"](_click_params(1, 1785388629830))

        files = fake_fs.get_written_files()
        event = narrow_json_to_dict(
            load_json_str(files[artifacts["latest_events_path"]].strip().splitlines()[-1])
        )
        recent = narrow_json_to_list(load_json_str(narrow_json_to_str(event["recent_ticks"])))
        fuels = [narrow_json_to_dict(tick)["fuel"] for tick in recent]

        assert len(recent) == FLAG_RING_SIZE
        assert fuels == list(range(3, FLAG_RING_SIZE + 3))


class TestBindingCalled:
    """Click-to-diagnostic relay."""

    def test_click_emits_human_flag_with_ring_snapshot(self, fake_fs: FakeFileSystem) -> None:
        """One click lands one ``human_flag`` event with full context."""
        artifacts = configure_bot_runtime_logging("20260729-000000")
        service = FlagCaptureService()
        cdp = _RecordingCDP()
        service.ensure(cdp)
        overlay = make_overlay()
        service.record_tick(overlay)

        cdp.handlers["Runtime.bindingCalled"](_click_params(2, 1785388629830))

        files = fake_fs.get_written_files()
        event = narrow_json_to_dict(
            load_json_str(files[artifacts["latest_events_path"]].strip().splitlines()[-1])
        )
        assert event["channel"] == "DIAGNOSTIC"
        assert event["diagnostic_kind"] == "human_flag"
        assert event["flag_seq"] == 2
        assert event["clicked_at_ms"] == 1785388629830
        recent = narrow_json_to_list(load_json_str(narrow_json_to_str(event["recent_ticks"])))
        assert [narrow_json_to_dict(tick) for tick in recent] == [encode_overlay_state(overlay)]

    def test_foreign_binding_is_ignored(self, fake_fs: FakeFileSystem) -> None:
        """A different binding's event emits nothing."""
        artifacts = configure_bot_runtime_logging("20260729-000000")
        service = FlagCaptureService()
        cdp = _RecordingCDP()
        service.ensure(cdp)

        cdp.handlers["Runtime.bindingCalled"](
            {"name": "__botCastDeliver", "payload": "data:image/jpeg;base64,"}
        )

        files = fake_fs.get_written_files()
        assert "human_flag" not in files[artifacts["latest_events_path"]]
