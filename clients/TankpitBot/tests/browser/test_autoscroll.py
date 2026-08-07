"""Tests for :mod:`tankpit_bot.browser.autoscroll`.

Covers the ack scanner's frame-walk edge cases and the press-verify
dance in all three shapes: setting was OFF (round trip), setting was
ON (single corrective press), and the two loud failures (no ack,
stuck ON).
"""

from __future__ import annotations

import base64

import pytest

from tankpit_bot.browser.autoscroll import (
    _read_autoscroll_ack,
    ensure_autoscroll_off,
)
from tankpit_bot.protocol.framing import FramingError
from tankpit_bot.sniffer.world_state import get_world_state, reset_world_state
from tankpit_bot.state.types import make_self_state
from tankpit_bot.types.message import CapturedMessage


def _spawn() -> None:
    """Seed the global world with a spawned self tank (in-game proof)."""
    get_world_state()["self_state"] = make_self_state(
        tank_id=1301, x=100, y=100, fuel=800, team=2, rank=1, leaderboard_position=1
    )


def _frame(body: bytes) -> str:
    """Encode one length-prefixed wire body as a base64 payload."""
    raw = bytes([len(body) & 0xFF, (len(body) >> 8) & 0xFF]) + body
    return base64.b64encode(raw).decode()


def _received(payload: str) -> CapturedMessage:
    """Build a received capture row carrying ``payload``."""
    return CapturedMessage(
        timestamp_ms=0,
        direction="received",
        payload=payload,
        ws_url="wss://test/ws/",
    )


class _FakeKeyboard:
    """Keyboard fake that appends a scripted ack frame per press."""

    def __init__(self, messages: list[CapturedMessage], acks: list[bytes | None]) -> None:
        self.messages = messages
        self.acks = acks
        self.presses: list[str] = []

    def press(self, key: str, *, delay: float | None = None) -> None:
        del delay
        self.presses.append(key)
        ack = self.acks.pop(0)
        if ack is not None:
            self.messages.append(_received(_frame(ack)))


class _FakePage:
    """Page fake exposing the keyboard + a no-op settle wait."""

    def __init__(self, keyboard: _FakeKeyboard) -> None:
        self.keyboard = keyboard
        self.waits: list[float] = []

    def wait_for_timeout(self, timeout_ms: float) -> None:
        self.waits.append(timeout_ms)


class _SpawningPage(_FakePage):
    """Page fake whose Nth settle wait "spawns" the tank on the wire."""

    def __init__(self, keyboard: _FakeKeyboard, *, spawn_after_waits: int) -> None:
        super().__init__(keyboard)
        self._spawn_after_waits = spawn_after_waits

    def wait_for_timeout(self, timeout_ms: float) -> None:
        super().wait_for_timeout(timeout_ms)
        if len(self.waits) == self._spawn_after_waits:
            _spawn()


class TestReadAutoscrollAck:
    """Frame-walk contract for the plaintext ack scanner."""

    def test_finds_the_ack_after_the_start_index(self) -> None:
        """An ``A1`` body after the press index decodes as enabled."""
        messages = [_received(_frame(b"A0")), _received(_frame(b"A1"))]

        assert _read_autoscroll_ack(messages, 1) is True

    def test_skips_sent_frames_and_non_acks(self) -> None:
        """Sent frames and binary bodies never satisfy the scan."""
        sent = CapturedMessage(
            timestamp_ms=0,
            direction="sent",
            payload=_frame(b"A0"),
            ws_url="wss://test/ws/",
        )
        messages = [sent, _received(_frame(b"\x41\x99\x01"))]

        assert _read_autoscroll_ack(messages, 0) is None

    def test_refuses_an_undecodable_payload(self) -> None:
        """A payload that is not valid base64 is fatal, not skipped."""
        messages = [_received("!!!not-base64!!!"), _received(_frame(b"A0"))]

        with pytest.raises(FramingError, match="not valid base64"):
            _read_autoscroll_ack(messages, 0)

    def test_refuses_a_truncated_frame(self) -> None:
        """A length prefix that overruns its payload is fatal.

        This asserted tolerance: the inline walk skipped a torn tail and
        read on. Measured over the whole archive, no received payload is
        ever undecodable or torn (230,323 of them across 407 sessions),
        so the tolerance only ever hid corruption
        ([[session-state-deglobalisation]]).
        """
        # Length prefix claims 200 bytes but the frame carries 2.
        messages = [_received(base64.b64encode(b"\xc8\x00AB").decode())]

        with pytest.raises(FramingError, match="Incomplete frame"):
            _read_autoscroll_ack(messages, 0)

    def test_reads_the_ack_past_a_zero_length_frame(self) -> None:
        """A zero-length frame is legal framing; the scan reads past it.

        The inline walk treated ``length == 0`` as a torn frame and
        stopped, losing every later frame in the same payload. The
        shared splitter yields the empty body and carries on
        ([[session-state-deglobalisation]]).
        """
        payload = base64.b64encode(b"\x00\x00" + b"\x02\x00A0").decode()

        assert _read_autoscroll_ack([_received(payload)], 0) is False


class TestEnsureAutoscrollOff:
    """The press-verify dance in every shape."""

    def setup_method(self) -> None:
        """Seed a spawned tank -- the toggle only works in-game."""
        reset_world_state()
        _spawn()

    def teardown_method(self) -> None:
        """Reset shared world-state globals after each test."""
        reset_world_state()

    def test_setting_was_on_single_press_corrects(self) -> None:
        """Ack ``A0`` after the first press means the toggle is fixed."""
        messages: list[CapturedMessage] = []
        keyboard = _FakeKeyboard(messages, [b"A0"])
        page = _FakePage(keyboard)

        ensure_autoscroll_off(page, messages)

        assert keyboard.presses == ["a"]
        assert page.waits == [1500.0]

    def test_setting_was_off_round_trips_back_off(self) -> None:
        """Ack ``A1`` then ``A0`` proves off -> on -> off."""
        messages: list[CapturedMessage] = []
        keyboard = _FakeKeyboard(messages, [b"A1", b"A0"])
        page = _FakePage(keyboard)

        ensure_autoscroll_off(page, messages)

        assert keyboard.presses == ["a", "a"]

    def test_missing_ack_raises(self) -> None:
        """No ack after a press is a hard failure, never a guess."""
        messages: list[CapturedMessage] = []
        keyboard = _FakeKeyboard(messages, [None])
        page = _FakePage(keyboard)

        with pytest.raises(RuntimeError, match="toggle unverified"):
            ensure_autoscroll_off(page, messages)

    def test_stuck_on_raises(self) -> None:
        """Two consecutive ``A1`` acks mean the protocol drifted."""
        messages: list[CapturedMessage] = []
        keyboard = _FakeKeyboard(messages, [b"A1", b"A1"])
        page = _FakePage(keyboard)

        with pytest.raises(RuntimeError, match="stuck ON"):
            ensure_autoscroll_off(page, messages)


def test_real_hook_delegates_to_the_module() -> None:
    """The ``_test_hooks`` seam's real implementation runs the dance."""
    from tankpit_bot._test_hooks import _real_ensure_autoscroll_off

    reset_world_state()
    _spawn()
    try:
        messages: list[CapturedMessage] = []
        keyboard = _FakeKeyboard(messages, [b"A0"])
        page = _FakePage(keyboard)

        _real_ensure_autoscroll_off(page, messages)
    finally:
        reset_world_state()

    assert keyboard.presses == ["a"]


class TestInGameGate:
    """The spawn gate in front of the first toggle press."""

    def setup_method(self) -> None:
        """Start each case from an unspawned world."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset shared world-state globals after each test."""
        reset_world_state()

    def test_waits_for_spawn_then_presses(self) -> None:
        """The dance polls until ``self_state`` appears, then toggles.

        User ruling 2026-07-29: "you cant enable or disable autoscroll
        til the bot is in the game" -- the 23:08 live firing pressed on
        the entry screen and correctly failed loud. The wait pumps the
        page loop; here the third poll "spawns" the tank.
        """
        messages: list[CapturedMessage] = []
        keyboard = _FakeKeyboard(messages, [b"A0"])
        page = _SpawningPage(keyboard, spawn_after_waits=3)

        ensure_autoscroll_off(page, messages)

        assert keyboard.presses == ["a"]
        assert len(page.waits) >= 3

    def test_never_spawning_raises_within_budget(self) -> None:
        """A tank that never spawns fails loud instead of blind-pressing."""
        messages: list[CapturedMessage] = []
        keyboard = _FakeKeyboard(messages, [b"A0"])
        page = _FakePage(keyboard)

        with pytest.raises(RuntimeError, match="never spawned"):
            ensure_autoscroll_off(page, messages)

        assert keyboard.presses == []
