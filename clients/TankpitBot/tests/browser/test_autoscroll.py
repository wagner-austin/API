"""Tests for :mod:`tankpit_bot.browser.autoscroll`.

Covers the ack scanner's frame-walk edge cases and the direct-send
enforcement in every shape: acked OFF, slow ack re-sent, no ack after
every send, the ``A1``-to-an-``A0``-request drift, and a failed send.
The instrument is the plaintext ``A0`` settings command over the
websocket, NOT a keypress -- hotkey maps are per-account server state
and a fresh account's default binds ``a`` to a scope pan (2026-08-13,
arterial: both presses emitted ``03 5a 06``).
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot.browser.autoscroll import (
    _read_autoscroll_ack,
    ensure_autoscroll_off,
)
from tankpit_bot.protocol.framing import FramingError, decode_frame
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_self_state
from tankpit_bot.types.message import CapturedMessage


def _spawned_world() -> WorldService:
    """Return a service whose world holds a spawned self tank.

    Returned rather than seeded into a global so the caller passes the
    SAME instance the code under test reads; seeding one service and
    handing over another tests nothing
    ([[session-state-deglobalisation]] step 8).

    Returns:
        A service carrying an in-game self state.
    """
    ws = WorldService()
    _seed_spawn(ws)
    return ws


def _seed_spawn(ws: WorldService) -> None:
    """Put a spawned self tank into ``ws``.

    Args:
        ws: Service to seed.
    """
    ws.get_world_state()["self_state"] = make_self_state(
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


class _FakeSendCDP:
    """CDP fake that answers websocket sends with scripted acks.

    Each ``A0`` send pops the next scripted entry: an ack body appended
    to the shared capture buffer as a received frame, ``None`` for a
    send the server never answers, or a status string starting with
    anything but ``SENT_`` to model a failed send.
    """

    def __init__(self, messages: list[CapturedMessage], acks: list[bytes | str | None]) -> None:
        self.messages = messages
        self.acks = acks
        self.sent_bodies: list[bytes] = []

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Handle the injected websocket-send evaluation."""
        assert method == "Runtime.evaluate"
        assert params is not None
        expression = str(params["expression"])
        match = re.search(r"atob\('([^']+)'\)", expression)
        assert match is not None, "expected a websocket send expression"
        body, remaining = decode_frame(base64.b64decode(match.group(1)))
        assert remaining == b""
        self.sent_bodies.append(body)
        ack = self.acks.pop(0)
        if isinstance(ack, str):
            return {"result": {"value": ack}}
        if ack is not None:
            self.messages.append(_received(_frame(ack)))
        return {"result": {"value": "SENT_4_BYTES via wss://test/ws/"}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Ignore event registration -- no live events in this fake."""
        del event, handler

    def detach(self) -> None:
        """Nothing to detach in this fake."""


class _FakePage:
    """Page fake exposing a no-op settle wait."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def wait_for_timeout(self, timeout_ms: float) -> None:
        self.waits.append(timeout_ms)


class _SpawningPage(_FakePage):
    """Page fake whose Nth settle wait "spawns" the tank on the wire."""

    def __init__(self, ws: WorldService, *, spawn_after_waits: int) -> None:
        super().__init__()
        self._spawn_after_waits = spawn_after_waits
        self._ws = ws

    def wait_for_timeout(self, timeout_ms: float) -> None:
        super().wait_for_timeout(timeout_ms)
        if len(self.waits) == self._spawn_after_waits:
            _seed_spawn(self._ws)


class TestReadAutoscrollAck:
    """Frame-walk contract for the plaintext ack scanner."""

    def test_finds_the_ack_after_the_start_index(self) -> None:
        """An ``A1`` body after the send index decodes as enabled."""
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

    def test_takes_the_last_ack_in_the_window(self) -> None:
        """With several acks the LATEST states the current truth."""
        messages = [_received(_frame(b"A1")), _received(_frame(b"A0"))]

        assert _read_autoscroll_ack(messages, 0) is False


class TestEnsureAutoscrollOff:
    """The direct ``A0`` send in every shape."""

    def setup_method(self) -> None:
        """Seed a spawned tank -- the command only works in-game."""
        self.ws = _spawned_world()

    def test_single_send_acked_off(self, caplog: pytest.LogCaptureFixture) -> None:
        """One ``A0`` send, one ``A0`` echo: verified OFF."""
        messages: list[CapturedMessage] = []
        cdp = _FakeSendCDP(messages, [b"A0"])
        page = _FakePage()

        with caplog.at_level(logging.INFO):
            ensure_autoscroll_off(page, cdp, messages, self.ws)

        assert cdp.sent_bodies == [b"A0"]
        assert page.waits == [1500.0]
        assert any("Autoscroll verified OFF" in r.message for r in caplog.records)

    def test_slow_ack_is_resent_and_verified(self) -> None:
        """An unanswered send re-sends; the command is absolute.

        The re-send requests the same state, so unlike the old key
        toggle there is no parity to corrupt -- two sends landing is
        exactly as OFF as one.
        """
        messages: list[CapturedMessage] = []
        cdp = _FakeSendCDP(messages, [None, b"A0"])
        page = _FakePage()

        ensure_autoscroll_off(page, cdp, messages, self.ws)

        assert cdp.sent_bodies == [b"A0", b"A0"]

    def test_no_ack_after_every_send_raises(self) -> None:
        """Three unanswered sends is a hard failure, never a guess."""
        messages: list[CapturedMessage] = []
        cdp = _FakeSendCDP(messages, [None, None, None])
        page = _FakePage()

        with pytest.raises(RuntimeError, match="setting unverified"):
            ensure_autoscroll_off(page, cdp, messages, self.ws)

        assert cdp.sent_bodies == [b"A0", b"A0", b"A0"]

    def test_enabled_ack_to_an_off_request_raises(self) -> None:
        """``A1`` echoed to an ``A0`` request means the protocol drifted."""
        messages: list[CapturedMessage] = []
        cdp = _FakeSendCDP(messages, [b"A1"])
        page = _FakePage()

        with pytest.raises(RuntimeError, match="protocol drifted"):
            ensure_autoscroll_off(page, cdp, messages, self.ws)

    def test_failed_send_raises(self) -> None:
        """A send the browser refuses fails loud immediately."""
        messages: list[CapturedMessage] = []
        cdp = _FakeSendCDP(messages, ["WEBSOCKET_NOT_OPEN: 3"])
        page = _FakePage()

        with pytest.raises(RuntimeError, match="send failed"):
            ensure_autoscroll_off(page, cdp, messages, self.ws)


def test_real_hook_delegates_to_the_module() -> None:
    """The ``_test_hooks`` seam's real implementation runs the enforcement."""
    from tankpit_bot.browser._test_hooks import _real_ensure_autoscroll_off

    ws = _spawned_world()
    messages: list[CapturedMessage] = []
    cdp = _FakeSendCDP(messages, [b"A0"])
    page = _FakePage()

    _real_ensure_autoscroll_off(page, cdp, messages, ws)

    assert cdp.sent_bodies == [b"A0"]


class TestInGameGate:
    """The spawn gate in front of the first send."""

    def test_waits_for_spawn_then_sends(self) -> None:
        """The enforcement polls until ``self_state`` appears, then sends.

        User ruling 2026-07-29: "you cant enable or disable autoscroll
        til the bot is in the game" -- the 23:08 live firing acted on
        the entry screen and correctly failed loud. The wait pumps the
        page loop; here the third poll "spawns" the tank.
        """
        ws = WorldService()
        messages: list[CapturedMessage] = []
        cdp = _FakeSendCDP(messages, [b"A0"])
        page = _SpawningPage(ws, spawn_after_waits=3)

        ensure_autoscroll_off(page, cdp, messages, ws)

        assert cdp.sent_bodies == [b"A0"]
        assert len(page.waits) >= 3

    def test_never_spawning_raises_within_budget(self) -> None:
        """A tank that never spawns fails loud instead of blind-sending."""
        ws = WorldService()
        messages: list[CapturedMessage] = []
        cdp = _FakeSendCDP(messages, [b"A0"])
        page = _FakePage()

        with pytest.raises(RuntimeError, match="never spawned"):
            ensure_autoscroll_off(page, cdp, messages, ws)

        assert cdp.sent_bodies == []
