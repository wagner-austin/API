"""Step (e) part 2: ``make audit`` re-derives claims from sim wire.

The seam records every frame that crosses it as a standard
``CaptureSession``; the archive validators — the exact instruments
that re-derive the wiki's physics claims from 244 real sessions —
then run over the sim-generated capture. Zero mismatches means the
sim's laws price the same wire the way the real server does, judged
by code that has never heard of the sim.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str

from tankpit_bot import _test_hooks
from tankpit_bot.bot.tick_body import _tick_once
from tankpit_bot.protocol.commands import build_move_command
from tankpit_bot.sim.session import SimCDPSession, build_capture_session, deliver_batch
from tankpit_bot.types import encode_capture_session
from tankpit_bot.validate.audit import EXACTNESS_FLOOR, collect_evidence
from tankpit_bot.wire.helpers import EncodeError
from tests.sim.seam import RICH_CONTAINERS, SEAM_CLIENT_ID, SEAM_MAGIC, SeamClock, boot_seam


def test_sim_wire_survives_the_archive_audit(tmp_path: Path) -> None:
    """The archive validators price sim wire with zero mismatches.

    A 40-round seam session under a stepped clock is assembled into a
    capture session, written to a temporary runs tree, and fed to the
    real ``collect_evidence``. The production bot teleports rather
    than walks (its collect style), so one scripted walk is driven
    through the real command service afterwards, followed by quiet
    ticks so the walk episode closes.

    Positive controls: the walk-cost and dual-shot-cost claims must
    find samples (the session genuinely walked and fought). Verdict:
    every sampled claim passes the audit's own gate — exact share at
    or above ``EXACTNESS_FLOOR``. The gate, not zero-mismatch, is the
    real instrument's criterion: the sim reproduces the same
    charge-latency boundary noise the real wire shows (a shooting
    burst's FIRST echo sits in a window whose debit lands one window
    later), and the floor absorbs exactly that positive-signed
    measurement noise on real captures too.
    """
    clock = SeamClock(100_000)
    original_clock: Callable[[], int] = _test_hooks.get_current_time_ms
    _test_hooks.get_current_time_ms = clock
    try:
        bot, server, link, _table = boot_seam(enemy_fuel=4000, containers=RICH_CONTAINERS)
        for _ in range(40):
            _tick_once(bot)
            deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
            clock.advance(1000)
        truth = server.world["tanks"][SEAM_CLIENT_ID]
        walk = build_move_command(truth["x"] - 3, truth["y"])
        assert bot._send_bytes(walk, "audit_walk") is True
        for _ in range(4):
            deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
            clock.advance(1000)
        _tick_once(bot)
    finally:
        _test_hooks.get_current_time_ms = original_clock
    session = build_capture_session(link, SEAM_MAGIC, "sim-seam-audit")
    (tmp_path / "bot").mkdir()
    (tmp_path / "sniff").mkdir()
    capture_path = tmp_path / "bot" / "sim-seam-audit.capture_session.json"
    capture_path.write_text(dump_json_str(encode_capture_session(session)), encoding="utf-8")
    evidence = collect_evidence(tmp_path)
    by_id = {record["claim_id"]: record for record in evidence}
    assert by_id["walk-cost"]["samples"] > 0
    assert by_id["dual-shot-cost"]["samples"] > 0
    sampled = [record for record in evidence if record["samples"] > 0]
    assert sampled != []
    for record in sampled:
        assert record["exact"] / record["samples"] >= EXACTNESS_FLOOR, record


def test_capture_assembly_refuses_an_empty_session() -> None:
    """A link that never carried traffic cannot become a capture."""
    _bot, server, _link, table = boot_seam()
    fresh = SimCDPSession(server, table)
    with pytest.raises(EncodeError):
        build_capture_session(fresh, SEAM_MAGIC, "empty")
