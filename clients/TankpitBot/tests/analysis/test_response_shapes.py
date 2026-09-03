"""Tests for the response-shape differ.

The differ's whole value is that a shape only ONE side produces is a
law gap, so the tests exercise both verdicts against real sessions on
disk rather than hand-built distributions: a fixture that agreed with
the tokenizer would prove nothing about the tokenizer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot.analysis.response_shapes import (
    WINDOW_MS,
    analyze_response_shapes,
    diff_shapes,
    format_response_shape_diff,
    mine_session,
    shape_token,
    tally,
)
from tankpit_bot.analysis.response_shapes_types import (
    CAUSE_COMMAND_NEVER_SENT,
    CAUSE_LIVE_SILENT_WINDOW,
    CAUSE_SHAPE_NEVER_ASSEMBLED,
    CAUSE_SIM_ONLY,
    CAUSE_TOKEN_NEVER_EMITTED,
    VERDICT_INVENTED_LAW,
    VERDICT_MISSING_LAW,
    CommandShapesDict,
    CommandWindowDict,
    ResponseShapeDiffDict,
    ShapeCountDict,
    ShapeDivergenceDict,
    decode_command_shapes,
    decode_command_window,
    decode_response_shape_diff,
    decode_shape_count,
    decode_shape_divergence,
    encode_command_shapes,
    encode_command_window,
    encode_response_shape_diff,
    encode_shape_count,
    encode_shape_divergence,
)
from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.analysis.types import ScannedSessionDict
from tankpit_bot.protocol.commands import build_move_command, build_teleport_command
from tankpit_bot.protocol.types import (
    MovementDict,
    MovementResponseDict,
    RadarResultDict,
    ShootEventDict,
    SupervisorDict,
    SyncDict,
)
from tests.analysis._capture_fixtures import (
    OWN_TANK,
    _ciphered,
    _command,
    _payload,
    _radar_result,
    _received,
    _sent,
    _session_json,
    _tank_info,
    _write,
)

OTHER_TANK = 4242


def _scan(tmp_path: Path, name: str, *messages: JSONObject) -> ScannedSessionDict:
    """Write a one-session capture and scan it, failing on a skip.

    Args:
        tmp_path: Directory to write into.
        name: Capture file name.
        *messages: Captured-message objects, in order.

    Returns:
        The decoded session.

    Raises:
        AssertionError: If the session did not decode — a fixture bug,
            never a condition under test.
    """
    text = _session_json(messages=list(messages))
    result = scan_session(_write(tmp_path, name, text))
    if result["kind"] != "scanned":
        raise AssertionError(f"fixture session did not decode: {result}")
    return result


def _shapes(windows: list[CommandWindowDict]) -> list[tuple[str, tuple[str, ...]]]:
    """Reduce mined windows to (command kind, shape) pairs."""
    return [(w["command_kind"], tuple(w["shape"])) for w in windows]


def test_self_scoped_tokens_require_the_self_id() -> None:
    """A shot, walk or position by ANOTHER tank is background."""
    mine = ShootEventDict(
        msg_type=0x53,
        team=0,
        shooter_id=OWN_TANK,
        source_x=1,
        source_y=1,
        target_x=2,
        target_y=2,
        aim_x=2,
        aim_y=2,
        weapon=0,
    )
    theirs = ShootEventDict(
        msg_type=0x53,
        team=1,
        shooter_id=OTHER_TANK,
        source_x=1,
        source_y=1,
        target_x=2,
        target_y=2,
        aim_x=2,
        aim_y=2,
        weapon=0,
    )
    assert shape_token(mine, OWN_TANK) == "53self"
    assert shape_token(theirs, OWN_TANK) is None
    # Before the session's first 0x21 the id is unknown, and guessing
    # would attribute another tank's echo to the client.
    assert shape_token(mine, None) is None


def test_movement_and_position_tokens_are_self_scoped() -> None:
    """0x47 and 0x3D tokenize only for the capturing client."""
    walk = MovementDict(
        msg_type=0x47,
        tank_id=OWN_TANK,
        start_x=1,
        start_y=1,
        direction=0,
        damage_state=0,
        lb_score=0,
        rank=1,
        flag=0,
        is_carrying=False,
        waypoints=[(2, 1)],
        path_tiles=1,
        path="e",
    )
    position = MovementResponseDict(
        msg_type=0x3D,
        team=0,
        tank_id=OWN_TANK,
        x=4,
        y=5,
        direction=0,
        damage_state=0,
        rank=1,
        lb_score=0,
        carrying=0,
    )
    assert shape_token(walk, OWN_TANK) == "47self"
    assert shape_token(walk, OTHER_TANK) is None
    assert shape_token(position, OWN_TANK) == "3Dself"
    assert shape_token(position, OTHER_TANK) is None


def test_supervisor_tokens_carry_their_code_and_both_directives() -> None:
    """A 0x52 is its code AND its two client directives.

    ``reset_action`` ("reset to idle") and ``close_map`` ("close map
    view") are things the real client DOES ([[decode-coverage]]), so a
    sim sending the right code with the wrong fields drives a real
    client wrongly while looking identical to a code-only differ. It
    did look identical for months — the out-of-window move refusal
    sent ``(1, 0)`` where all 10 archived move code-0 windows send
    ``(0, 1)``, and this function threw the evidence away.
    """
    refusal = SupervisorDict(msg_type=0x52, reset_action=1, close_map=0, error_code=5)
    assert shape_token(refusal, OWN_TANK) == "52c5r1m0"


def test_two_refusals_with_one_code_are_different_shapes() -> None:
    """The whole point: same code, different directives, different token."""
    code_zero = SupervisorDict(msg_type=0x52, reset_action=0, close_map=1, error_code=0)
    towing = SupervisorDict(msg_type=0x52, reset_action=1, close_map=1, error_code=0)

    assert shape_token(code_zero, OWN_TANK) != shape_token(towing, OWN_TANK)


def test_plain_tokens_and_background_messages() -> None:
    """Field-less families map by type; everything else is background."""
    assert shape_token(RadarResultDict(msg_type=0x46, detection_type=0, found=True), 1) == "46"
    # 0x3F Sync is real wire but not self-caused response: background.
    assert shape_token(SyncDict(msg_type=0x3F), 1) is None


def test_a_window_closes_at_the_next_command(tmp_path: Path) -> None:
    """Each token belongs to exactly one command's window."""
    session = _scan(
        tmp_path,
        "a.capture_session.json",
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_command(build_teleport_command(9, 9))), timestamp_ms=1100),
        _received(_payload(_radar_result()), timestamp_ms=1200),
        _sent(_payload(_command(build_move_command(5, 5))), timestamp_ms=1300),
        _received(_payload(_radar_result()), timestamp_ms=1400),
    )
    assert _shapes(mine_session(session)) == [("teleport", ("46",)), ("move", ("46",))]


def test_a_window_expires_on_the_wall_clock(tmp_path: Path) -> None:
    """A response past WINDOW_MS belongs to no command.

    Sim sessions compress to sub-second wall time and live deferred
    teleports bleed into the preceding command's window, so the cap is
    load-bearing in both archives.
    """
    session = _scan(
        tmp_path,
        "b.capture_session.json",
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_command(build_teleport_command(9, 9))), timestamp_ms=1100),
        _received(_payload(_radar_result()), timestamp_ms=1100 + WINDOW_MS + 1),
    )
    assert _shapes(mine_session(session)) == [("teleport", ())]


def test_lobby_sends_and_undecodable_commands_open_no_window(tmp_path: Path) -> None:
    """The lobby shares the socket; only '!' frames are commands."""
    session = _scan(
        tmp_path,
        "c.capture_session.json",
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_ciphered(bytes([0x2B, 0x01, 0x02]))), timestamp_ms=1100),
        _received(_payload(_radar_result()), timestamp_ms=1200),
    )
    assert mine_session(session) == []


def test_tally_groups_shapes_by_command_kind() -> None:
    """Identical shapes collapse into one counted row."""
    windows = [
        CommandWindowDict(command_kind="radar", shape=["46"], timestamp_ms=1),
        CommandWindowDict(command_kind="radar", shape=["46"], timestamp_ms=2),
        CommandWindowDict(command_kind="radar", shape=[], timestamp_ms=3),
    ]
    distribution = tally(windows)
    assert [entry["command_kind"] for entry in distribution] == ["radar"]
    assert distribution[0]["windows"] == 3
    assert distribution[0]["shapes"][0] == ShapeCountDict(shape=["46"], count=2)


def _distribution(kind: str, shape: list[str], count: int) -> list[CommandShapesDict]:
    """A one-row distribution, for diff tests."""
    return [
        CommandShapesDict(
            command_kind=kind,
            windows=count,
            shapes=[ShapeCountDict(shape=shape, count=count)],
        )
    ]


def test_diff_names_missing_and_invented_laws() -> None:
    """One-sided shapes are reported; shared ones are not."""
    live = _distribution("radar", ["49", "4F", "46"], 30) + _distribution("move", ["47self"], 5)
    sim = _distribution("radar", ["4F", "46"], 7) + _distribution("move", ["47self"], 5)
    rows = diff_shapes(live, sim)

    assert [(r["command_kind"], r["verdict"]) for r in rows] == [
        ("radar", VERDICT_MISSING_LAW),
        ("radar", VERDICT_INVENTED_LAW),
    ]
    assert rows[0]["live_count"] == 30
    assert rows[0]["sim_count"] == 0
    assert rows[1]["sim_count"] == 7
    # move's shape is on BOTH sides and is therefore not a divergence.
    assert [r for r in rows if r["command_kind"] == "move"] == []


def test_analyze_reads_both_archives(tmp_path: Path) -> None:
    """End to end: a live shape the sim never produces is a missing law."""
    live_dir = tmp_path / "live"
    sim_dir = tmp_path / "sim"
    live_dir.mkdir()
    sim_dir.mkdir()
    _write(
        live_dir,
        "live.capture_session.json",
        _session_json(
            messages=[
                _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
                _sent(_payload(_command(build_teleport_command(9, 9))), timestamp_ms=1100),
                _received(_payload(_radar_result()), timestamp_ms=1200),
            ]
        ),
    )
    _write(
        sim_dir,
        "sim.capture_session.json",
        _session_json(
            messages=[
                _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
                _sent(_payload(_command(build_teleport_command(9, 9))), timestamp_ms=1100),
            ]
        ),
    )

    diff = analyze_response_shapes([live_dir], [sim_dir])
    assert diff["live_sessions"] == 1
    assert diff["sim_sessions"] == 1
    assert diff["live_windows"] == 1
    assert diff["sim_windows"] == 1
    verdicts = {(row["verdict"], tuple(row["shape"])) for row in diff["divergences"]}
    assert (VERDICT_MISSING_LAW, ("46",)) in verdicts
    assert (VERDICT_INVENTED_LAW, ()) in verdicts


def test_analyze_skips_a_session_that_cannot_decode(tmp_path: Path) -> None:
    """A magic-less capture is a typed skip, not a crash or a session."""
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    _write(live_dir, "nomagic.capture_session.json", _session_json(magic=None))
    diff = analyze_response_shapes([live_dir], [live_dir])
    assert diff["live_sessions"] == 0
    assert diff["divergences"] == []


def test_a_missing_row_is_classified_by_why_it_is_one_sided() -> None:
    """THE TRIAGE. Four causes, four different pieces of work.

    Reported flat, 208 rows read as 208 sim gaps (2026-09-02). They
    were not: most were a timing artifact a tick-synchronous sim
    cannot reproduce, or commands the corpus never sent. Each row
    below is a live shape the sim lacks for a different reason, and
    the classifier has to tell them apart from the sim's own observed
    vocabulary alone.
    """
    sim = _distribution("radar", ["49", "4F", "46"], 5) + _distribution("teleport", ["5A"], 2)
    live = (
        # nothing at all came back: the live window is silent
        _distribution("radar", [], 3)
        # the sim corpus never sent a move command
        + _distribution("move", ["47self"], 4)
        # 'pickup' is a token the sim never emits for a radar
        + _distribution("radar", ["4F", "46", "pickup"], 6)
        # every token here IS in the sim's radar vocabulary
        + _distribution("radar", ["4F", "46"], 7)
    )

    causes = {
        (row["command_kind"], tuple(row["shape"])): row["cause"] for row in diff_shapes(live, sim)
    }
    assert causes[("radar", ())] == CAUSE_LIVE_SILENT_WINDOW
    assert causes[("move", ("47self",))] == CAUSE_COMMAND_NEVER_SENT
    assert causes[("radar", ("4F", "46", "pickup"))] == CAUSE_TOKEN_NEVER_EMITTED
    assert causes[("radar", ("4F", "46"))] == CAUSE_SHAPE_NEVER_ASSEMBLED


def test_an_invented_row_carries_no_missing_side_cause() -> None:
    """The triage answers "why is this live shape absent from the sim".

    An invented row is the opposite question, so it is marked as
    out of scope rather than given a cause that would read as one.
    """
    rows = diff_shapes([], _distribution("shoot", ["53self", "49"], 4))

    assert [row["cause"] for row in rows] == [CAUSE_SIM_ONLY]


def test_the_report_splits_the_missing_side_and_counts_each_bucket() -> None:
    """The reader is shown which rows imply which work, and how many."""
    sim = _distribution("radar", ["49", "4F", "46"], 5)
    live = _distribution("radar", [], 3) + _distribution("move", ["47self"], 4)

    text = format_response_shape_diff(_diff_of(live, sim), limit=5)

    assert "[timing, not a gap]" in text
    assert "[corpus gap]" in text
    assert "1 rows, 3 live windows behind them" in text
    assert "1 rows, 4 live windows behind them" in text
    # A bucket with no rows is omitted rather than printed as zero.
    assert "[READ THESE FIRST]" not in text


def test_the_invented_side_leads_the_report() -> None:
    """It is the half that means something at any sample size.

    A shape the sim produced is one it CAN produce however small the
    corpus; a shape it did not produce may only be undersampling.
    """
    sim = _distribution("shoot", ["53self", "49"], 4)
    live = _distribution("move", ["47self"], 4)

    text = format_response_shape_diff(_diff_of(live, sim), limit=5)

    assert text.index("INVENTED LAWS") < text.index("MISSING LAWS")


def test_format_reports_both_verdicts_and_names_what_it_dropped() -> None:
    """A capped report says how many rows it did not show."""
    live = _distribution("radar", ["49"], 9) + _distribution("move", ["47self"], 8)
    sim = _distribution("shoot", ["53self", "49"], 4)
    diff = _diff_of(live, sim)

    text = format_response_shape_diff(diff, limit=1)
    assert "MISSING LAWS (real server does it, sim never does): 2" in text
    assert "INVENTED LAWS (sim does it, archive never shows it): 1" in text
    assert "... 1 more rows not shown (limit=1)" in text


def test_format_renders_a_silent_window_readably() -> None:
    """An empty shape prints as (silent), not as nothing."""
    diff = _diff_of(_distribution("map_open", [], 3), [])
    assert "(silent)" in format_response_shape_diff(diff, limit=5)


def _diff_of(live: list[CommandShapesDict], sim: list[CommandShapesDict]) -> ResponseShapeDiffDict:
    """Build a diff result around two distributions, for format tests.

    Args:
        live: The live distribution.
        sim: The sim distribution.

    Returns:
        A diff carrying the divergences those distributions imply.
    """
    return ResponseShapeDiffDict(
        live_sessions=1,
        sim_sessions=1,
        live_windows=sum(entry["windows"] for entry in live),
        sim_windows=sum(entry["windows"] for entry in sim),
        divergences=diff_shapes(live, sim),
    )


def test_command_window_round_trips() -> None:
    """Encode then decode returns an equal window."""
    original = CommandWindowDict(command_kind="radar", shape=["49", "4F"], timestamp_ms=7)
    assert decode_command_window(encode_command_window(original)) == original


def test_shape_count_round_trips() -> None:
    """Encode then decode returns an equal tally."""
    original = ShapeCountDict(shape=["47self"], count=3)
    assert decode_shape_count(encode_shape_count(original)) == original


def test_command_shapes_round_trips() -> None:
    """The nested shape list survives the round trip."""
    original = CommandShapesDict(
        command_kind="move",
        windows=3,
        shapes=[ShapeCountDict(shape=["47self"], count=3)],
    )
    assert decode_command_shapes(encode_command_shapes(original)) == original


def test_shape_divergence_round_trips() -> None:
    """Verdict and both counts survive the round trip."""
    original = ShapeDivergenceDict(
        command_kind="radar",
        shape=["49", "4F", "46"],
        live_count=3403,
        sim_count=0,
        verdict=VERDICT_MISSING_LAW,
        cause=CAUSE_SHAPE_NEVER_ASSEMBLED,
    )
    assert decode_shape_divergence(encode_shape_divergence(original)) == original


def test_response_shape_diff_round_trips() -> None:
    """The whole comparison survives the round trip."""
    original = analyze_response_shapes([], [])
    assert decode_response_shape_diff(encode_response_shape_diff(original)) == original


def test_decode_rejects_a_shape_token_that_is_not_a_string() -> None:
    """A malformed artifact fails decode rather than coercing."""
    with pytest.raises(TypeError):
        decode_command_window({"command_kind": "radar", "shape": [42], "timestamp_ms": 1})


def test_a_plaintext_ack_is_never_tokenized(tmp_path: Path) -> None:
    """The toggle acks echo UN-XORed and must not reach the cipher.

    ``A1``/``C1`` are raw two-byte bodies whose leading letters are
    also binary families (0x41 Deactivation, 0x43 CacheUpdate), so the
    discrimination happens before any XOR decode.
    """
    session = _scan(
        tmp_path,
        "ack.capture_session.json",
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_command(build_teleport_command(9, 9))), timestamp_ms=1100),
        _received(_payload(b"A1"), timestamp_ms=1200),
    )
    assert _shapes(mine_session(session)) == [("teleport", ())]


def test_a_text_route_frame_is_never_tokenized(tmp_path: Path) -> None:
    """0x3D is dual-use, so lobby text must not become a 3Dself.

    Decoding a long 0x2B/0x3D text body as binary would manufacture a
    self-position token out of lobby chatter — the reason the
    pre-cipher discriminators run at all.
    """
    session = _scan(
        tmp_path,
        "text.capture_session.json",
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_command(build_teleport_command(9, 9))), timestamp_ms=1100),
        _received(_payload(b"+1|Practice|1|0,0,0|2|p|field01.gif|2026"), timestamp_ms=1200),
    )
    assert _shapes(mine_session(session)) == [("teleport", ())]


def test_a_malformed_known_frame_contributes_no_token(tmp_path: Path) -> None:
    """A short 0x41 is archive noise: no token, no crash.

    101 of the archive's 262,588 received frames are short 0x41
    Deactivations ([[recipient-policy]]).
    """
    session = _scan(
        tmp_path,
        "short.capture_session.json",
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_command(build_teleport_command(9, 9))), timestamp_ms=1100),
        _received(_payload(_ciphered(bytes([0x41, 0x99]))), timestamp_ms=1200),
    )
    assert _shapes(mine_session(session)) == [("teleport", ())]


def test_an_undecodable_command_frame_opens_no_window(tmp_path: Path) -> None:
    """A '!' frame whose body will not decode is not attributed."""
    session = _scan(
        tmp_path,
        "badcmd.capture_session.json",
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_ciphered(bytes([0x21]))), timestamp_ms=1100),
        _received(_payload(_radar_result()), timestamp_ms=1200),
    )
    assert mine_session(session) == []
