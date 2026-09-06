"""Tests for the client-command coverage audit.

The audit exists because three unmapped command bytes crashed the sim
on 2026-09-03 and each was found by a hand-written sweep. Its whole
value is catching the FOURTH automatically, so the tests drive real
capture sessions through the real decoder rather than hand-built rows:
a fixture that agreed with the classifier would prove nothing about
the classifier.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot.analysis.command_coverage import (
    analyze_command_coverage,
    crashing_rows,
    format_command_coverage,
)
from tankpit_bot.analysis.command_coverage_types import (
    STATUS_HANDLED,
    CommandByteRowDict,
    CommandCoverageDict,
    decode_command_byte_row,
    decode_command_coverage,
    encode_command_byte_row,
    encode_command_coverage,
)
from tankpit_bot.protocol.command_builders import (
    build_deposit_fuel_command,
    build_query_command,
)
from tankpit_bot.protocol.commands import (
    CMD_DEPOSIT_FUEL,
    CMD_KEEPALIVE,
    CMD_RADAR,
)
from tankpit_bot.protocol.commands import COMMAND_PREFIX as _PREFIX
from tests.analysis._capture_fixtures import (
    OWN_TANK,
    _ciphered,
    _command,
    _payload,
    _received,
    _sent,
    _session_json,
    _tank_info,
    _write,
)

#: A command byte no constant names and the decoder cannot map — the
#: shape of every crash this audit exists to catch.
_UNKNOWN_BYTE = 0xFE


def _client_command(command: int) -> bytes:
    """One client command frame, built by the PRODUCTION builder.

    A hand-rolled frame here would let the fixture and the decoder
    drift apart and agree on the wrong bytes, so every frame comes
    from the real builder — including for the bytes no constant names,
    which ride the bare ``[len][!][type][cmd]`` query shape.

    The deposit is the exception that has to be spelled out: its
    decoder requires six payload bytes, so a bare query frame for it
    would be discarded as truncated noise and the audit would report
    no row at all.

    Args:
        command: The command byte.

    Returns:
        Framed command bytes, length header included.
    """
    if command == CMD_DEPOSIT_FUEL:
        return build_deposit_fuel_command(174, 49, 294)
    return build_query_command(command)


def _archive(tmp_path: Path, commands: tuple[int, ...], name: str = "archive") -> Path:
    """Write a one-session archive carrying the given command bytes.

    Args:
        tmp_path: Test temp directory.
        commands: Command bytes the session sends.
        name: Sub-directory name, so a test comparing two archives
            gives each its own rather than sharing one and mixing
            their sessions.

    Returns:
        The archive directory.
    """
    directory = tmp_path / name
    directory.mkdir()
    messages = [_received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000)]
    for index, command in enumerate(commands):
        messages.append(
            _sent(_payload(_command(_client_command(command))), timestamp_ms=1100 + index * 10)
        )
    _write(directory, "probe.capture_session.json", _session_json(messages=messages))
    return directory


def test_a_handled_command_is_reported_as_handled(tmp_path: Path) -> None:
    """The ordinary case: a byte the sim routes and answers."""
    coverage = analyze_command_coverage([_archive(tmp_path, (CMD_RADAR,))])

    assert coverage["sessions"] == 1
    radar = [row for row in coverage["rows"] if row["byte"] == CMD_RADAR]
    assert [row["status"] for row in radar] == [STATUS_HANDLED]
    assert [row["kind"] for row in radar] == ["radar"]
    assert crashing_rows(coverage) == []


def test_an_unmapped_byte_is_reported_as_a_crash(tmp_path: Path) -> None:
    """THE POINT OF THE AUDIT.

    A byte the decoder cannot map resolves to ``other``, which
    ``queue_command`` refuses — so a real client sending it takes the
    server down. The audit has to say so, not merely list it.
    """
    coverage = analyze_command_coverage([_archive(tmp_path, (_UNKNOWN_BYTE,))])

    crashing = crashing_rows(coverage)
    assert [row["byte"] for row in crashing] == [_UNKNOWN_BYTE]
    assert [row["kind"] for row in crashing] == ["other"]
    assert [row["constant"] for row in crashing] == [""]


def test_the_audit_is_binary_with_no_parking_status(tmp_path: Path) -> None:
    """0x44 was the audit's one ``declared_unmodelled`` row, and is not.

    That status existed for a byte the decoder mapped and the sim had
    no law for. Reading the JS serializer and six archive windows made
    it the fuel-deposit law, so the class emptied and was deleted:
    every byte is now handled or a crash, and there is nowhere to park
    the next one where it stops counting as a defect.
    """
    coverage = analyze_command_coverage([_archive(tmp_path, (CMD_DEPOSIT_FUEL,))])

    rows = [row for row in coverage["rows"] if row["byte"] == CMD_DEPOSIT_FUEL]
    assert [row["status"] for row in rows] == [STATUS_HANDLED]
    assert [row["kind"] for row in rows] == ["deposit_fuel"]
    assert [row["constant"] for row in rows] == ["CMD_DEPOSIT_FUEL"]
    assert crashing_rows(coverage) == []


def test_sends_are_counted_and_ordered_by_frequency(tmp_path: Path) -> None:
    """The busiest byte leads, because it is the one that matters most."""
    coverage = analyze_command_coverage(
        [_archive(tmp_path, (CMD_RADAR, CMD_KEEPALIVE, CMD_KEEPALIVE, CMD_KEEPALIVE))]
    )

    assert [(row["byte"], row["sends"]) for row in coverage["rows"]] == [
        (CMD_KEEPALIVE, 3),
        (CMD_RADAR, 1),
    ]


def test_constants_never_sent_are_named(tmp_path: Path) -> None:
    """Written-down protocol versus OBSERVED protocol, kept visible.

    Not a defect — real client capabilities nobody in this corpus
    used — but the gap is worth seeing, because it bounds what any
    archive-derived claim can cover.
    """
    coverage = analyze_command_coverage([_archive(tmp_path, (CMD_RADAR,))])

    assert "CMD_PING" in coverage["unsent_constants"]
    assert "CMD_RADAR" not in coverage["unsent_constants"]


def test_the_report_says_plainly_when_something_would_crash(tmp_path: Path) -> None:
    """A reader must not have to scan a table to find the danger."""
    clean = format_command_coverage(analyze_command_coverage([_archive(tmp_path, (CMD_RADAR,))]))
    assert "Every command byte in this archive is handled." in clean

    broken = format_command_coverage(
        analyze_command_coverage([_archive(tmp_path, (_UNKNOWN_BYTE,), name="broken")])
    )
    assert "WOULD CRASH A HOSTED SERVER: 1" in broken


def test_an_empty_archive_audits_to_nothing(tmp_path: Path) -> None:
    """No sessions, no rows — and no crash claim either."""
    empty = tmp_path / "empty"
    empty.mkdir()

    coverage = analyze_command_coverage([empty])

    assert coverage["sessions"] == 0
    assert coverage["rows"] == []
    assert crashing_rows(coverage) == []


def test_rows_and_the_whole_audit_round_trip(tmp_path: Path) -> None:
    """The audit is written to an artifact, so it must survive one."""
    coverage = analyze_command_coverage([_archive(tmp_path, (CMD_RADAR, _UNKNOWN_BYTE))])

    assert decode_command_coverage(encode_command_coverage(coverage)) == coverage
    row = coverage["rows"][0]
    assert decode_command_byte_row(encode_command_byte_row(row)) == row


def test_decode_rejects_a_constant_name_that_is_not_a_string() -> None:
    """A malformed artifact fails decode rather than coercing."""
    broken: JSONObject = {"sessions": 0, "rows": [], "unsent_constants": [7]}
    with pytest.raises(TypeError):
        decode_command_coverage(broken)


def test_a_row_states_every_field_the_report_needs() -> None:
    """The typed row is the contract the formatter reads."""
    row = CommandByteRowDict(
        byte=0x21, constant="CMD_KEEPALIVE", kind="keepalive", sends=11871, status=STATUS_HANDLED
    )
    coverage = CommandCoverageDict(sessions=1, rows=[row], unsent_constants=[])

    text = format_command_coverage(coverage)
    assert "0x21" in text
    assert "CMD_KEEPALIVE" in text
    assert "11871" in text


def test_a_session_that_cannot_decode_is_skipped_not_counted(tmp_path: Path) -> None:
    """A magic-less capture is a typed skip, not a session and not a crash.

    Counting it would inflate the denominator; crashing on it would
    make one corrupt archive file hide every real finding behind it.
    """
    directory = tmp_path / "unreadable"
    directory.mkdir()
    _write(directory, "nomagic.capture_session.json", _session_json(magic=None))

    coverage = analyze_command_coverage([directory])

    assert coverage["sessions"] == 0
    assert coverage["rows"] == []


def test_a_frame_the_decoder_cannot_read_is_not_a_command_byte(tmp_path: Path) -> None:
    """Archive noise contributes no row rather than a phantom crash.

    A bare prefix with no type or command byte is a truncated frame,
    not a command the sim fails to handle — reporting it as an
    unmapped byte would raise a false alarm on the one signal this
    audit exists to make trustworthy.
    """
    directory = tmp_path / "noisy"
    directory.mkdir()
    _write(
        directory,
        "noise.capture_session.json",
        _session_json(
            messages=[
                _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
                _sent(_payload(_ciphered(bytes([_PREFIX]))), timestamp_ms=1100),
                _sent(_payload(_command(_client_command(CMD_RADAR))), timestamp_ms=1200),
            ]
        ),
    )

    coverage = analyze_command_coverage([directory])

    assert [row["byte"] for row in coverage["rows"]] == [CMD_RADAR]
    assert crashing_rows(coverage) == []
