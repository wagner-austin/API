"""Tests for the capture replay cross-validation.

Frames are XOR-encoded with the same table the audit builds (fake
static key from the ``fake_fs`` fixture + the capture's magic), so the
replay path runs the REAL decoder end to end. Nothing is mocked.
"""

from __future__ import annotations

import base64

from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks
from tankpit_bot.capture.xor import build_xor_table, load_xor_static_key
from tankpit_bot.diagnostics.capture_audit import audit_capture
from tankpit_bot.diagnostics.run_audit_types import FindingDict, make_finding
from tankpit_bot.runtime_logging import RuntimeEventRecordDict
from tankpit_bot.types import CapturedMessage, CaptureSession, GameLogEntryWithTimestamp

_MAGIC = "testmagic"


def _record(**fields: str | int | float | bool) -> RuntimeEventRecordDict:
    """Build one DIAGNOSTIC event record with the given fields."""
    return RuntimeEventRecordDict(
        timestamp="2026-07-19T00:48:00",
        level="INFO",
        logger="tankpit_bot.runtime.events",
        mode="bot",
        channel="DIAGNOSTIC",
        fields=dict(fields),
        message="",
    )


def _table() -> bytes:
    """Build the XOR table the audit will build (fake key + magic)."""
    static_key, _ = load_xor_static_key(None)
    assert static_key is not None
    return build_xor_table(static_key, _MAGIC)


def _frame(msg_type: int, plain: bytes) -> bytes:
    """Encode one length-prefixed frame with an XOR-encoded body."""
    table = _table()
    encoded = bytearray([msg_type])
    for index, value in enumerate(plain):
        encoded.append(value ^ table[index] if index < len(table) else value)
    body = bytes(encoded)
    return bytes([len(body) & 0xFF, len(body) >> 8]) + body


def _received(*frames: bytes) -> CapturedMessage:
    """Wrap frames into one received capture message."""
    return CapturedMessage(
        timestamp_ms=1000,
        direction="received",
        payload=base64.b64encode(b"".join(frames)).decode("ascii"),
        ws_url="wss://test",
    )


def _capture(
    messages: list[CapturedMessage],
    game_log: list[GameLogEntryWithTimestamp] | None = None,
    magic: str | None = _MAGIC,
) -> CaptureSession:
    """Build a capture session around the given messages."""
    return CaptureSession(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1,
        base_url="https://test",
        messages=messages,
        magic=magic,
        game_log=[] if game_log is None else game_log,
        tank_names={},
    )


def _deactivation_frame(victim_id: int, killer_id: int) -> bytes:
    """Encode a top-level 0x41 Deactivation frame."""
    plain = bytes(
        [
            0,
            victim_id & 0xFF,
            victim_id >> 8,
            1,
            killer_id & 0xFF,
            killer_id >> 8,
        ]
    )
    return _frame(0x41, plain)


def _supervisor_frame(error_code: int) -> bytes:
    """Encode a top-level 0x52 Supervisor frame."""
    return _frame(0x52, bytes([1, 0, error_code]))


def _by_check(findings: list[FindingDict], check: str) -> list[FindingDict]:
    """Return the findings produced by one check."""
    return [f for f in findings if f["check"] == check]


def test_missing_magic_skips_the_replay(fake_fs: FakeFileSystem) -> None:
    """A capture without XOR magic cannot be replayed."""
    findings = audit_capture(_capture([], magic=None), [])
    assert findings == [
        make_finding(
            "capture_unreadable",
            "warning",
            "capture carries no XOR magic -- replay audit skipped",
        )
    ]


def test_missing_static_key_skips_the_replay() -> None:
    """Without the XOR static key file the replay is skipped."""
    original = _test_hooks.path_exists
    _test_hooks.path_exists = lambda path: False
    try:
        findings = audit_capture(_capture([]), [])
    finally:
        _test_hooks.path_exists = original
    assert findings == [
        make_finding(
            "capture_unreadable",
            "warning",
            "XOR static key file missing -- replay audit skipped",
        )
    ]


def test_matching_channels_report_agreement(fake_fs: FakeFileSystem) -> None:
    """Wire counts that match the ledger produce info verdicts."""
    capture = _capture(
        [
            _received(
                _deactivation_frame(500, 1301),
                _supervisor_frame(4),
            ),
            CapturedMessage(
                timestamp_ms=1001,
                direction="sent",
                payload=base64.b64encode(b"ignored").decode("ascii"),
                ws_url="wss://test",
            ),
        ],
        game_log=[
            GameLogEntryWithTimestamp(
                timestamp_ms=1002,
                text="red-1 has been deactivated by you",
                category="combat",
            ),
            GameLogEntryWithTimestamp(
                timestamp_ms=1003,
                text="Empty container",
                category="other",
            ),
        ],
    )
    records = [
        _record(diagnostic_kind="tank_identity", tank_id=1301, name="Artax"),
        _record(
            diagnostic_kind="tank_deactivated",
            origin="protocol_0x41",
            victim_id=500,
            killer_id=1301,
        ),
        _record(diagnostic_kind="command_error", error_code=4),
    ]
    findings = audit_capture(capture, records)
    assert _by_check(findings, "deactivation_channel_diff") == [
        make_finding(
            "deactivation_channel_diff",
            "info",
            "0x41 deactivations: wire and ledger agree (1)",
            wire=1,
            ledger=1,
        )
    ]
    assert _by_check(findings, "supervisor_channel_diff") == [
        make_finding(
            "supervisor_channel_diff",
            "info",
            "0x52 command errors: wire and ledger agree (1)",
            wire=1,
            ledger=1,
        )
    ]
    assert _by_check(findings, "dom_witness_diff") == [
        make_finding(
            "dom_witness_diff",
            "info",
            "kill banners consistent with the wire (1 banner(s), 1 wire message(s))",
            banners=1,
            wire=1,
        ),
        make_finding(
            "dom_witness_diff",
            "info",
            "empty-container banners consistent with the wire (1 banner(s), 1 wire message(s))",
            banners=1,
            wire=1,
        ),
    ]
    assert _by_check(findings, "decode_error") == []
    assert _by_check(findings, "unknown_container_subtypes") == []


def test_wire_ledger_mismatch_is_critical(fake_fs: FakeFileSystem) -> None:
    """A wire message the run never ingested is the June class of bug."""
    capture = _capture([_received(_deactivation_frame(500, 1301))])
    findings = audit_capture(capture, [])
    assert _by_check(findings, "deactivation_channel_diff") == [
        make_finding(
            "deactivation_channel_diff",
            "critical",
            "0x41 deactivations: capture replay found 1 but the run "
            "ingested 0 -- decode/dispatch gap",
            wire=1,
            ledger=0,
        )
    ]


def test_unrendered_wire_banner_gap_is_critical(fake_fs: FakeFileSystem) -> None:
    """A banner the client rendered with no wire message is the canary."""
    capture = _capture(
        [],
        game_log=[
            GameLogEntryWithTimestamp(
                timestamp_ms=1000,
                text="You can't go there!",
                category="other",
            )
        ],
    )
    findings = audit_capture(capture, [])
    assert _by_check(findings, "dom_witness_diff") == [
        make_finding(
            "dom_witness_diff",
            "critical",
            "the client rendered 1 blocked-move banner(s) but the wire "
            "carried only 0 0x52 code-1 errors -- the decoder is "
            "missing something the client can see",
            banners=1,
            wire=0,
        )
    ]


def test_unknown_container_subtype_is_the_blind_spot_canary(
    fake_fs: FakeFileSystem,
) -> None:
    """An undecoded 0x2E subtype surfaces as a warning with its census."""
    capture = _capture([_received(_frame(0x2E, bytes([0x99, 1, 2, 3, 4, 5])))])
    findings = audit_capture(capture, [])
    assert _by_check(findings, "unknown_container_subtypes") == [
        make_finding(
            "unknown_container_subtypes",
            "warning",
            "1 0x2E message(s) fell through to unknown_container -- "
            "undecoded wire channels (the June-blind-spot canary)",
            subtypes="0x99x1",
        )
    ]


def test_decoder_crash_is_critical(fake_fs: FakeFileSystem) -> None:
    """A frame the current decoder raises on is a critical finding."""
    # RadarScanResult (0x4F 'O'): remaining bytes after the u16 count
    # must divide by 3; two trailing bytes force a DecodeError.
    capture = _capture([_received(_frame(0x4F, bytes([1, 0, 7, 7])))])
    findings = audit_capture(capture, [])
    assert _by_check(findings, "decode_error") == [
        make_finding(
            "decode_error",
            "critical",
            "1 received frame(s) crashed the current decoder -- the "
            "wire carries a shape the decoder rejects",
            count=1,
        )
    ]


def test_malformed_payloads_and_short_frames_are_skipped(
    fake_fs: FakeFileSystem,
) -> None:
    """Garbage base64, truncated frames, and sub-minimum bodies replay to nothing."""
    table_frame = _frame(0x41, bytes([0]))  # below the 0x41 minimum length
    truncated = bytes([200, 0, 0x41])  # length prefix exceeds the data
    capture = _capture(
        [
            CapturedMessage(
                timestamp_ms=1000,
                direction="received",
                payload="not-base64!!!",
                ws_url="wss://test",
            ),
            _received(table_frame),
            CapturedMessage(
                timestamp_ms=1001,
                direction="received",
                payload=base64.b64encode(truncated).decode("ascii"),
                ws_url="wss://test",
            ),
        ]
    )
    findings = audit_capture(capture, [])
    assert _by_check(findings, "decode_error") == []
    assert _by_check(findings, "deactivation_channel_diff") == [
        make_finding(
            "deactivation_channel_diff",
            "info",
            "0x41 deactivations: wire and ledger agree (0)",
            wire=0,
            ledger=0,
        )
    ]


def test_xor_passthrough_beyond_the_table_length() -> None:
    """Bytes past the XOR table's end pass through unmodified."""
    from tankpit_bot.diagnostics.capture_audit import _xor_with_table

    assert _xor_with_table(bytes([0x41, 0x0F, 0x22]), bytes([0x01])) == bytes([0x0E, 0x22])


def test_other_messages_and_unknown_identity_fields(fake_fs: FakeFileSystem) -> None:
    """Non-audited decodes, 1-byte bodies, and untyped identity ids are benign."""
    capture = _capture(
        [
            _received(
                _frame(0x44, bytes([1, 0, 100])),  # FuelGain -- audited by no check
                bytes([1, 0, 0x41]),  # 1-byte body: type byte only, nothing to decode
                _deactivation_frame(500, 1301),
            )
        ],
        game_log=[
            GameLogEntryWithTimestamp(
                timestamp_ms=1000,
                text="You hit red-1",
                category="combat",
            )
        ],
    )
    records = [
        _record(diagnostic_kind="tank_identity", tank_id="not-an-id"),
        _record(
            diagnostic_kind="tank_deactivated",
            origin="protocol_0x41",
            victim_id=500,
            killer_id=1301,
        ),
    ]
    findings = audit_capture(capture, records)
    assert _by_check(findings, "decode_error") == []
    assert _by_check(findings, "dom_witness_diff") == []
    assert _by_check(findings, "deactivation_channel_diff") == [
        make_finding(
            "deactivation_channel_diff",
            "info",
            "0x41 deactivations: wire and ledger agree (1)",
            wire=1,
            ledger=1,
        )
    ]
