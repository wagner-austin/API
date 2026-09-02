"""Tests for the recipient-policy sweep and its data contract.

Frames are built with the production encoders and the production
cipher, so a fixture cannot quietly disagree with the decoder it is
meant to exercise. Sessions are real JSON on disk read through the real
scan pipeline.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.analysis.recipient_policy import (
    FAMILY_TABLE,
    analyze_recipient_policy,
    format_recipient_policy,
    merge_session_evidence,
    sweep_session,
)
from tankpit_bot.analysis.recipient_policy_types import (
    VERDICT_BROADCAST,
    VERDICT_PER_RECIPIENT,
    VERDICT_UNDETERMINED,
    FamilyCountDict,
    FamilyEvidenceDict,
    RecipientPolicyDict,
    SessionEvidenceDict,
    decode_family_count,
    decode_family_evidence,
    decode_recipient_policy,
    decode_session_evidence,
    encode_family_count,
    encode_family_evidence,
    encode_recipient_policy,
    encode_session_evidence,
)
from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.analysis.types import ScannedSessionDict
from tankpit_bot.protocol.commands import (
    build_block_command,
    build_teleport_command,
)
from tankpit_bot.protocol.framing import encode_frame
from tests.analysis._capture_fixtures import (
    FOREIGN_TANK,
    OWN_TANK,
    _build_pickup,
    _ciphered,
    _payload,
    _received,
    _sent,
    _session_json,
    _tank_info,
    _write,
)

_BUILD_PICKUP_INDEX = 0
_TERRAIN_INDEX = 1
_TOGGLE_INDEX = 2
_TELEPORT_LANDED_INDEX = 6


def _command(framed: bytes) -> bytes:
    """Cipher one client command into its sent-frame body.

    The production ``build_*_command`` helpers return a COMPLETE framed
    message — a 2-byte LE length prefix, then ``!``, then the command —
    so the prefix is dropped here and :func:`_payload` re-frames. Left
    on, the frame's leading byte reads 0x05 instead of ``!`` and the
    sweep correctly declines to treat it as a command at all.

    Args:
        framed: Bytes from a production ``build_*_command``.

    Returns:
        Wire body the scanner decodes back to that command.
    """
    return _ciphered(framed[2:])


def _scan(tmp_path: Path, *messages: JSONObject) -> ScannedSessionDict:
    """Write a one-session capture and scan it, failing on a skip.

    Args:
        tmp_path: Directory to write into.
        *messages: Captured-message objects, in order.

    Returns:
        The decoded session.

    Raises:
        AssertionError: If the session did not decode — a fixture bug,
            never a condition under test here.
    """
    text = _session_json(messages=list(messages))
    result = scan_session(_write(tmp_path, "a.capture_session.json", text))
    if result["kind"] != "scanned":
        raise AssertionError(f"fixture session did not decode: {result}")
    return result


def _family(entry: SessionEvidenceDict, index: int) -> FamilyCountDict:
    """Return one family's tally from a session's evidence.

    Args:
        entry: The session evidence.
        index: Position in :data:`FAMILY_TABLE`.

    Returns:
        The tally at that position.
    """
    return entry["families"][index]


def test_family_table_order_is_the_reported_order() -> None:
    """The table is the contract every ``families`` list is built to."""
    assert [label for _key, label, _trigger in FAMILY_TABLE] == [
        "0x42 BuildPickup",
        "0x4A TerrainUpdate",
        "0x74 EquipmentToggle",
        "0x4F RadarScanResult",
        "0x46 RadarResult",
        "0x4C MapData",
        "TeleportLanded",
    ]


def test_sweep_counts_received_families_and_finds_the_own_tank(tmp_path: Path) -> None:
    """The first 0x21 names the own tank; families tally by decoded type."""
    entry = sweep_session(
        _scan(
            tmp_path,
            _received(_payload(_tank_info(OWN_TANK))),
            _received(_payload(_build_pickup(OWN_TANK), _build_pickup(OWN_TANK))),
        )
    )
    assert entry["session_id"] == "s-1"
    assert entry["own_tank_id"] == OWN_TANK
    assert entry["identified_own_tank"] is True
    assert _family(entry, _BUILD_PICKUP_INDEX)["received"] == 2
    assert _family(entry, _BUILD_PICKUP_INDEX)["foreign_actor_hits"] == 0
    assert _family(entry, _TERRAIN_INDEX)["received"] == 0
    assert entry["undecodable_frames"] == 0


def test_sweep_flags_a_build_pickup_naming_another_tank(tmp_path: Path) -> None:
    """A 0x42 naming a foreign actor is the positive broadcast proof.

    This is the shape of the single archive sample that settles 0x42:
    ``bot-20260826-003928``, own tank 601, a drop by tank 709
    ([[recipient-policy]]).
    """
    entry = sweep_session(
        _scan(
            tmp_path,
            _received(_payload(_tank_info(OWN_TANK))),
            _received(_payload(_build_pickup(FOREIGN_TANK))),
        )
    )
    assert _family(entry, _BUILD_PICKUP_INDEX)["foreign_actor_hits"] == 1


def test_sweep_cannot_judge_a_foreign_actor_without_an_own_tank(tmp_path: Path) -> None:
    """No 0x21 means no baseline, so the foreign test is not run at all.

    Reporting a hit against a fabricated id would invent evidence for
    the verdict this module exists to decide.
    """
    entry = sweep_session(_scan(tmp_path, _received(_payload(_build_pickup(FOREIGN_TANK)))))
    assert entry["identified_own_tank"] is False
    assert entry["own_tank_id"] == 0
    assert _family(entry, _BUILD_PICKUP_INDEX)["received"] == 1
    assert _family(entry, _BUILD_PICKUP_INDEX)["foreign_actor_hits"] == 0


def test_sweep_finds_the_own_tank_past_leading_noise(tmp_path: Path) -> None:
    """The identity is found behind sent, unclaimed and other frames.

    A real session opens with lobby and command traffic before the
    roster dump, so the search walks past four kinds of frame that are
    not the answer: a sent command, a frame no decoder claims, a SHORT
    frame of a known type, and a decoded message that is simply not a
    0x21. The short frame matters — the archive holds 101 short 0x41s
    and any of them can precede the identity.
    """
    entry = sweep_session(
        _scan(
            tmp_path,
            _sent(_payload(_command(build_block_command(1, 2)))),
            _received(_payload(_ciphered(bytes([0x01, 0x99])))),
            _received(_payload(_ciphered(bytes([0x41, 0x99])))),
            _received(_payload(_build_pickup(FOREIGN_TANK))),
            _received(_payload(_tank_info(OWN_TANK))),
        )
    )
    assert entry["identified_own_tank"] is True
    assert entry["own_tank_id"] == OWN_TANK
    assert entry["undecodable_frames"] == 1
    assert entry["malformed_frames"] == 1
    # The 0x42 arrived BEFORE the identity and is still judged against
    # it: the own tank is resolved in its own pass, so ordering inside
    # the session cannot hide a foreign actor.
    assert _family(entry, _BUILD_PICKUP_INDEX)["foreign_actor_hits"] == 1


def test_sweep_separates_a_short_known_frame_from_an_unknown_one(tmp_path: Path) -> None:
    """A short 0x41 is malformed, not undecodable — different facts.

    An unknown message type says the decoder has a gap; a KNOWN type
    whose body fails validation says the capture holds a short frame.
    Conflating them hides both. This is the archive's actual shape:
    101 of its 262,588 received frames are short 0x41 Deactivations,
    and the real sweep died on the first one until they were counted.
    """
    entry = sweep_session(
        _scan(
            tmp_path,
            _received(_payload(_tank_info(OWN_TANK))),
            _received(_payload(_ciphered(bytes([0x41, 0x99])))),
            _received(_payload(_ciphered(bytes([0x01, 0x99])))),
        )
    )
    assert entry["malformed_frames"] == 1
    assert entry["undecodable_frames"] == 1
    assert entry["identified_own_tank"] is True


def test_sweep_counts_own_triggers_from_sent_command_frames(tmp_path: Path) -> None:
    """Sent frames are decoded as commands and tallied by kind."""
    entry = sweep_session(
        _scan(
            tmp_path,
            _received(_payload(_tank_info(OWN_TANK))),
            _sent(_payload(_command(build_block_command(10, 11)))),
            _sent(_payload(_command(build_teleport_command(12, 13)))),
        )
    )
    assert _family(entry, _BUILD_PICKUP_INDEX)["own_triggers"] == 1
    assert _family(entry, _TERRAIN_INDEX)["own_triggers"] == 1
    assert _family(entry, _TELEPORT_LANDED_INDEX)["own_triggers"] == 1


def test_sweep_ignores_sent_frames_that_are_not_commands(tmp_path: Path) -> None:
    """The lobby shares the socket; only ``!`` frames are commands."""
    entry = sweep_session(
        _scan(
            tmp_path,
            _received(_payload(_tank_info(OWN_TANK))),
            _sent(_payload(_ciphered(bytes([0x2B, 0x01, 0x02])))),
        )
    )
    assert _family(entry, _BUILD_PICKUP_INDEX)["own_triggers"] == 0


def test_sweep_counts_frames_no_decoder_claims(tmp_path: Path) -> None:
    """An unclaimed frame is COUNTED, never silently dropped.

    An unreported skip would understate every ``received`` tally it
    hides, which is the one failure mode a coverage sweep cannot
    tolerate.
    """
    entry = sweep_session(
        _scan(
            tmp_path,
            _received(_payload(_tank_info(OWN_TANK))),
            _received(_payload(_ciphered(bytes([0x01, 0x99])))),
        )
    )
    assert entry["undecodable_frames"] == 1
    assert entry["identified_own_tank"] is True


def test_merge_rules_broadcast_on_a_single_zero_trigger_session() -> None:
    """One session receiving with no trigger settles broadcast."""
    silent = _evidence(received=1, own_triggers=0)
    result = merge_session_evidence(
        [silent], sessions_examined=1, sessions_without_magic=0, framing_errors=0
    )
    family = result["families"][_BUILD_PICKUP_INDEX]
    assert family["zero_trigger_sessions"] == 1
    assert family["verdict"] == VERDICT_BROADCAST


def test_merge_rules_broadcast_on_a_foreign_actor_alone() -> None:
    """A foreign actor settles broadcast even with triggers present."""
    entry = _evidence(received=1, own_triggers=1, foreign=1)
    result = merge_session_evidence(
        [entry], sessions_examined=1, sessions_without_magic=0, framing_errors=0
    )
    family = result["families"][_BUILD_PICKUP_INDEX]
    assert family["zero_trigger_sessions"] == 0
    assert family["foreign_actor_hits"] == 1
    assert family["verdict"] == VERDICT_BROADCAST


def test_merge_rules_per_recipient_when_the_archive_is_silent() -> None:
    """Per-recipient is the verdict only with no positive evidence."""
    entry = _evidence(received=5, own_triggers=5)
    result = merge_session_evidence(
        [entry], sessions_examined=1, sessions_without_magic=0, framing_errors=0
    )
    family = result["families"][_BUILD_PICKUP_INDEX]
    assert family["received"] == 5
    assert family["own_triggers"] == 5
    assert family["zero_trigger_sessions"] == 0
    assert family["verdict"] == VERDICT_PER_RECIPIENT


def test_a_family_no_command_triggers_is_left_undetermined() -> None:
    """The zero-trigger test cannot rule on a join-burst family.

    0x74 arrives once per session having answered no command, which
    says "not command-triggered" — a different fact from "broadcast".
    Emitting a verdict here would put the machine in silent conflict
    with the structural ruling in [[recipient-policy]], and a future
    reader would trust the machine.
    """
    entry = _evidence(received=1, own_triggers=0, index=_TOGGLE_INDEX)
    result = merge_session_evidence(
        [entry], sessions_examined=1, sessions_without_magic=0, framing_errors=0
    )
    family = result["families"][_TOGGLE_INDEX]
    assert family["trigger_kind"] == ""
    assert family["zero_trigger_sessions"] == 1
    assert family["verdict"] == VERDICT_UNDETERMINED


def test_a_foreign_actor_outranks_a_missing_trigger() -> None:
    """Direct proof settles broadcast even with no triggering command."""
    entry = _evidence(received=1, own_triggers=0, foreign=1, index=_TOGGLE_INDEX)
    result = merge_session_evidence(
        [entry], sessions_examined=1, sessions_without_magic=0, framing_errors=0
    )
    assert result["families"][_TOGGLE_INDEX]["verdict"] == VERDICT_BROADCAST


def test_merge_carries_the_skip_denominators() -> None:
    """Skips are reported, so the denominator stays honest."""
    result = merge_session_evidence(
        [], sessions_examined=9, sessions_without_magic=2, framing_errors=1
    )
    assert result["sessions_examined"] == 9
    assert result["sessions_decoded"] == 0
    assert result["sessions_without_magic"] == 2
    assert result["framing_errors"] == 1
    assert result["undecodable_frames"] == 0


def test_analyze_sweeps_a_directory_and_tallies_both_skip_kinds(tmp_path: Path) -> None:
    """End to end: a decoded session, a magic-less one, an unframed one."""
    _write(
        tmp_path,
        "a.capture_session.json",
        _session_json(
            messages=[
                _received(_payload(_tank_info(OWN_TANK))),
                _received(_payload(_build_pickup(FOREIGN_TANK))),
            ]
        ),
    )
    _write(tmp_path, "b.capture_session.json", _session_json(magic=None))
    truncated = base64.b64encode(encode_frame(bytes([0x53, 0x11, 0x22]))[:-1]).decode("ascii")
    _write(
        tmp_path,
        "c.capture_session.json",
        _session_json(messages=[_sent(truncated)]),
    )

    result = analyze_recipient_policy([tmp_path])
    assert result["sessions_examined"] == 3
    assert result["sessions_decoded"] == 1
    assert result["sessions_without_magic"] == 1
    assert result["framing_errors"] == 1
    assert result["families"][_BUILD_PICKUP_INDEX]["verdict"] == VERDICT_BROADCAST


def test_format_names_every_family_and_its_verdict() -> None:
    """The report carries the verdict and the evidence behind it."""
    result = merge_session_evidence(
        [_evidence(received=3, own_triggers=0)],
        sessions_examined=1,
        sessions_without_magic=0,
        framing_errors=0,
    )
    text = format_recipient_policy(result)
    assert "sessions_examined=1" in text
    assert "sessions_decoded=1" in text
    assert f"0x42 BuildPickup -> {VERDICT_BROADCAST}" in text
    assert "zero_trigger_sessions=1" in text
    assert "malformed_frames=0" in text
    for _key, label, _trigger in FAMILY_TABLE:
        assert label in text


def _evidence(
    *, received: int, own_triggers: int, foreign: int = 0, index: int = _BUILD_PICKUP_INDEX
) -> SessionEvidenceDict:
    """Build one session's evidence with a single family populated.

    Args:
        received: Arrivals of the family at ``index``.
        own_triggers: Triggering commands the client sent.
        foreign: Arrivals naming another tank as actor.
        index: Which family in :data:`FAMILY_TABLE` to populate.

    Returns:
        Session evidence in :data:`FAMILY_TABLE` order.
    """
    families = [
        FamilyCountDict(
            label=label,
            received=received if position == index else 0,
            own_triggers=own_triggers if position == index else 0,
            foreign_actor_hits=foreign if position == index else 0,
        )
        for position, (_key, label, _trigger) in enumerate(FAMILY_TABLE)
    ]
    return SessionEvidenceDict(
        session_id="s-1",
        own_tank_id=OWN_TANK,
        identified_own_tank=True,
        families=families,
        framing_errors=0,
        undecodable_frames=0,
        malformed_frames=0,
    )


def test_family_count_round_trips() -> None:
    """Encode then decode returns an equal tally."""
    original = FamilyCountDict(
        label="0x42 BuildPickup", received=3, own_triggers=1, foreign_actor_hits=2
    )
    assert decode_family_count(encode_family_count(original)) == original


def test_session_evidence_round_trips() -> None:
    """The nested family list survives the round trip."""
    original = _evidence(received=2, own_triggers=1, foreign=1)
    assert decode_session_evidence(encode_session_evidence(original)) == original


def test_family_evidence_round_trips() -> None:
    """Verdict and every counter survive the round trip."""
    original = FamilyEvidenceDict(
        label="0x4A TerrainUpdate",
        trigger_kind="block",
        received=309,
        own_triggers=59,
        zero_trigger_sessions=45,
        foreign_actor_hits=0,
        verdict=VERDICT_BROADCAST,
    )
    assert decode_family_evidence(encode_family_evidence(original)) == original


def test_recipient_policy_round_trips() -> None:
    """The whole sweep survives the round trip, families included."""
    original: RecipientPolicyDict = merge_session_evidence(
        [_evidence(received=1, own_triggers=0)],
        sessions_examined=2,
        sessions_without_magic=1,
        framing_errors=0,
    )
    assert decode_recipient_policy(encode_recipient_policy(original)) == original


def test_decode_family_count_rejects_a_missing_field() -> None:
    """A malformed tally fails decode rather than defaulting."""
    with pytest.raises((JSONTypeError, KeyError)):
        decode_family_count({"label": "x", "received": 1, "own_triggers": 1})


def test_decode_session_evidence_rejects_a_non_object_family() -> None:
    """A family that is not an object is a decode failure, not a skip."""
    encoded = encode_session_evidence(_evidence(received=1, own_triggers=0))
    encoded["families"] = ["not an object"]
    with pytest.raises(JSONTypeError):
        decode_session_evidence(encoded)


def test_decode_recipient_policy_rejects_a_non_object_family() -> None:
    """The sweep's family list is validated element by element."""
    encoded = encode_recipient_policy(
        merge_session_evidence([], sessions_examined=0, sessions_without_magic=0, framing_errors=0)
    )
    encoded["families"] = [42]
    with pytest.raises(JSONTypeError):
        decode_recipient_policy(encoded)
