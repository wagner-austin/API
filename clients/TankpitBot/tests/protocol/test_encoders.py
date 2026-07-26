"""Byte-exact round-trip tests for every server-message encoder.

Each case builds a wire payload, decodes it with the production
decoder, re-encodes with the matching encoder, and requires byte
identity — the same contract ``make roundtrip`` enforces over the
full capture archive.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container.decoders import decode_container_message
from tankpit_bot.protocol.decoders import (
    decode_0x2e_message,
    decode_action_done,
    decode_active_forces,
    decode_active_players,
    decode_build_pickup,
    decode_cache_update,
    decode_chat_message,
    decode_deactivation,
    decode_decoration,
    decode_enemy_detection,
    decode_equipment_gain,
    decode_equipment_toggle,
    decode_fuel_deposit,
    decode_fuel_gain,
    decode_inventory,
    decode_map_data,
    decode_movement,
    decode_movement_response,
    decode_overlay_update,
    decode_promotion,
    decode_radar_result,
    decode_radar_scan_result,
    decode_shoot_event,
    decode_statistics,
    decode_supervisor,
    decode_supervisor_text,
    decode_sync,
    decode_tank_entry,
    decode_tank_exit,
    decode_tank_info,
    decode_tank_remove,
    decode_tank_status,
    decode_tank_status_sync,
    decode_terrain_update,
    decode_viewport_update,
    try_decode_plaintext_ack,
)
from tankpit_bot.protocol.decoders.session_events import (
    decode_connection_lost,
    decode_ping_response,
    decode_top10,
)
from tankpit_bot.protocol.encoders import (
    encode_envelope_body,
    encode_message_payload,
    encode_plaintext_ack,
)
from tankpit_bot.protocol.helpers import EncodeError, pack16, pack24, x16, x24


def test_pack16_and_pack24_invert_the_unpack_helpers() -> None:
    """pack16/pack24 are exact inverses of x16/x24 at the boundaries."""
    for value in (0, 1, 0x1234, 0xFFFF):
        low, high = pack16(value)
        assert x16(low, high) == value
    for value in (0, 1, 0x123456, 0xFFFFFF):
        a, b, c = pack24(value)
        assert x24(a, b, c) == value


def test_tank_info_roundtrip() -> None:
    """0x21 TankInfo re-encodes byte-identically, name included."""
    payload = bytes([2, 45, 1]) + b"\x01\x02\x03\x04" + bytes([0, 3, 231]) + b"Artax"
    message = decode_tank_info(payload)
    assert message["persistent_tank_id"] == 256 * (256 * 0 + 3) + 231
    assert encode_message_payload(message) == payload


def test_tank_entry_roundtrip_rederives_flags_from_team() -> None:
    """0x28 TankEntry re-encodes with the corpus flags-equals-team rule."""
    packed = 3 | (2 << 2) | (9 << 4)
    payload = bytes([3, 16, 0, packed, 0, 1, 44, 7, 8])
    message = decode_tank_entry(payload)
    assert message["team"] == 3
    assert message["damage_state"] == 2
    assert message["rank"] == 9
    assert encode_message_payload(message) == payload


def test_tank_exit_roundtrip() -> None:
    """0x29 TankExit re-encodes both boolean bytes."""
    payload = bytes([1, 9, 0, 1, 0])
    message = decode_tank_exit(payload)
    assert message["was_silent"] is True
    assert message["was_eliminated"] is False
    assert encode_message_payload(message) == payload


def test_tank_status_sync_all_three_length_variants() -> None:
    """0x2E sync re-encodes the 8/9/12-byte wire variants."""
    bare = bytes([0, 7, 0, 1, 4, 0, 0, 55])
    with_promo = bare + bytes([2])
    with_fuel = with_promo + bytes([1, 0x4C, 0x04])
    for payload in (bare, with_promo, with_fuel):
        message = decode_tank_status_sync(payload)
        assert encode_message_payload(message) == payload
    full = decode_tank_status_sync(with_fuel)
    assert full["fuel"] == 1100


def test_tank_status_roundtrip_preserves_info_byte_bits() -> None:
    """0x3E TankStatus keeps the damage bits of the info byte."""
    info = 2 | (3 << 2) | (1 << 4)
    payload = bytes([info, 21, 0]) + bytes(4) + pack24(412) + pack24(9) + b"Artax"
    message = decode_tank_status(payload)
    assert message["team"] == 2
    assert message["damage_state"] == 3
    assert message["rank"] == 1
    assert encode_message_payload(message) == payload


def test_tank_remove_roundtrip() -> None:
    """0x58 TankRemove re-encodes its tank id."""
    payload = pack16(528)
    assert encode_message_payload(decode_tank_remove(payload)) == payload


def test_movement_roundtrip_with_and_without_path() -> None:
    """0x47 Movement re-encodes the nsew tail verbatim."""
    head = pack16(1301) + bytes([9, 218, 2, 0]) + pack24(1000) + bytes([7, 1, 0])
    for path in (b"", b"sswwwwww"):
        payload = head + path
        message = decode_movement(payload)
        assert message["path"] == path.decode("ascii")
        assert encode_message_payload(message) == payload


def test_movement_response_roundtrip() -> None:
    """0x3D MovementResponse re-encodes all 12 bytes."""
    payload = bytes([1]) + pack16(9) + bytes([97, 165, 3, 2, 5]) + pack24(88) + bytes([1])
    message = decode_movement_response(payload)
    assert message["carrying"] == 1
    assert encode_message_payload(message) == payload


def test_fuel_gain_roundtrip_preserves_raw_flag() -> None:
    """0x44 FuelGain re-emits the raw flag byte, not a re-derived bool."""
    free = pack16(1100) + bytes([0])
    valued = pack16(1100) + bytes([0x2B])
    for payload in (free, valued):
        message = decode_fuel_gain(payload)
        assert encode_message_payload(message) == payload
    assert decode_fuel_gain(valued)["is_free"] is False


def test_fuel_deposit_roundtrip() -> None:
    """0x64 FuelDeposit re-encodes the absolute total."""
    payload = pack16(755)
    assert encode_message_payload(decode_fuel_deposit(payload)) == payload


def test_inventory_roundtrip_all_display_modes() -> None:
    """0x49 Inventory re-encodes show/alternate/neither and the disable bit."""
    slots = bytes([25, 10 | 128, 0, 99, 5])
    for head in (0, 1, 2):
        payload = bytes([head]) + slots
        message = decode_inventory(payload)
        assert encode_message_payload(message) == payload
    disabled = decode_inventory(bytes([1]) + slots)
    assert disabled["enabled"] == [True, False, True, True, True]


def test_equipment_gain_roundtrip() -> None:
    """0x67 EquipmentGain re-encodes both show variants."""
    for head in (0, 1):
        payload = bytes([head, 1, 2, 3, 4, 5])
        assert encode_message_payload(decode_equipment_gain(payload)) == payload


def test_equipment_toggle_roundtrip() -> None:
    """0x74 EquipmentToggle re-encodes the five slot booleans."""
    payload = bytes([1, 0, 1, 1, 0])
    assert encode_message_payload(decode_equipment_toggle(payload)) == payload


def test_shoot_event_roundtrip() -> None:
    """0x53 ShootEvent re-encodes all ten bytes."""
    payload = bytes([0]) + pack16(1301) + bytes([42, 164, 46, 165, 55, 167, 0])
    message = decode_shoot_event(payload)
    assert message["target_x"] == 46
    assert encode_message_payload(message) == payload


def test_deactivation_roundtrip_mine_and_tank_killer() -> None:
    """0x41 Deactivation restores the 65530 mine-killer offset."""
    tank_kill = bytes([1]) + pack16(9) + bytes([1]) + pack16(1301)
    mine_kill = bytes([1]) + pack16(9) + bytes([0]) + pack16(65530 + 2)
    for payload in (tank_kill, mine_kill):
        message = decode_deactivation(payload)
        assert encode_message_payload(message) == payload
    mine_message = decode_deactivation(mine_kill)
    assert mine_message["msg_type"] == 0x41
    assert mine_message["is_mine_kill"] is True


def test_sync_roundtrip() -> None:
    """0x3F Sync re-encodes the constant 0x01 wire body."""
    assert encode_message_payload(decode_sync(bytes([1]))) == bytes([1])


def test_cache_update_roundtrip_with_equipment_sentinel() -> None:
    """0x43 CacheUpdate restores 0xFFFF for the -1 equipment sentinel."""
    payload = bytes([10, 20]) + pack16(120) + bytes([11, 21, 0xFF, 0xFF])
    message = decode_cache_update(payload)
    assert message["msg_type"] == 0x43
    assert message["updates"][1][2] == -1
    assert encode_message_payload(message) == payload


def test_chat_ack_roundtrip() -> None:
    """The plaintext 0x43 chat ack round-trips its raw two-byte body."""
    for raw in (b"C0", b"C1"):
        message = try_decode_plaintext_ack(raw)
        if message is None:
            raise AssertionError("expected the raw chat ack to decode")
        assert message["msg_type"] == "chat_ack"
        assert message["enabled"] is (raw == b"C1")
        assert encode_plaintext_ack(message) == raw


def test_overlay_and_terrain_update_roundtrip() -> None:
    """0x40 Overlay and 0x4A Terrain re-encode their triple lists."""
    payload = bytes([5, 6, 2, 7, 8, 0])
    assert encode_message_payload(decode_overlay_update(payload)) == payload
    assert encode_message_payload(decode_terrain_update(payload)) == payload


def test_viewport_update_roundtrip_covers_skip_and_sentinels() -> None:
    """0x5A Viewport re-encodes skips, the no-mine nibble, and -1 cache."""
    entry_a = bytes([3]) + bytes([0x00, 0x78, 0x85])
    entry_b = bytes([255, 255, 30]) + bytes([0xFF, 0xFF, 0x85])
    payload = bytes([200, 100]) + entry_a + entry_b
    message = decode_viewport_update(payload)
    assert message["entities"][0]["overlay_value"] == 255
    assert encode_message_payload(message) == payload
    empty = bytes([1, 2])
    assert encode_message_payload(decode_viewport_update(empty)) == empty


def test_viewport_update_roundtrip_real_mine_overlay() -> None:
    """A mine overlay nibble (< 8) survives the round trip verbatim."""
    packed = (150 << 8) | (3 << 4) | 2
    payload = bytes([0, 0, 4]) + bytes([(packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF])
    message = decode_viewport_update(payload)
    assert message["entities"][0]["overlay_value"] == 3
    assert encode_message_payload(message) == payload


def test_supervisor_roundtrip() -> None:
    """0x52 Supervisor re-encodes its three bytes."""
    payload = bytes([1, 0, 6])
    assert encode_message_payload(decode_supervisor(payload)) == payload


def test_supervisor_text_roundtrip() -> None:
    """0x3C SupervisorText re-encodes latin-1 bytes verbatim."""
    payload = "Message: caf\xe9".encode("latin-1")
    assert encode_message_payload(decode_supervisor_text(payload)) == payload


def test_map_data_roundtrip_with_skips_and_tanks() -> None:
    """0x4C MapData re-emits the greedy skip-RLE atlas and tank slots."""
    rle = bytes([10, 255, 45, 255, 255, 0])
    tank = bytes([131, 126]) + pack16(1301) + bytes([2 | (1 << 2) | (9 << 4)])
    payload = pack16(len(rle)) + rle + tank
    message = decode_map_data(payload)
    assert len(message["fuel_dots"]) == 3
    assert message["tanks"][0]["rank"] == 9
    assert encode_message_payload(message) == payload


def test_map_data_roundtrip_empty() -> None:
    """An empty MapData (no dots, no tanks) round-trips."""
    payload = pack16(0)
    assert encode_message_payload(decode_map_data(payload)) == payload


def test_chat_message_roundtrip_all_tail_lengths() -> None:
    """0x4D Chat re-encodes the 3/4/5-byte tail variants."""
    base = pack16(9) + bytes([2])
    for payload in (base, base + bytes([7]), base + bytes([7, 8])):
        assert encode_message_payload(decode_chat_message(payload)) == payload


def test_statistics_roundtrip_long_form() -> None:
    """0x56 Statistics re-encodes the 14-byte long format."""
    payload = (
        pack16(3)
        + bytes([4, 5])
        + (1000).to_bytes(4, "big")
        + pack16(7)
        + (123456).to_bytes(4, "big")
    )
    assert encode_message_payload(decode_statistics(payload)) == payload


def test_promotion_roundtrip() -> None:
    """0x2B Promotion re-encodes both banner variants."""
    for shown in (0, 1):
        payload = bytes([7, shown])
        assert encode_message_payload(decode_promotion(payload)) == payload


def test_build_pickup_and_decoration_roundtrip() -> None:
    """0x42 BuildPickup and 0x4E Decoration re-encode field-for-field."""
    build = pack16(9) + bytes([1, 2, 3, 4, 5, 1, 0])
    assert encode_message_payload(decode_build_pickup(build)) == build
    decoration = pack16(9) + bytes([2, 5])
    assert encode_message_payload(decode_decoration(decoration)) == decoration


def test_active_forces_and_players_roundtrip() -> None:
    """0x2A ActiveForces and 0x2F ActivePlayers re-encode their records."""
    forces = bytes([4, 0, 2, 9])
    assert encode_message_payload(decode_active_forces(forces)) == forces
    players = pack16(9) + bytes([3]) + pack16(11) + bytes([7])
    assert encode_message_payload(decode_active_players(players)) == players


def test_top10_roundtrip() -> None:
    """0x31 Top10 re-encodes the header and variable-length rows."""
    header = bytes([255]) + pack24(4000) + bytes([2])
    row = bytes([1]) + pack24(9000) + bytes([0, 9, 5]) + b"Artax"
    for payload in (header, header + row):
        assert encode_message_payload(decode_top10(payload)) == payload


def test_action_done_ping_and_connection_lost_bodies() -> None:
    """The bodyless heartbeats emit their documented constant bodies."""
    assert encode_message_payload(decode_action_done(bytes([0]))) == bytes([0])
    assert encode_message_payload(decode_ping_response(b"")) == b""
    assert encode_message_payload(decode_connection_lost(b"")) == b""


def test_radar_result_and_enemy_detection_roundtrip() -> None:
    """0x46 RadarResult and 0x48 EnemyDetection re-encode exactly."""
    for found in (0, 1):
        payload = bytes([0, found])
        assert encode_message_payload(decode_radar_result(payload)) == payload
    detection = bytes([12, 55, 1, 9]) + pack16(528)
    assert encode_message_payload(decode_enemy_detection(detection)) == detection


def test_radar_scan_result_roundtrip() -> None:
    """0x4F RadarScanResult re-encodes containers, mines, and clears."""
    containers = pack16(2) + bytes([1, 2]) + pack16(120) + bytes([3, 4, 0xFF, 0xFF])
    mines = bytes([5, 6, 2])
    clears = bytes([7, 8, 255])
    payload = containers + mines + clears
    message = decode_radar_scan_result(payload)
    assert message["containers"][1]["volume"] == -1
    assert len(message["mines"]) == 1
    assert len(message["mine_clears"]) == 1
    assert encode_message_payload(message) == payload


def test_container_pickup_roundtrip_via_envelope() -> None:
    """A multi-record ContainerPickup body re-encodes verbatim."""
    body = bytes([0x43, 98, 166]) + pack16(204) + bytes([99, 167]) + pack16(0)
    message = decode_0x2e_message(body)
    assert message["msg_type"] == "container_pickup"
    assert encode_envelope_body(message) == body


def test_teleport_landed_roundtrip_via_envelope() -> None:
    """The 1-byte TeleportLanded body re-encodes verbatim."""
    body = bytes([0x0C])
    message = decode_0x2e_message(body)
    assert message["msg_type"] == "teleport_landed"
    assert encode_envelope_body(message) == body


def test_mine_detonation_and_placement_roundtrip_via_envelope() -> None:
    """Mine bodies (0x45/0x4B container forms) re-encode verbatim."""
    detonation = bytes([0x45, 54, 170, 55, 170])
    placement = bytes([0x4B, 2]) + pack16(1301) + bytes([3, 42, 161, 41, 160, 42, 162])
    for body in (detonation, placement):
        message = decode_container_message(body)
        assert encode_envelope_body(message) == body


def test_unknown_container_roundtrip_preserves_raw_bytes() -> None:
    """UnknownContainer bodies re-encode as the preserved raw bytes."""
    body = bytes([0x99, 1, 2, 3])
    message = decode_container_message(body)
    assert message["msg_type"] == "unknown_container"
    assert encode_envelope_body(message) == body


def test_envelope_prepends_subtype_for_protocol_messages() -> None:
    """Tunneled protocol messages get their msg_type as the subtype byte."""
    inner = bytes([0, 7, 0, 1, 4]) + pack24(55) + bytes([2, 1]) + pack16(1100)
    body = bytes([0x2E]) + inner
    message = decode_0x2e_message(body)
    assert message["msg_type"] == 0x2E
    assert encode_envelope_body(message) == body


def test_encode_message_payload_rejects_container_only_messages() -> None:
    """Container-only messages have no top-level protocol form."""
    message = decode_0x2e_message(bytes([0x0C]))
    with pytest.raises(EncodeError):
        encode_message_payload(message)


def test_autoscroll_ack_roundtrip() -> None:
    """The plaintext 0x41 autoscroll ack round-trips its raw two-byte body.

    Wire truth from the 2026-07-24 key-probe capture: the ack is the
    server's un-XORed echo of the plaintext toggle — raw ``4130``
    (``"A0"``), never a binary flag byte.
    """
    for raw in (b"A0", b"A1"):
        message = try_decode_plaintext_ack(raw)
        if message is None:
            raise AssertionError("expected the raw autoscroll ack to decode")
        assert message["msg_type"] == "autoscroll_ack"
        assert message["enabled"] is (raw == b"A1")
        assert encode_plaintext_ack(message) == raw
