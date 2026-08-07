"""Round-trip + validation tests for enemy tracking probe TypedDicts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError
from tests.action_lab._enemy_tracking_harness import (
    _make_js_belief,
    _make_observation,
    _make_our_belief,
    _make_session,
    _make_shot,
    _make_tracked,
)

from tankpit_bot.action_lab.enemy_tracking_codecs import (
    decode_enemy_tracking_probe_session,
    decode_js_tank_belief,
    decode_our_tank_belief,
    decode_shot_event,
    decode_tracked_enemy,
    decode_tracking_observation,
    encode_enemy_tracking_probe_session,
    encode_js_tank_belief,
    encode_our_tank_belief,
    encode_shot_event,
    encode_tracked_enemy,
    encode_tracking_observation,
)


def test_our_tank_belief_roundtrip_preserves_fields() -> None:
    """OurTankBeliefDict encode/decode round-trip preserves every field."""
    belief = _make_our_belief()
    decoded = decode_our_tank_belief(encode_our_tank_belief(belief))
    assert decoded == belief


def test_js_tank_belief_roundtrip_preserves_fields() -> None:
    """JSTankBeliefDict encode/decode round-trip preserves presence + fields."""
    belief = _make_js_belief()
    decoded = decode_js_tank_belief(encode_js_tank_belief(belief))
    assert decoded == belief


def test_js_tank_belief_roundtrip_handles_absent() -> None:
    """JSTankBeliefDict with present=False round-trips with empty fields."""
    belief = _make_js_belief(present=False)
    decoded = decode_js_tank_belief(encode_js_tank_belief(belief))
    assert decoded == belief


def test_tracking_observation_roundtrip_preserves_fields() -> None:
    """TrackingObservationDict encode/decode round-trip preserves every field."""
    observation = _make_observation()
    decoded = decode_tracking_observation(encode_tracking_observation(observation))
    assert decoded == observation


def test_tracked_enemy_roundtrip_preserves_fields() -> None:
    """TrackedEnemyDict encode/decode round-trip preserves every field."""
    tracked = _make_tracked()
    decoded = decode_tracked_enemy(encode_tracked_enemy(tracked))
    assert decoded == tracked


def test_shot_event_roundtrip_preserves_fields() -> None:
    """ShotEventDict encode/decode round-trip preserves every field."""
    shot = _make_shot()
    decoded = decode_shot_event(encode_shot_event(shot))
    assert decoded == shot


def test_session_roundtrip_with_shot_preserves_fields() -> None:
    """Session round-trip preserves every field when a shot fired."""
    session = _make_session(with_shot=True)
    decoded = decode_enemy_tracking_probe_session(
        encode_enemy_tracking_probe_session(session),
    )
    assert decoded == session


def test_session_roundtrip_without_shot_preserves_null() -> None:
    """Session round-trip preserves None when no shot fired."""
    session = _make_session(with_shot=False)
    decoded = decode_enemy_tracking_probe_session(
        encode_enemy_tracking_probe_session(session),
    )
    assert decoded["shot"] is None


def test_our_belief_decode_rejects_non_bool_present() -> None:
    """decode_our_tank_belief rejects a non-boolean ``present``."""
    encoded = encode_our_tank_belief(_make_our_belief())
    encoded["present"] = "yes"
    with pytest.raises(JSONTypeError):
        decode_our_tank_belief(encoded)


def test_js_belief_decode_rejects_non_object_fields() -> None:
    """decode_js_tank_belief rejects a non-object ``fields``."""
    encoded = encode_js_tank_belief(_make_js_belief())
    encoded["fields"] = []
    with pytest.raises(JSONTypeError):
        decode_js_tank_belief(encoded)


def test_observation_decode_rejects_non_object_our_belief() -> None:
    """decode_tracking_observation rejects a non-object nested belief."""
    encoded = encode_tracking_observation(_make_observation())
    encoded["our_belief"] = "oops"
    with pytest.raises(JSONTypeError):
        decode_tracking_observation(encoded)


def test_session_decode_rejects_non_object_shot() -> None:
    """decode_enemy_tracking_probe_session rejects a non-object shot field."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["shot"] = 42
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_list_tracked() -> None:
    """decode_enemy_tracking_probe_session rejects a non-list tracked field."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["tracked"] = {}
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_object_tracked_item() -> None:
    """decode_enemy_tracking_probe_session rejects non-object items in tracked."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["tracked"] = [42]
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_list_observations() -> None:
    """decode_enemy_tracking_probe_session rejects a non-list observations field."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["observations"] = 42
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_object_observation_item() -> None:
    """decode_enemy_tracking_probe_session rejects non-object items in observations."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["observations"] = [42]
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_object_startup_timing() -> None:
    """decode_enemy_tracking_probe_session rejects a non-object startup_timing."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["startup_timing"] = "bad"
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)


def test_session_decode_rejects_non_object_snapshot() -> None:
    """decode_enemy_tracking_probe_session rejects a non-object snapshot field."""
    encoded = encode_enemy_tracking_probe_session(_make_session())
    encoded["snapshot_at_acquisition"] = []
    with pytest.raises(JSONTypeError):
        decode_enemy_tracking_probe_session(encoded)
