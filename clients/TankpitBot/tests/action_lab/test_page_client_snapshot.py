"""Tests for the lifted page-client snapshot module.

Covers the universal :class:`PageClientSnapshotDict`, its encode/decode
round-trip, every strict-validation failure mode of the decoder, and the
``capture_page_client_snapshot`` CDP integration. The tests are
fake-driven (no Playwright), exercise every branch the decoder can
encounter, and guarantee the strict-typed contract the rest of the
action lab now depends on.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab.page_client_snapshot import (
    PageClientSnapshotDict,
    capture_page_client_snapshot,
    decode_page_client_snapshot,
    encode_page_client_snapshot,
)


class _FakeCDPSession:
    """Minimal CDPSessionProtocol implementation returning a fixed value.

    Returns the configured ``value`` from every ``Runtime.evaluate``
    call, ignoring all other CDP methods. Used to exercise the snapshot
    capture function end-to-end without spinning up a real browser.
    """

    def __init__(self, value: JSONObject) -> None:
        """Store the value the fake will return from ``Runtime.evaluate``."""
        self._value = value
        self.last_expression: str | None = None

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return the canned value wrapped in the CDP result envelope.

        Args:
            method: CDP method name (must be Runtime.evaluate).
            params: CDP method parameters including the JS expression.

        Returns:
            ``{"result": {"value": <stored value>}}``.
        """
        assert method == "Runtime.evaluate"
        if params is not None:
            expression_raw = params.get("expression")
            if isinstance(expression_raw, str):
                self.last_expression = expression_raw
        return {"result": {"value": self._value}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """No-op handler registration."""
        _ = (event, handler)

    def detach(self) -> None:
        """No-op detach."""
        return


def _full_snapshot_payload() -> JSONObject:
    """Return a complete, valid snapshot JSON payload used across tests."""
    return {
        "timestamp_ms": 17_000,
        "client_present": True,
        "map_visible": True,
        "client_state": 7,
        "client_busy": False,
        "pending_actions": 2,
        "heartbeat_age_ms": 45,
        "last_page_client_send_age_ms": 12,
        "last_bot_send_age_ms": 8,
        "ws_ready_state": 1,
        "current_send_label": "move",
        "sent_frame_meta_queue_length": 1,
        "self_fields": {"a": 105, "b": 110, "c": 1100, "d": True, "e": "hello"},
        "world_fields": {"x": 0, "y": 1, "z": None},
        "map_fields": {"q": 64},
        "world_collections": {
            "ba": [{"u": 146, "v": 44, "w": True}, {"u": 150, "v": 48, "w": False}],
            "cc": [{"n": "Artax", "z": None}],
        },
    }


def _make_snapshot() -> PageClientSnapshotDict:
    """Return a fully populated typed snapshot value used across tests."""
    return PageClientSnapshotDict(
        timestamp_ms=17_000,
        client_present=True,
        map_visible=True,
        client_state=7,
        client_busy=False,
        pending_actions=2,
        heartbeat_age_ms=45,
        last_page_client_send_age_ms=12,
        last_bot_send_age_ms=8,
        ws_ready_state=1,
        current_send_label="move",
        sent_frame_meta_queue_length=1,
        self_fields={"a": 105, "b": 110, "c": 1100, "d": True, "e": "hello"},
        world_fields={"x": 0, "y": 1, "z": None},
        map_fields={"q": 64},
        world_collections={
            "ba": [{"u": 146, "v": 44, "w": True}, {"u": 150, "v": 48, "w": False}],
            "cc": [{"n": "Artax", "z": None}],
        },
    )


def test_decode_page_client_snapshot_round_trips_a_complete_payload() -> None:
    """Decoding then re-encoding a full payload yields the original JSON."""
    payload = _full_snapshot_payload()

    decoded = decode_page_client_snapshot(payload)
    re_encoded = encode_page_client_snapshot(decoded)

    assert re_encoded == payload


def test_decode_page_client_snapshot_accepts_all_null_metadata_fields() -> None:
    """Optional fields decode to None and round-trip without loss."""
    payload: JSONObject = {
        "timestamp_ms": 17_000,
        "client_present": False,
        "map_visible": None,
        "client_state": None,
        "client_busy": None,
        "pending_actions": None,
        "heartbeat_age_ms": None,
        "last_page_client_send_age_ms": None,
        "last_bot_send_age_ms": None,
        "ws_ready_state": None,
        "current_send_label": None,
        "sent_frame_meta_queue_length": 0,
        "self_fields": {},
        "world_fields": {},
        "map_fields": {},
        "world_collections": {},
    }

    decoded = decode_page_client_snapshot(payload)
    assert decoded["map_visible"] is None
    assert decoded["client_state"] is None
    assert decoded["client_busy"] is None
    assert decoded["pending_actions"] is None
    assert decoded["heartbeat_age_ms"] is None
    assert decoded["last_page_client_send_age_ms"] is None
    assert decoded["last_bot_send_age_ms"] is None
    assert decoded["ws_ready_state"] is None
    assert decoded["current_send_label"] is None
    assert decoded["self_fields"] == {}
    assert decoded["world_fields"] == {}
    assert decoded["map_fields"] == {}
    assert decoded["world_collections"] == {}
    assert encode_page_client_snapshot(decoded) == payload


def test_decode_page_client_snapshot_rejects_missing_world_collections() -> None:
    """A payload without ``world_collections`` raises JSONTypeError."""
    payload = _full_snapshot_payload()
    del payload["world_collections"]

    with pytest.raises(JSONTypeError, match=r"world_collections.*must be an object"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_non_list_collection() -> None:
    """A collection entry that is not a list raises JSONTypeError."""
    payload = _full_snapshot_payload()
    payload["world_collections"] = {"ba": {"u": 1}}

    with pytest.raises(JSONTypeError, match=r"world_collections.*'ba'.*must be a list"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_non_object_collection_item() -> None:
    """A collection item that is not an object raises JSONTypeError."""
    payload = _full_snapshot_payload()
    payload["world_collections"] = {"ba": [42]}

    with pytest.raises(JSONTypeError, match=r"'ba\[0\]' must be an object"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_nested_value_in_collection_item() -> None:
    """A collection item carrying a non-primitive field raises JSONTypeError."""
    payload = _full_snapshot_payload()
    payload["world_collections"] = {"ba": [{"u": [1, 2]}]}

    with pytest.raises(JSONTypeError, match=r"world_collections\.ba\[0\].*'u'.*JSON primitive"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_missing_timestamp() -> None:
    """A missing required integer field raises JSONTypeError."""
    payload = _full_snapshot_payload()
    del payload["timestamp_ms"]

    with pytest.raises(JSONTypeError, match=r"timestamp_ms"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_non_bool_client_present() -> None:
    """A required boolean field rejects non-bool values."""
    payload = _full_snapshot_payload()
    payload["client_present"] = "yes"

    with pytest.raises(JSONTypeError, match=r"client_present"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_string_optional_int() -> None:
    """An optional integer field rejects a string value."""
    payload = _full_snapshot_payload()
    payload["client_state"] = "seven"

    with pytest.raises(JSONTypeError, match=r"client_state"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_int_for_optional_bool() -> None:
    """An optional boolean field rejects an integer value (no widening)."""
    payload = _full_snapshot_payload()
    payload["map_visible"] = 1

    with pytest.raises(JSONTypeError, match=r"map_visible"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_int_for_optional_str() -> None:
    """An optional string field rejects an integer value."""
    payload = _full_snapshot_payload()
    payload["current_send_label"] = 42

    with pytest.raises(JSONTypeError, match=r"current_send_label"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_bool_for_optional_int() -> None:
    """An optional integer field rejects a boolean value (bool is not int here)."""
    payload = _full_snapshot_payload()
    payload["pending_actions"] = True

    with pytest.raises(JSONTypeError, match=r"pending_actions"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_non_object_field_map() -> None:
    """A field map that is not a JSON object fails strict validation."""
    payload = _full_snapshot_payload()
    payload["self_fields"] = [1, 2, 3]

    with pytest.raises(JSONTypeError, match=r"self_fields"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_nested_object_in_field_map() -> None:
    """Field map entries must be JSON primitives, not nested containers."""
    payload = _full_snapshot_payload()
    payload["self_fields"] = {"a": {"nested": "dict"}}

    with pytest.raises(JSONTypeError, match=r"self_fields"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_rejects_list_entry_in_field_map() -> None:
    """Field map entries must be JSON primitives, not lists."""
    payload = _full_snapshot_payload()
    payload["world_fields"] = {"actions": [1, 2]}

    with pytest.raises(JSONTypeError, match=r"world_fields"):
        decode_page_client_snapshot(payload)


def test_decode_page_client_snapshot_accepts_float_in_field_map() -> None:
    """Field maps preserve floats since JS numbers can be non-integer."""
    payload = _full_snapshot_payload()
    payload["self_fields"] = {"angle": 1.5}

    decoded = decode_page_client_snapshot(payload)

    assert decoded["self_fields"] == {"angle": 1.5}


def test_decode_page_client_snapshot_accepts_null_entries_in_field_map() -> None:
    """Field maps may contain ``None`` entries (JS ``null`` survived filtering)."""
    payload = _full_snapshot_payload()
    payload["map_fields"] = {"q": None}

    decoded = decode_page_client_snapshot(payload)

    assert decoded["map_fields"] == {"q": None}


def test_encode_page_client_snapshot_is_invariant_under_round_trip() -> None:
    """encode -> decode round-trips the typed snapshot value exactly."""
    snapshot = _make_snapshot()

    re_decoded = decode_page_client_snapshot(encode_page_client_snapshot(snapshot))

    assert re_decoded == snapshot


def test_capture_page_client_snapshot_decodes_a_validated_cdp_response() -> None:
    """The capture helper drives CDP and returns the validated typed snapshot."""
    cdp: CDPSessionProtocol = _FakeCDPSession(_full_snapshot_payload())

    snapshot = capture_page_client_snapshot(cdp)

    assert snapshot["timestamp_ms"] == 17_000
    assert snapshot["client_present"] is True
    assert snapshot["self_fields"] == {
        "a": 105,
        "b": 110,
        "c": 1100,
        "d": True,
        "e": "hello",
    }


def test_capture_page_client_snapshot_reads_active_game_handle() -> None:
    """The capture JS expression reads ``window.__tankpitActiveGame``."""
    fake = _FakeCDPSession(_full_snapshot_payload())
    cdp: CDPSessionProtocol = fake

    capture_page_client_snapshot(cdp)

    expression = fake.last_expression
    if expression is None:
        pytest.fail("capture_page_client_snapshot did not invoke Runtime.evaluate")
    assert "window.__tankpitActiveGame" in expression
    assert "primitivesOnly" in expression
    assert "self_fields" in expression
    assert "world_fields" in expression
    assert "map_fields" in expression


def test_capture_page_client_snapshot_raises_when_value_field_missing() -> None:
    """A CDP response without a value field fails fast."""

    class _MissingValueCDPSession:
        """CDP fake that returns an empty result object."""

        def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
            """Return ``{"result": {}}`` to trigger the missing-value branch."""
            _ = (method, params)
            return {"result": {}}

        def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
            """No-op handler registration."""
            _ = (event, handler)

        def detach(self) -> None:
            """No-op detach."""
            return

    cdp: CDPSessionProtocol = _MissingValueCDPSession()

    with pytest.raises(ValueError, match="missing value"):
        capture_page_client_snapshot(cdp)


def test_capture_page_client_snapshot_raises_when_value_is_not_an_object() -> None:
    """A CDP response whose value is not a JSON object fails strict decoding."""

    class _ListValueCDPSession:
        """CDP fake that returns a list value instead of an object."""

        def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
            """Return a list value, which is invalid for the snapshot."""
            _ = (method, params)
            return {"result": {"value": [1, 2, 3]}}

        def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
            """No-op handler registration."""
            _ = (event, handler)

        def detach(self) -> None:
            """No-op detach."""
            return

    cdp: CDPSessionProtocol = _ListValueCDPSession()

    with pytest.raises(JSONTypeError, match=r"snapshot"):
        capture_page_client_snapshot(cdp)
