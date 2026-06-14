"""Universal page-client snapshot shared by all live action probes.

Reads ``window.__tankpitActiveGame`` -- the live tankpit JS client object
captured by the browser inject script in ``browser/session.py`` -- so any
probe can verify post-action state from the same source the canvas reads
from. The 12 fields here were previously teleport-only; lifting them
across probes lets movement, fuel, equipment, and enemy-teleport probes
all answer "did the bot actually accomplish the action" without watching
the screen.

Architecture notes:
* ``PageClientSnapshotDict`` carries only the universal client metadata.
  Teleport probes need a per-attempt ``phase`` label and so wrap this
  type with ``TeleportPageSnapshotDict`` in ``types.py``.
* Capture is a single CDP ``Runtime.evaluate`` call returning a strict
  JSON object that the decoder validates field-by-field.
* No fallback: a missing or malformed snapshot raises immediately.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_dict,
    require_int,
)

from tankpit_bot._test_hooks import CDPSessionProtocol


class PageClientSnapshotDict(TypedDict):
    """Observed page-client state at one instant in time.

    Attributes:
        timestamp_ms: Wall-clock timestamp when the snapshot was captured.
        client_present: Whether the inject script has captured the game
            object on ``window.__tankpitActiveGame``.
        map_visible: Whether the client believes the map overlay is open
            (``activeGame.map.h``). ``None`` when the map object is
            unavailable.
        client_state: Internal page-client action state identifier
            (``activeGame.s``).
        client_busy: Whether the page client marks itself busy
            (``activeGame.Ha``).
        pending_actions: Length of the client action queue
            (``activeGame.h.j.actions``).
        heartbeat_age_ms: Milliseconds since the most recent server
            heartbeat seen by the client transport (``activeGame.va.j``).
        last_page_client_send_age_ms: Milliseconds since the page client
            itself sent a frame (browser-hook record).
        last_bot_send_age_ms: Milliseconds since the bot's injected
            ``_send_bytes`` ran (browser-hook record).
        ws_ready_state: Browser WebSocket ``readyState`` for the
            captured socket.
        current_send_label: Bot send label currently active in the
            browser hook, used to correlate sends to attempt phases.
        sent_frame_meta_queue_length: Pending outbound metadata queue
            length on the page side.
        self_fields: Primitive properties of ``activeGame.i`` (the
            self-tank object), keyed by minified name; primitives of
            its direct child objects appear under dotted keys
            (``h.j``). Flat-only capture proved blind to the fields
            that matter -- run 20260611-005x held every flat tank field
            static through a wire-confirmed damage countdown -- so one
            nested level is captured, bounded by a per-object field
            cap. Empty when the client object is not yet captured. The
            semantic-to-minified mapping is identified offline by
            comparing values to a known-good world state.
        world_fields: Primitive properties of ``activeGame.h`` (the
            world object) keyed by minified name, including one nested
            level under dotted keys. Empty when the client object is
            not yet captured.
        map_fields: Primitive properties of ``activeGame.map`` (the
            map object) keyed by minified name, including one nested
            level under dotted keys. Empty when the map object is
            unavailable.
        world_collections: Every collection-like property of the world
            object and of its direct children -- arrays of objects AND
            keyed objects whose values are objects. Depth-two entries
            use dotted keys (the live tank registry is
            ``activeGame.P.j``, keyed by tank id, found by the
            structure survey in run 20260610-223x), with each item
            reduced to its primitive fields plus one nested level under
            dotted keys. This is the truth side for entity
            (container/tank/mine) divergence detection: the semantic
            meaning of each collection key is identified offline by
            matching item coordinate pairs against the bot's
            wire-derived world state. Items are capped per collection
            and fields per item to bound capture size; non-object items
            and items with no primitive fields are skipped.
    """

    timestamp_ms: int
    client_present: bool
    map_visible: bool | None
    client_state: int | None
    client_busy: bool | None
    pending_actions: int | None
    heartbeat_age_ms: int | None
    last_page_client_send_age_ms: int | None
    last_bot_send_age_ms: int | None
    ws_ready_state: int | None
    current_send_label: str | None
    sent_frame_meta_queue_length: int
    self_fields: dict[str, int | float | bool | str | None]
    world_fields: dict[str, int | float | bool | str | None]
    map_fields: dict[str, int | float | bool | str | None]
    world_collections: dict[str, list[dict[str, int | float | bool | str | None]]]


_CAPTURE_EXPRESSION = """
(() => {
    const activeGame =
        window.__tankpitActiveGame && typeof window.__tankpitActiveGame === 'object'
            ? window.__tankpitActiveGame
            : null;
    const selfObject =
        activeGame && activeGame.i && typeof activeGame.i === 'object'
            ? activeGame.i
            : null;
    const worldObject =
        activeGame && activeGame.h && typeof activeGame.h === 'object'
            ? activeGame.h
            : null;
    const mapObject =
        activeGame && activeGame.map && typeof activeGame.map === 'object'
            ? activeGame.map
            : null;
    const actionSource =
        worldObject && worldObject.j && typeof worldObject.j === 'object'
            ? worldObject.j
            : null;
    const actions =
        actionSource && Array.isArray(actionSource.actions)
            ? actionSource.actions
            : null;
    const heartbeatSource =
        activeGame && activeGame.va && typeof activeGame.va === 'object'
            ? activeGame.va
            : null;
    const lastHeartbeat =
        heartbeatSource && typeof heartbeatSource.j === 'number'
            ? heartbeatSource.j
            : null;
    const lastPageClientSend =
        typeof window.__lastPageClientSendPerfMs === 'number'
            ? window.__lastPageClientSendPerfMs
            : null;
    const lastBotSend =
        typeof window.__lastBotInjectedSendPerfMs === 'number'
            ? window.__lastBotInjectedSendPerfMs
            : null;
    const ws =
        window.__capturedWS instanceof WebSocket
            ? window.__capturedWS
            : null;
    const queue =
        Array.isArray(window.__sentFrameMetaQueue)
            ? window.__sentFrameMetaQueue
            : [];
    function isPrimitive(value) {
        return (
            value === null ||
            typeof value === 'number' ||
            typeof value === 'boolean' ||
            typeof value === 'string'
        );
    }
    // One nested level is captured with dotted keys: flat-only capture
    // proved blind to the fields that matter (enemy damage is NOT in
    // the tank entries' top-level primitives -- run 20260611-005x held
    // every flat field static through a wire-confirmed 3 -> 2 -> 1 tier
    // countdown, so the truth lives in the entries' child objects).
    // The field cap bounds pathological objects (canvas contexts carry
    // hundreds of properties) without dropping game-entity fields,
    // which run well under it.
    const MAX_FIELDS = 96;
    function primitivesOnly(obj) {
        if (obj === null || typeof obj !== 'object') return {};
        const out = {};
        let count = 0;
        for (const key of Object.keys(obj)) {
            if (count >= MAX_FIELDS) break;
            const value = obj[key];
            if (isPrimitive(value)) {
                out[key] = value;
                count += 1;
                continue;
            }
            if (typeof value !== 'object' || Array.isArray(value)) continue;
            for (const subKey of Object.keys(value)) {
                if (count >= MAX_FIELDS) break;
                const sub = value[subKey];
                if (isPrimitive(sub)) {
                    out[key + '.' + subKey] = sub;
                    count += 1;
                }
            }
        }
        return out;
    }
    function collectionItems(value, maxItems) {
        let entries = null;
        if (Array.isArray(value)) {
            entries = value;
        } else if (value !== null && typeof value === 'object') {
            entries = Object.values(value);
        }
        if (entries === null) return null;
        const items = [];
        for (const item of entries) {
            if (items.length >= maxItems) break;
            if (item === null || typeof item !== 'object' || Array.isArray(item)) continue;
            const fields = primitivesOnly(item);
            if (Object.keys(fields).length === 0) continue;
            items.push(fields);
        }
        return items.length > 0 ? items : null;
    }
    function objectCollections(obj, maxItems) {
        if (obj === null || typeof obj !== 'object') return {};
        const out = {};
        for (const key of Object.keys(obj)) {
            const value = obj[key];
            const items = collectionItems(value, maxItems);
            if (items !== null) out[key] = items;
            // Registries hang one level deeper than the world object
            // (the live tank registry is activeGame.P.j, keyed by tank
            // id), so every child object is also swept -- even when the
            // parent itself looked collection-like, because a mixed
            // parent flattens to junk while its child holds the data.
            if (value === null || typeof value !== 'object' || Array.isArray(value)) continue;
            for (const subKey of Object.keys(value)) {
                const subItems = collectionItems(value[subKey], maxItems);
                if (subItems !== null) out[key + '.' + subKey] = subItems;
            }
        }
        return out;
    }
    const now = performance.now();
    return {
        timestamp_ms: Date.now(),
        client_present: activeGame !== null,
        map_visible: mapObject === null ? null : !!mapObject.h,
        client_state:
            activeGame !== null && typeof activeGame.s === 'number'
                ? activeGame.s
                : null,
        client_busy:
            activeGame !== null && typeof activeGame.Ha === 'boolean'
                ? activeGame.Ha
                : null,
        pending_actions: actions === null ? null : actions.length,
        heartbeat_age_ms:
            lastHeartbeat === null ? null : Math.max(0, Math.floor(now - lastHeartbeat)),
        last_page_client_send_age_ms:
            lastPageClientSend === null
                ? null
                : Math.max(0, Math.floor(now - lastPageClientSend)),
        last_bot_send_age_ms:
            lastBotSend === null
                ? null
                : Math.max(0, Math.floor(now - lastBotSend)),
        ws_ready_state: ws === null ? null : ws.readyState,
        current_send_label:
            typeof window.__codexCurrentSendLabel === 'string'
                ? window.__codexCurrentSendLabel
                : null,
        sent_frame_meta_queue_length: queue.length,
        self_fields: primitivesOnly(selfObject),
        world_fields: primitivesOnly(worldObject),
        map_fields: primitivesOnly(mapObject),
        // Swept from the game ROOT, not activeGame.h: the structure
        // survey (run 20260610-223x) located the live tank registry at
        // activeGame.P.j, a sibling of h that an h-rooted sweep can
        // never reach.
        world_collections: objectCollections(activeGame, 128)
    };
})()
"""


def _extract_runtime_value(result: JSONObject) -> JSONValue:
    """Return the ``Runtime.evaluate`` value field.

    Args:
        result: Raw CDP result object returned by ``cdp.send``.

    Returns:
        The evaluated JavaScript value.

    Raises:
        ValueError: If the CDP result is missing the value field.
    """
    result_obj = require_dict(result, "result")
    if "value" not in result_obj:
        raise ValueError(f"Runtime.evaluate result missing value: {result_obj}")
    return result_obj["value"]


def _require_optional_int(data: JSONObject, field: str) -> int | None:
    """Return an optional integer field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Integer value or None when the field is null.

    Raises:
        JSONTypeError: If the field is present but not an integer.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise JSONTypeError(f"Field '{field}' must be an integer or null")
    return raw


def _require_optional_bool(data: JSONObject, field: str) -> bool | None:
    """Return an optional boolean field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Boolean value or None when the field is null.

    Raises:
        JSONTypeError: If the field is present but not a boolean.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise JSONTypeError(f"Field '{field}' must be a boolean or null")
    return raw


def _require_optional_str(data: JSONObject, field: str) -> str | None:
    """Return an optional string field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        String value or None when the field is null.

    Raises:
        JSONTypeError: If the field is present but not a string.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise JSONTypeError(f"Field '{field}' must be a string or null")
    return raw


def _require_bool_field(data: JSONObject, field: str) -> bool:
    """Return a required boolean field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If the field is not a boolean.
    """
    raw = data.get(field)
    if not isinstance(raw, bool):
        raise JSONTypeError(f"Field '{field}' must be a boolean")
    return raw


def _require_client_field_value(
    value: JSONValue,
    field: str,
    key: str,
) -> int | float | bool | str | None:
    """Validate one entry of a client field map as a JSON primitive.

    Args:
        value: Raw value associated with ``key``.
        field: Outer field name (for error reporting).
        key: Inner key inside the field map (for error reporting).

    Returns:
        Validated primitive value.

    Raises:
        JSONTypeError: If ``value`` is a list or dict (only primitives are
            accepted in field maps; the JS-side primitivesOnly filter is
            the source of truth, this is the strict re-check).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    raise JSONTypeError(
        f"Field '{field}' entry '{key}' must be a JSON primitive (got {type(value).__name__})"
    )


def decode_client_field_map(
    raw: JSONObject,
    *,
    field: str,
) -> dict[str, int | float | bool | str | None]:
    """Decode a raw JSON object as a minified-key field map.

    Args:
        raw: JSON object whose entries are all expected to be primitives.
        field: Outer field name used in error messages.

    Returns:
        Validated mapping of minified key names to primitive scalars.

    Raises:
        JSONTypeError: If any entry holds a non-primitive value.
    """
    result: dict[str, int | float | bool | str | None] = {}
    for key, value in raw.items():
        result[key] = _require_client_field_value(value, field, key)
    return result


def _require_client_field_map(
    data: JSONObject,
    field: str,
) -> dict[str, int | float | bool | str | None]:
    """Decode a discovery field map (minified key -> primitive value).

    Args:
        data: JSON object to inspect.
        field: Field name to read and validate.

    Returns:
        Validated mapping of minified key names to primitive scalars.

    Raises:
        JSONTypeError: If the field is missing, not an object, or contains
            non-primitive values.
    """
    raw = data.get(field)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object")
    return decode_client_field_map(raw, field=field)


def encode_client_field_map(
    field_map: dict[str, int | float | bool | str | None],
) -> JSONObject:
    """Encode a discovery field map as a JSON object.

    Args:
        field_map: Mapping of minified key names to primitive scalars.

    Returns:
        JSON-ready object that preserves insertion order.
    """
    result: JSONObject = {}
    for key, value in field_map.items():
        result[key] = value
    return result


def decode_client_collections(
    raw: JSONObject,
    *,
    field: str,
) -> dict[str, list[dict[str, int | float | bool | str | None]]]:
    """Decode a raw JSON object as a minified-key collection map.

    Args:
        raw: JSON object whose values are lists of primitive field maps.
        field: Outer field name used in error messages.

    Returns:
        Validated mapping of minified property names to item lists.

    Raises:
        JSONTypeError: If any value is not a list, any item is not an
            object, or any item field holds a non-primitive value.
    """
    result: dict[str, list[dict[str, int | float | bool | str | None]]] = {}
    for key, value in raw.items():
        if not isinstance(value, list):
            raise JSONTypeError(f"Field '{field}' entry '{key}' must be a list")
        items: list[dict[str, int | float | bool | str | None]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise JSONTypeError(f"Field '{field}' entry '{key}[{index}]' must be an object")
            items.append(decode_client_field_map(item, field=f"{field}.{key}[{index}]"))
        result[key] = items
    return result


def _require_client_collections(
    data: JSONObject,
    field: str,
) -> dict[str, list[dict[str, int | float | bool | str | None]]]:
    """Decode a required collection map field from a snapshot payload.

    Args:
        data: JSON object to inspect.
        field: Field name to read and validate.

    Returns:
        Validated mapping of minified property names to item lists.

    Raises:
        JSONTypeError: If the field is missing, not an object, or fails
            item validation.
    """
    raw = data.get(field)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object")
    return decode_client_collections(raw, field=field)


def encode_client_collections(
    collections: dict[str, list[dict[str, int | float | bool | str | None]]],
) -> JSONObject:
    """Encode a collection map as a JSON object.

    Args:
        collections: Mapping of minified property names to item lists.

    Returns:
        JSON-ready object that preserves insertion order.
    """
    result: JSONObject = {}
    for key, items in collections.items():
        encoded_items: list[JSONValue] = [encode_client_field_map(item) for item in items]
        result[key] = encoded_items
    return result


def encode_page_client_snapshot(snapshot: PageClientSnapshotDict) -> JSONObject:
    """Encode a universal page-client snapshot to a JSON object.

    Args:
        snapshot: Snapshot to encode.

    Returns:
        JSON-serializable object representation in stable field order.
    """
    map_visible: JSONValue = snapshot["map_visible"]
    client_state: JSONValue = snapshot["client_state"]
    client_busy: JSONValue = snapshot["client_busy"]
    pending_actions: JSONValue = snapshot["pending_actions"]
    heartbeat_age_ms: JSONValue = snapshot["heartbeat_age_ms"]
    last_page_client_send_age_ms: JSONValue = snapshot["last_page_client_send_age_ms"]
    last_bot_send_age_ms: JSONValue = snapshot["last_bot_send_age_ms"]
    ws_ready_state: JSONValue = snapshot["ws_ready_state"]
    current_send_label: JSONValue = snapshot["current_send_label"]
    return {
        "timestamp_ms": snapshot["timestamp_ms"],
        "client_present": snapshot["client_present"],
        "map_visible": map_visible,
        "client_state": client_state,
        "client_busy": client_busy,
        "pending_actions": pending_actions,
        "heartbeat_age_ms": heartbeat_age_ms,
        "last_page_client_send_age_ms": last_page_client_send_age_ms,
        "last_bot_send_age_ms": last_bot_send_age_ms,
        "ws_ready_state": ws_ready_state,
        "current_send_label": current_send_label,
        "sent_frame_meta_queue_length": snapshot["sent_frame_meta_queue_length"],
        "self_fields": encode_client_field_map(snapshot["self_fields"]),
        "world_fields": encode_client_field_map(snapshot["world_fields"]),
        "map_fields": encode_client_field_map(snapshot["map_fields"]),
        "world_collections": encode_client_collections(snapshot["world_collections"]),
    }


def decode_page_client_snapshot(data: JSONObject) -> PageClientSnapshotDict:
    """Decode a universal page-client snapshot from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated typed snapshot.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return PageClientSnapshotDict(
        timestamp_ms=require_int(data, "timestamp_ms"),
        client_present=_require_bool_field(data, "client_present"),
        map_visible=_require_optional_bool(data, "map_visible"),
        client_state=_require_optional_int(data, "client_state"),
        client_busy=_require_optional_bool(data, "client_busy"),
        pending_actions=_require_optional_int(data, "pending_actions"),
        heartbeat_age_ms=_require_optional_int(data, "heartbeat_age_ms"),
        last_page_client_send_age_ms=_require_optional_int(data, "last_page_client_send_age_ms"),
        last_bot_send_age_ms=_require_optional_int(data, "last_bot_send_age_ms"),
        ws_ready_state=_require_optional_int(data, "ws_ready_state"),
        current_send_label=_require_optional_str(data, "current_send_label"),
        sent_frame_meta_queue_length=require_int(data, "sent_frame_meta_queue_length"),
        self_fields=_require_client_field_map(data, "self_fields"),
        world_fields=_require_client_field_map(data, "world_fields"),
        map_fields=_require_client_field_map(data, "map_fields"),
        world_collections=_require_client_collections(data, "world_collections"),
    )


def capture_page_client_snapshot(cdp: CDPSessionProtocol) -> PageClientSnapshotDict:
    """Capture the universal page-client snapshot via CDP ``Runtime.evaluate``.

    Args:
        cdp: Active CDP session attached to the live tankpit page.

    Returns:
        Validated typed snapshot read from ``window.__tankpitActiveGame``.

    Raises:
        ValueError: If the CDP response omits the value field.
        JSONTypeError: If the evaluated payload fails strict decoding.
    """
    result = cdp.send(
        "Runtime.evaluate",
        {"expression": _CAPTURE_EXPRESSION, "returnByValue": True},
    )
    raw_value = _extract_runtime_value(result)
    return decode_page_client_snapshot(require_dict({"snapshot": raw_value}, "snapshot"))


__all__ = [
    "PageClientSnapshotDict",
    "capture_page_client_snapshot",
    "decode_client_collections",
    "decode_client_field_map",
    "decode_page_client_snapshot",
    "encode_client_collections",
    "encode_client_field_map",
    "encode_page_client_snapshot",
]
