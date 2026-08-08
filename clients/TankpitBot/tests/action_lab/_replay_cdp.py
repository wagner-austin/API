"""Replay CDP doubles.

The stub CDP session that answers page-client snapshot captures, and
the variant that derives its answer from live world state.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
)

from tankpit_bot.sniffer.world_service import WorldService


def build_world_derived_snapshot(ws: WorldService) -> JSONObject:
    """Build a page-client snapshot from ``ws``'s world state.

    No browser is running during replay. The :class:`WorldStateDerivedCDP`
    therefore answers snapshot queries with a deterministic projection of
    what the session's own world service already knows. Field semantics
    mirror the live snapshot but every value here is reachable from that
    one service.

    Args:
        ws: The session's world service -- the projection's only source.

    Returns:
        JSON-shaped page-client snapshot payload.
    """
    world = ws.get_world_state()
    self_state = world["self_state"]
    self_fields: dict[str, JSONValue] = {}
    if self_state is not None:
        self_fields["x"] = self_state["x"]
        self_fields["y"] = self_state["y"]
        self_fields["fuel"] = self_state["fuel"]
    return {
        "timestamp_ms": world["timestamp_ms"],
        "client_present": True,
        "map_visible": False,
        "client_state": 0,
        "client_busy": False,
        "pending_actions": 0,
        "heartbeat_age_ms": 0,
        "last_page_client_send_age_ms": 0,
        "last_bot_send_age_ms": 0,
        "ws_ready_state": 1,
        "current_send_label": None,
        "sent_frame_meta_queue_length": 0,
        "self_fields": self_fields,
        "world_fields": {},
        "map_fields": {},
        "world_collections": {},
    }


_DEFAULT_PAGE_CLIENT_SNAPSHOT_VALUE: JSONObject = {
    "timestamp_ms": 1000,
    "client_present": True,
    "map_visible": False,
    "client_state": 13,
    "client_busy": False,
    "pending_actions": 0,
    "heartbeat_age_ms": 10,
    "last_page_client_send_age_ms": 20,
    "last_bot_send_age_ms": 5,
    "ws_ready_state": 1,
    "current_send_label": None,
    "sent_frame_meta_queue_length": 0,
    "self_fields": {},
    "world_fields": {},
    "map_fields": {},
    "world_collections": {},
}


class StubSnapshotCDPSession:
    """CDPSessionProtocol that returns a fixed snapshot on ``Runtime.evaluate``.

    Other CDP methods return an empty dict; ``on`` and ``detach`` are
    no-ops. Used by tests whose probe bootstrap or attempt body is
    fully stubbed (the CDP session is only present to satisfy the
    type system, never read by production logic).

    When ``snapshot`` is omitted, returns the shared
    :data:`_DEFAULT_PAGE_CLIENT_SNAPSHOT_VALUE` -- one constant, not
    eight forked copies.
    """

    def __init__(self, snapshot: JSONObject | None = None) -> None:
        """Initialize with an optional snapshot value.

        Args:
            snapshot: ``Runtime.evaluate`` response payload. When
                ``None``, the shared identity snapshot is returned.
        """
        self._snapshot = snapshot if snapshot is not None else _DEFAULT_PAGE_CLIENT_SNAPSHOT_VALUE

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return the canned snapshot for ``Runtime.evaluate``.

        Args:
            method: CDP method name.
            params: CDP method params (ignored).

        Returns:
            ``{"result": {"value": snapshot}}`` for ``Runtime.evaluate``;
            an empty dict for every other method.
        """
        _ = params
        if method == "Runtime.evaluate":
            return {"result": {"value": self._snapshot}}
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """No-op event handler registration."""
        _ = (event, handler)

    def detach(self) -> None:
        """No-op CDP session detach."""


class WorldStateDerivedCDP:
    """CDP substitute that derives ``Runtime.evaluate`` results from world state.

    The harness routes every CDP call through this class. Snapshot
    queries return a payload built from the session's world service
    (see :func:`build_world_derived_snapshot`); all other CDP methods
    are no-ops. The CDP substitute carries no behavior of its own --
    it is a pure projection of that one service's truth.
    """

    def __init__(self, ws: WorldService) -> None:
        """Bind the world service this substitute projects.

        Args:
            ws: The session's world service.
        """
        self._ws = ws

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Service a CDP command.

        Args:
            method: CDP method name (only ``Runtime.evaluate`` is honored).
            params: Optional method params.

        Returns:
            Snapshot payload for snapshot queries; string for WebSocket
            send evaluations; otherwise an empty evaluate response.
        """
        if method == "Runtime.evaluate" and params is not None:
            expression = params.get("expression", "")
            if isinstance(expression, str) and "ws.send" in expression:
                return {"result": {"value": "SENT_REPLAY_BYTES"}}
            return {"result": {"value": build_world_derived_snapshot(self._ws)}}
        return {"result": {"value": None}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """No-op event subscription."""
        _ = (event, handler)

    def detach(self) -> None:
        """No-op detach."""
        return
