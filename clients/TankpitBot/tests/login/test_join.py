"""Tests for join_room, auto_join_room, and ensure_on_play_page."""

from __future__ import annotations

import base64
from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from tankpit_bot import _test_hooks
from tankpit_bot.browser.cdp_helpers import decode_captured_body, load_tpclient_static_key
from tankpit_bot.browser.login import (
    _collect_room_entries,
    _has_enter_response,
    _has_join_confirm,
    _register_room_entries,
    _resolve_room_id,
    _wait_for_enter_response,
    _wait_for_join_confirm,
    _wait_for_room_id,
    ensure_on_play_page,
    handle_login_flow,
    join_room,
)
from tankpit_bot.protocol.framing import encode_frame
from tankpit_bot.sniffer import world_state
from tests.login.conftest import (
    FakeCDPLogin,
    FakeCDPLoginNonDictResult,
    FakePageLogin,
)

# =============================================================================
# Tests for ensure_on_play_page
# =============================================================================


def test_ensure_on_play_page_already_there() -> None:
    """No navigation when already on play page."""
    page = FakePageLogin(start_url="https://tankpit.com/play")

    ensure_on_play_page(page)

    assert page.url == "https://tankpit.com/play"


def test_ensure_on_play_page_navigates() -> None:
    """Navigates to play page when on different page."""
    page = FakePageLogin(start_url="https://tankpit.com/")

    ensure_on_play_page(page)

    assert page.url == "https://tankpit.com/play"


# =============================================================================
# Tests for join_room
# =============================================================================


def test_join_room_success() -> None:
    """Join room succeeds via protocol SELECT plus enter packet."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = join_room(page, cdp)

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True
    assert cdp.entered_room_id == "1"


def test_join_room_returns_false_when_target_room_missing() -> None:
    """Join room fails when the configured room never appears in ROOM_LIST."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(include_practice_room=False)

    result = join_room(page, cdp)

    assert result is False
    assert cdp.join_room_called is False
    assert cdp.enter_room_called is False


def test_join_room_non_dict_result() -> None:
    """Join room rejects malformed raw-message snapshots."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLoginNonDictResult()

    with pytest.raises(JSONTypeError, match="result"):
        join_room(page, cdp)


class _RawMessageCDP:
    """CDP fake that only exposes captured raw websocket messages."""

    def __init__(self, payloads: list[str]) -> None:
        """Initialize fake raw-message source."""
        payload_values: list[JSONValue] = []
        payload_values.extend(payloads)
        self._payloads = payload_values

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return the synthetic raw-message snapshot."""
        _ = (method, params)
        return {"result": {"value": self._payloads}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Ignore event registration in tests."""
        _ = (event, handler)

    def detach(self) -> None:
        """Ignore detach in tests."""


class _MissingMagicCDP(FakeCDPLogin):
    """Login fake that withholds tankpit.magic from the runtime."""

    def _handle_metadata_evaluate(self, expression: str) -> JSONObject | None:
        """Return an empty magic string while preserving other metadata."""
        if "tankpit.magic" in expression:
            return {"result": {"value": ""}}
        return super()._handle_metadata_evaluate(expression)


class _MissingTpclientUrlCDP(FakeCDPLogin):
    """Login fake that withholds the loaded tpclient URL."""

    def _handle_metadata_evaluate(self, expression: str) -> JSONObject | None:
        """Return an empty tpclient URL while preserving other metadata."""
        if "script[src]" in expression and "tpclient" in expression:
            return {"result": {"value": ""}}
        return super()._handle_metadata_evaluate(expression)


class _MissingStaticKeyCDP(FakeCDPLogin):
    """Login fake whose fetched tpclient source lacks the static key."""

    def _handle_metadata_evaluate(self, expression: str) -> JSONObject | None:
        """Return keyless tpclient source for static-key extraction tests."""
        if "fetch(" in expression and "tpclient-test.js" in expression:
            return {"result": {"value": "window.fakeTpclientKey='missing';"}}
        return super()._handle_metadata_evaluate(expression)


def _frame_payload(body: bytes) -> str:
    """Encode one framed raw-message payload for login helper tests."""
    return base64.b64encode(encode_frame(body)).decode("utf-8")


def test_decode_captured_body_rejects_trailing_bytes() -> None:
    """Captured body decode rejects trailing bytes after a framed payload."""
    payload = base64.b64encode(encode_frame(b"+1|Practice") + b"\x00").decode("utf-8")

    with pytest.raises(ValueError, match="unexpected trailing bytes"):
        decode_captured_body(payload)


def test_collect_room_entries_skips_non_room_messages_and_short_entries() -> None:
    """Room collection ignores non-ROOM_LIST payloads and malformed entries."""
    cdp = _RawMessageCDP(
        [
            _frame_payload(b"=1|date|Artax|4"),
            _frame_payload(b"+9"),
            _frame_payload(b"+1|Practice|1"),
            _frame_payload(b"+1|2|118|101|not-a-room"),
            _frame_payload(b"+1|Practice|1|0,0,0,0,0,0,0|1|p|field01.gif|2026"),
        ]
    )

    entries = _collect_room_entries(cdp)

    assert len(entries) == 1
    assert entries[0]["room_id"] == "1"
    assert entries[0]["name"] == "Practice"
    assert entries[0]["image"] == "field01.gif"


def test_register_room_entries_skips_missing_images() -> None:
    """Room registration stores the field image from ROOM_LIST data."""
    world_state.reset_world_state()

    _register_room_entries(
        [
            {
                "room_id": "1",
                "name": "Practice",
                "field_id": 1,
                "game_modes": "0,0,0,0,0,0,0",
                "default_troop": 2,
                "mode_code": "p",
                "image": "field01.gif",
                "year": "2026",
            }
        ]
    )

    assert world_state._room_images["1"] == "field01.gif"


def test_resolve_room_id_supports_prefix_match() -> None:
    """Room resolution accepts stable prefix matches for renamed world rooms."""
    room_id = _resolve_room_id(
        [
            {
                "room_id": "4",
                "name": "World (President Trump)",
                "field_id": 24,
                "game_modes": "5,1,0,0,0,0,0",
                "default_troop": 2,
                "mode_code": "n",
                "image": "field24.gif",
                "year": "2026",
            }
        ],
        "World",
    )

    assert room_id == "4"


def test_resolve_room_id_supports_exact_match() -> None:
    """Room resolution returns the exact room ID on exact name matches."""
    room_id = _resolve_room_id(
        [
            {
                "room_id": "1",
                "name": "Practice",
                "field_id": 1,
                "game_modes": "0,0,0,0,0,0,0",
                "default_troop": 2,
                "mode_code": "p",
                "image": "field01.gif",
                "year": "2026",
            }
        ],
        "Practice",
    )

    assert room_id == "1"


def test_resolve_room_id_returns_none_when_room_is_missing() -> None:
    """Room resolution returns None when no room matches the target name."""
    room_id = _resolve_room_id(
        [
            {
                "room_id": "4",
                "name": "World (President Trump)",
                "field_id": 24,
                "game_modes": "5,1,0,0,0,0,0",
                "default_troop": 2,
                "mode_code": "n",
                "image": "field24.gif",
                "year": "2026",
            }
        ],
        "Practice",
    )

    assert room_id is None


def test_has_join_confirm_ignores_other_rooms_before_match() -> None:
    """Join confirm matching skips unrelated room confirmations."""
    cdp = _RawMessageCDP(
        [
            _frame_payload(b"=4|date|Artax|4"),
            _frame_payload(b"=1|date|Artax|4"),
        ]
    )

    assert _has_join_confirm(cdp, "1") is True


def test_has_join_confirm_respects_start_index() -> None:
    """Join confirm matching can ignore stale confirms captured earlier."""
    cdp = _RawMessageCDP(
        [
            _frame_payload(b"=1|date|Artax|4"),
            _frame_payload(b"=4|date|Artax|4"),
        ]
    )

    assert _has_join_confirm(cdp, "1", start_index=1) is False


def test_has_join_confirm_skips_non_join_messages() -> None:
    """Join confirm matching ignores non-'=' messages in the capture slice."""
    cdp = _RawMessageCDP(
        [
            _frame_payload(b"$1|0"),
            _frame_payload(b"=1|date|Artax|4"),
        ]
    )

    assert _has_join_confirm(cdp, "1") is True


def test_has_enter_response_ignores_other_rooms_before_match() -> None:
    """Enter-response matching skips unrelated room responses."""
    cdp = _RawMessageCDP(
        [
            _frame_payload(b"$4|0"),
            _frame_payload(b"$1|0"),
        ]
    )

    assert _has_enter_response(cdp, "1") is True


def test_has_enter_response_matches_target_room() -> None:
    """Enter-response matching returns True on a direct room match."""
    cdp = _RawMessageCDP([_frame_payload(b"$1|0")])

    assert _has_enter_response(cdp, "1") is True


def test_has_enter_response_skips_non_status_messages() -> None:
    """Enter-response matching ignores non-'$' payloads before a real response."""
    cdp = _RawMessageCDP(
        [
            _frame_payload(b"=1|Sep. 25, 2012|Artax|4|9|9|9|9"),
            _frame_payload(b"$1|0"),
        ]
    )

    assert _has_enter_response(cdp, "1") is True


def test_has_enter_response_respects_start_index() -> None:
    """Enter-response matching can ignore stale responses captured earlier."""
    cdp = _RawMessageCDP(
        [
            _frame_payload(b"$1|0"),
            _frame_payload(b"$4|0"),
        ]
    )

    assert _has_enter_response(cdp, "1", start_index=1) is False


def test_wait_for_join_confirm_times_out_without_match() -> None:
    """Join confirm polling returns False when the target room never confirms."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = _RawMessageCDP([_frame_payload(b"=4|date|Artax|4")])

    assert _wait_for_join_confirm(page, cdp, "1") is False


def test_wait_for_enter_response_times_out_without_match() -> None:
    """Enter-response polling returns False when the target room never responds."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = _RawMessageCDP([_frame_payload(b"$4|0")])

    assert _wait_for_enter_response(page, cdp, "1") is False


def test_wait_for_enter_response_returns_true_when_match_arrives() -> None:
    """Enter-response polling succeeds when the target room responds."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = _RawMessageCDP([_frame_payload(b"$1|0")])

    assert _wait_for_enter_response(page, cdp, "1") is True


def test_wait_for_room_id_returns_room_id_when_found() -> None:
    """Room-ID waiting unwraps the resolved room entry."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    assert _wait_for_room_id(page, cdp, "Practice") == "1"


def test_wait_for_room_id_returns_none_when_missing() -> None:
    """Room-ID waiting returns None when the target room never appears."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(include_practice_room=False)

    assert _wait_for_room_id(page, cdp, "Practice") is None


def test_load_tpclient_static_key_extracts_current_key() -> None:
    """Static-key extraction returns the 1000-char client key."""
    cdp = FakeCDPLogin()

    static_key = load_tpclient_static_key(cdp, "https://tankpit.com/game/tpclient-test.js")

    assert static_key == "A" * 1000


def test_load_tpclient_static_key_raises_when_missing() -> None:
    """Static-key extraction rejects tpclient source without the real key."""
    cdp = _MissingStaticKeyCDP()

    with pytest.raises(ValueError, match="static key was not found"):
        load_tpclient_static_key(cdp, "https://tankpit.com/game/tpclient-test.js")


def test_join_room_returns_false_when_select_send_fails() -> None:
    """Join room stops immediately when the SELECT send helper fails."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(select_send_result="NO_WEBSOCKET")

    result = join_room(page, cdp)

    assert result is False
    assert cdp.enter_room_called is False


def test_join_room_returns_false_when_join_confirm_times_out() -> None:
    """Join room fails when the server never confirms the selected room."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(emit_join_confirm=False)

    result = join_room(page, cdp)

    assert result is False
    assert cdp.enter_room_called is False


def test_join_room_returns_false_when_enter_response_times_out() -> None:
    """Join room fails when the server never acknowledges room entry."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(emit_enter_response=False)

    result = join_room(page, cdp)

    assert result is False
    assert cdp.enter_room_called is True


def test_join_room_returns_false_when_enter_send_fails() -> None:
    """Join room fails when the protocol enter send helper rejects the packet."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(enter_send_result="NO_WEBSOCKET")

    result = join_room(page, cdp)

    assert result is False
    assert cdp.enter_room_called is True


def test_join_room_returns_false_when_magic_missing() -> None:
    """Join room fails when tankpit.magic is unavailable after confirmation."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = _MissingMagicCDP()

    result = join_room(page, cdp)

    assert result is False
    assert cdp.enter_room_called is False


def test_join_room_returns_false_when_tpclient_url_missing() -> None:
    """Join room fails when the loaded tpclient URL is unavailable."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = _MissingTpclientUrlCDP()

    result = join_room(page, cdp)

    assert result is False
    assert cdp.enter_room_called is False


# =============================================================================
# Tests for auto_join_room parameter
# =============================================================================


def test_handle_login_flow_auto_join_room_not_on_before_playing() -> None:
    """Login flow auto-joins room when not on before-playing page."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, auto_join_room=True)

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True


def test_handle_login_flow_auto_join_room_after_guest_login() -> None:
    """Login flow auto-joins room after successful guest login."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, auto_join_room=True)

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True


def test_handle_login_flow_auto_join_room_after_account_login() -> None:
    """Login flow auto-joins room after successful account login."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    original_get_env = _test_hooks.get_env
    env_vars = {"TANKPIT_USERNAME": "testuser", "TANKPIT_PASSWORD": "testpass"}

    def fake_get_env(key: str) -> str | None:
        return env_vars.get(key)

    _test_hooks.get_env = fake_get_env
    try:
        result = handle_login_flow(page, cdp, prefer_account=True, auto_join_room=True)
    finally:
        _test_hooks.get_env = original_get_env

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True


def test_handle_login_flow_auto_join_room_calls_join() -> None:
    """Login flow auto-joins room when enabled."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, auto_join_room=True)

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True


def test_handle_login_flow_no_auto_join_room() -> None:
    """Login flow does not auto-join room when disabled."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, auto_join_room=False)

    assert result is True
    assert cdp.join_room_called is False


def test_handle_login_flow_auto_join_after_rate_limit_fallback() -> None:
    """Login flow auto-joins room after rate-limited account fallback."""
    page = FakePageLogin(
        start_url="https://tankpit.com/before-playing",
    )
    cdp = FakeCDPLogin(rate_limited=True)

    original_get_env = _test_hooks.get_env
    env_vars = {"TANKPIT_USERNAME": "testuser", "TANKPIT_PASSWORD": "testpass"}

    def fake_get_env(key: str) -> str | None:
        return env_vars.get(key)

    _test_hooks.get_env = fake_get_env
    try:
        result = handle_login_flow(page, cdp, auto_join_room=True)
    finally:
        _test_hooks.get_env = original_get_env

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True


def test_handle_login_flow_auto_join_after_guest_failure_no_rate_limit() -> None:
    """Login flow auto-joins room after guest failure (not rate limited)."""
    page = FakePageLogin(
        start_url="https://tankpit.com/before-playing",
        stays_on_before_playing=True,
    )
    cdp = FakeCDPLogin(rate_limited=False)

    result = handle_login_flow(page, cdp, allow_account_fallback=False, auto_join_room=True)

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.enter_room_called is True


def test_handle_login_flow_returns_false_when_auto_join_fails() -> None:
    """Login flow propagates room-join failure instead of continuing."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(include_practice_room=False)

    result = handle_login_flow(page, cdp, auto_join_room=True)

    assert result is False
    assert cdp.join_room_called is False
    assert cdp.enter_room_called is False
