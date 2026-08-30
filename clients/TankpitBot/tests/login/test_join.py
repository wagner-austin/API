"""Tests for join_room, auto_join_room, and ensure_on_play_page.

The room-list decoding that precedes the handshake lives in
``test_room_discovery.py``, split out 2026-08-28 when this module
crossed the 600-line ceiling.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.browser.cdp_helpers import load_tpclient_static_key
from tankpit_bot.browser.login import (
    ensure_on_play_page,
)
from tankpit_bot.browser.room_join import (
    _has_enter_response,
    _has_join_confirm,
    _wait_for_enter_response,
    _wait_for_join_confirm,
    _wait_for_room_id,
    join_room,
    resolve_room_troop,
)
from tankpit_bot.sniffer.world_service import WorldService
from tests.conftest import FakeEnv
from tests.login.conftest import (
    FakeCDPLogin,
    FakeCDPLoginNonDictResult,
    FakePageLogin,
    FakeRawMessageCDP,
    frame_payload,
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

    result = join_room(page, cdp, WorldService())

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True
    assert cdp.entered_room_id == "1"


def test_join_room_first_time_entry_chooses_a_troop() -> None:
    """A ``default_troop == -1`` room still joins: the bot picks a color.

    The 2026-08-13 Arterial failure: an account with no tank on the
    room gets ``-1`` in the lobby entry (the UI answers it with a
    color picker), and the join must substitute a chosen troop in the
    enter request instead of stalling at a picker it never clicks.
    """
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(practice_troop=-1)

    result = join_room(page, cdp, WorldService())

    assert result is True
    assert cdp.enter_room_called is True
    assert cdp.entered_room_id == "1"


def test_join_room_env_troop_overrides_the_account_default(fake_env: FakeEnv) -> None:
    """An explicit ``TANKPIT_TROOP`` picks WHICH of the account's tanks to play.

    Fleet ruling 2026-08-14 (same-team allies): accounts hold one tank
    per color per map, and arterial's account default is orange while
    the fleet plays blue -- the enter request must carry the
    configured color, not the lobby's ``default_troop``.
    """
    fake_env.set("TANKPIT_TROOP", "2")
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(practice_troop=3)

    result = join_room(page, cdp, WorldService())

    assert result is True
    assert cdp.entered_troop == 2


def test_join_room_first_time_entry_honours_env_troop(fake_env: FakeEnv) -> None:
    """A first-time entry (``default_troop == -1``) uses the configured color."""
    fake_env.set("TANKPIT_TROOP", "3")
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(practice_troop=-1)

    result = join_room(page, cdp, WorldService())

    assert result is True
    assert cdp.entered_troop == 3


def test_resolve_room_troop_unset_is_none() -> None:
    """Unset ``TANKPIT_TROOP`` means no explicit color was configured.

    The join flow then follows the account's default tank, or blue on
    a first-time entry -- the resolver only reports what the operator
    asked for.
    """
    assert resolve_room_troop() is None


def test_resolve_room_troop_honours_the_env_selector(fake_env: FakeEnv) -> None:
    """``TANKPIT_TROOP`` selects the color; out-of-range values raise.

    An explicit color OVERRIDES the account's default tank (fleet
    ruling 2026-08-14: accounts hold one tank per color per map, and
    the enter request's troop byte picks which one to play).
    """
    fake_env.set("TANKPIT_TROOP", "3")
    assert resolve_room_troop() == 3

    fake_env.set("TANKPIT_TROOP", "4")
    with pytest.raises(ValueError, match="TANKPIT_TROOP"):
        resolve_room_troop()


def test_join_room_returns_false_when_target_room_missing() -> None:
    """Join room fails when the configured room never appears in ROOM_LIST."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(include_practice_room=False)

    result = join_room(page, cdp, WorldService())

    assert result is False
    assert cdp.join_room_called is False
    assert cdp.enter_room_called is False


def test_join_room_non_dict_result() -> None:
    """Join room rejects malformed raw-message snapshots."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLoginNonDictResult()

    with pytest.raises(JSONTypeError, match="result"):
        join_room(page, cdp, WorldService())


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


def test_has_join_confirm_ignores_other_rooms_before_match() -> None:
    """Join confirm matching skips unrelated room confirmations."""
    cdp = FakeRawMessageCDP(
        [
            frame_payload(b"=4|date|Artax|4"),
            frame_payload(b"=1|date|Artax|4"),
        ]
    )

    assert _has_join_confirm(cdp, "1") is True


def test_has_join_confirm_respects_start_index() -> None:
    """Join confirm matching can ignore stale confirms captured earlier."""
    cdp = FakeRawMessageCDP(
        [
            frame_payload(b"=1|date|Artax|4"),
            frame_payload(b"=4|date|Artax|4"),
        ]
    )

    assert _has_join_confirm(cdp, "1", start_index=1) is False


def test_has_join_confirm_skips_non_join_messages() -> None:
    """Join confirm matching ignores non-'=' messages in the capture slice."""
    cdp = FakeRawMessageCDP(
        [
            frame_payload(b"$1|0"),
            frame_payload(b"=1|date|Artax|4"),
        ]
    )

    assert _has_join_confirm(cdp, "1") is True


def test_has_enter_response_ignores_other_rooms_before_match() -> None:
    """Enter-response matching skips unrelated room responses."""
    cdp = FakeRawMessageCDP(
        [
            frame_payload(b"$4|0"),
            frame_payload(b"$1|0"),
        ]
    )

    assert _has_enter_response(cdp, "1") is True


def test_has_enter_response_matches_target_room() -> None:
    """Enter-response matching returns True on a direct room match."""
    cdp = FakeRawMessageCDP([frame_payload(b"$1|0")])

    assert _has_enter_response(cdp, "1") is True


def test_has_enter_response_skips_non_status_messages() -> None:
    """Enter-response matching ignores non-'$' payloads before a real response."""
    cdp = FakeRawMessageCDP(
        [
            frame_payload(b"=1|Sep. 25, 2012|Artax|4|9|9|9|9"),
            frame_payload(b"$1|0"),
        ]
    )

    assert _has_enter_response(cdp, "1") is True


def test_has_enter_response_respects_start_index() -> None:
    """Enter-response matching can ignore stale responses captured earlier."""
    cdp = FakeRawMessageCDP(
        [
            frame_payload(b"$1|0"),
            frame_payload(b"$4|0"),
        ]
    )

    assert _has_enter_response(cdp, "1", start_index=1) is False


def test_wait_for_join_confirm_times_out_without_match() -> None:
    """Join confirm polling returns False when the target room never confirms."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeRawMessageCDP([frame_payload(b"=4|date|Artax|4")])

    assert _wait_for_join_confirm(page, cdp, "1") is False


def test_wait_for_enter_response_times_out_without_match() -> None:
    """Enter-response polling returns False when the target room never responds."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeRawMessageCDP([frame_payload(b"$4|0")])

    assert _wait_for_enter_response(page, cdp, "1") is False


def test_wait_for_enter_response_returns_true_when_match_arrives() -> None:
    """Enter-response polling succeeds when the target room responds."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeRawMessageCDP([frame_payload(b"$1|0")])

    assert _wait_for_enter_response(page, cdp, "1") is True


def test_wait_for_room_id_returns_room_id_when_found() -> None:
    """Room-ID waiting unwraps the resolved room entry."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    assert _wait_for_room_id(page, cdp, WorldService(), "Practice") == "1"


def test_wait_for_room_id_returns_none_when_missing() -> None:
    """Room-ID waiting returns None when the target room never appears."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(include_practice_room=False)

    assert _wait_for_room_id(page, cdp, WorldService(), "Practice") is None


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

    result = join_room(page, cdp, WorldService())

    assert result is False
    assert cdp.enter_room_called is False


def test_join_room_returns_false_when_join_confirm_times_out() -> None:
    """Join room fails when the server never confirms the selected room."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(emit_join_confirm=False)

    result = join_room(page, cdp, WorldService())

    assert result is False
    assert cdp.enter_room_called is False


def test_join_room_returns_false_when_enter_response_times_out() -> None:
    """Join room fails when the server never acknowledges room entry."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(emit_enter_response=False)

    result = join_room(page, cdp, WorldService())

    assert result is False
    assert cdp.enter_room_called is True


def test_join_room_returns_false_when_enter_send_fails() -> None:
    """Join room fails when the protocol enter send helper rejects the packet."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(enter_send_result="NO_WEBSOCKET")

    result = join_room(page, cdp, WorldService())

    assert result is False
    assert cdp.enter_room_called is True


def test_join_room_returns_false_when_magic_missing() -> None:
    """Join room fails when tankpit.magic is unavailable after confirmation."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = _MissingMagicCDP()

    result = join_room(page, cdp, WorldService())

    assert result is False
    assert cdp.enter_room_called is False


def test_join_room_returns_false_when_tpclient_url_missing() -> None:
    """Join room fails when the loaded tpclient URL is unavailable."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = _MissingTpclientUrlCDP()

    result = join_room(page, cdp, WorldService())

    assert result is False
    assert cdp.enter_room_called is False
