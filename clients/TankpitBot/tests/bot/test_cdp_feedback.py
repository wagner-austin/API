"""Tests for protocol-kill merging and combat shot feedback."""

from __future__ import annotations

from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_combat import (
    mark_combat_hit,
    mark_tank_killed,
)
from tankpit_bot.sniffer.world_state_inventory import (
    get_inventory_state,
    update_inventory_from_protocol,
)
from tests.conftest import FakeEnv


class TestBotCombatFeedback:
    """Protocol-kill merging and the combat feedback state machine."""

    def test_merge_protocol_kills_adds_to_ai_state(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_merge_protocol_kills adds Deactivation kills to AI killed_tank_ids."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _merge_protocol_kills
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        mark_tank_killed(get_world_service(), 50)
        mark_tank_killed(get_world_service(), 60)
        new_state = _merge_protocol_kills(bot._ai_state)
        assert "50" in new_state["killed_tank_ids"]
        assert "60" in new_state["killed_tank_ids"]

    def test_merge_protocol_kills_clears_combat_target_keeps_shot_target(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Kill merge clears the combat lock but preserves the shot target.

        The shot-target fields must survive the merge so the combat
        feedback classifier can resolve the kill shot as
        ``kill_confirmed`` -- a kill produces no damage-change
        feedback, so clearing the target here would leave the shot's
        ledger decision pending forever (run 2026-07-19 00:50:37:
        decision 235 unresolved at shutdown).
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _merge_protocol_kills
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "orange-8"
        bot._ai_state["combat_target_id"] = 50
        bot._ai_state["combat_target_x"] = 71
        bot._ai_state["combat_target_y"] = 53

        mark_tank_killed(get_world_service(), 50)
        new_state = _merge_protocol_kills(bot._ai_state)

        assert new_state["last_shot_target_id"] == 50
        assert new_state["last_shot_target_name"] == "orange-8"
        assert new_state["combat_target_id"] == -1
        assert new_state["combat_target_x"] == 0
        assert new_state["combat_target_y"] == 0

    def test_merge_protocol_kills_no_kills(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_merge_protocol_kills returns unchanged state when no kills."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _merge_protocol_kills
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        result = _merge_protocol_kills(bot._ai_state)
        assert result is bot._ai_state

    def test_get_combat_feedback_empty_until_response(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_get_combat_feedback returns '' until a wire ShootEvent arrives.

        Old heuristic guessed 'miss' just because dual was available. The
        tile-occupancy signal requires an actual wire response -- absent
        a response we wait, never invent a miss.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
        )

        reset_world_state()
        update_inventory_from_protocol(
            get_world_service(), [0, 10, 0, 0, 0], [False, True, False, False, False]
        )
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        result = _get_combat_feedback(bot)
        assert result == ""

    def test_get_combat_feedback_no_miss_without_dual(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_get_combat_feedback returns '' when dual shots depleted."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        result = _get_combat_feedback(bot)
        assert result == ""

    def test_get_combat_feedback_hit_when_combat_hit(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_get_combat_feedback returns 'hit' when a 0x53 ShootEvent was received."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state
        from tankpit_bot.sniffer.world_state_combat import mark_combat_hit

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        mark_combat_hit(get_world_service(), weapon_byte=1, victim_id=999)
        result = _get_combat_feedback(bot)
        assert result == "hit"

    def test_get_combat_feedback_hit_when_target_killed(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_get_combat_feedback returns 'hit' when the tracked target was killed.

        The ``kill_confirmed`` branch must also clear the shot-target
        fields itself: its trigger (``killed_tank_ids`` membership) is
        not a consumable wire flag, so without the clear a tick that
        dispatches no command would re-emit the outcome every tick.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["killed_tank_ids"] = {"50": 1000}
        result = _get_combat_feedback(bot)
        assert result == "hit"
        assert bot._ai_state["last_shot_target_id"] == -1
        assert bot._ai_state["last_shot_target_name"] == ""
        second = _get_combat_feedback(bot)
        assert second == ""

    def test_get_combat_feedback_empty_no_shot_pending(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_get_combat_feedback returns '' when no shot was fired."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        result = _get_combat_feedback(bot)
        assert result == ""

    def test_get_combat_feedback_rejected_on_command_error(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A shot-rejecting 0x52 code during a pending shot yields 'rejected'.

        A rejected dispatch produces no ShootEvent and no ammo delta
        (live run 2026-07-03 20:34: five code-0 rejections were
        invisible to the classifier and each burned the 4 s feedback
        window). The error is consumed, the reject counter advances,
        and the outcome is neither a hit nor a miss.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        get_world_service().last_command_error = 0

        result = _get_combat_feedback(bot)

        assert result == "rejected"
        assert bot._ai_state["session_reject_count"] == 1
        assert bot._ai_state["session_hit_count"] == 0
        assert bot._ai_state["session_miss_count"] == 0
        # The error was consumed so nothing else double-handles it.
        assert get_world_service().last_command_error == -1
        # Code 0 (aim geometry) carries no target semantics -- the
        # friendly-fire disproof must NOT fire for it.
        assert bot._ai_state["blocked_combat_targets"] == {}

    def test_get_combat_feedback_friendly_fire_disproves_target(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """An err=3 rejection blocklists the target and releases the lock.

        Session 4 of run 20260730 (20:36): Yuppler left the game and
        the bot fired 43 consecutive rejected shots at his ghost --
        the 0x58 grace keeps the registry entry and every map open
        re-stamps its freshness, so the server's friendly-fire receipt
        is the only truth that the id is not an engageable enemy.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 1229
        bot._ai_state["last_shot_target_name"] = "Yuppler"
        bot._ai_state["combat_target_id"] = 1229
        bot._ai_state["combat_target_x"] = 245
        bot._ai_state["combat_target_y"] = 76
        get_world_service().last_command_error = 3

        result = _get_combat_feedback(bot)

        assert result == "rejected"
        assert "1229" in bot._ai_state["blocked_combat_targets"]
        assert bot._ai_state["combat_target_id"] == -1

    def test_get_combat_feedback_ignores_non_shot_command_error(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A non-shot 0x52 code (e.g. 7 'Inventory full') is left pending.

        Codes outside the shot-rejecting set belong to other action
        machinery (pickup rejections route through
        ``_clear_command_error``); the feedback classifier must not
        consume them.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        get_world_service().last_command_error = 7

        result = _get_combat_feedback(bot)

        assert result == ""
        assert bot._ai_state["session_reject_count"] == 0
        assert get_world_service().last_command_error == 7

    def test_has_pending_shot_feedback_ends_wait_on_command_error(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A shot-rejecting 0x52 code ends the feedback wait immediately.

        Without this the bot idles the full ``shot_feedback_timeout_ms``
        (4 s) on a shot the server already refused.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _has_pending_shot_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = 100000

        assert _has_pending_shot_feedback(bot, 100500) is True

        get_world_service().last_command_error = 0
        assert _has_pending_shot_feedback(bot, 100500) is False

    def test_get_combat_feedback_hit_via_ammo_delta(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A 0x49 sync revealing an ammo debit is a hit even without a 0x53 echo.

        Reconciliation case (the per-shot ``weapon`` byte is the
        primary hit signal since 2026-07-02): if the 0x53 ShootEvent
        echo is lost, the server's 0x49 absolute inventory sync still
        reveals the debit against the pre-shot snapshot -- the shot
        landed, and the feedback must say so instead of timing out
        into a phantom miss.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state
        from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_protocol

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "purple-9"
        ws = get_world_service()
        update_inventory_from_protocol(ws, [25, 25, 25, 25, 25], [False] * 5)
        # Bot dispatched a shoot just now: snapshot pre-shot inventory.
        ws.pending_shot_inventory_snapshot = ws.inventory_state
        # The 0x53 echo never arrives (dropped frame) -- but the 0x49
        # absolute sync debits the homing: authoritative hit.
        update_inventory_from_protocol(ws, [25, 25, 25, 24, 25], [False] * 5)

        result = _get_combat_feedback(bot)
        assert result == "hit"

    def test_has_pending_shot_feedback_true_before_timeout(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_has_pending_shot_feedback waits while a shot is still inside its timeout."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _has_pending_shot_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = 1000

        assert _has_pending_shot_feedback(bot, 2000) is True

    def test_has_pending_shot_feedback_false_after_timeout(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_has_pending_shot_feedback stops waiting once the timeout expires."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _has_pending_shot_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = 1000

        assert _has_pending_shot_feedback(bot, 6000) is False

    def test_has_pending_shot_feedback_false_when_hit_buffered(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_has_pending_shot_feedback yields to feedback when a hit is already buffered."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _has_pending_shot_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state
        from tankpit_bot.sniffer.world_state_combat import mark_combat_hit

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = 1000
        mark_combat_hit(get_world_service(), weapon_byte=1, victim_id=999)

        assert _has_pending_shot_feedback(bot, 2000) is False

    def test_has_pending_shot_feedback_false_when_target_killed(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_has_pending_shot_feedback stops waiting when the target is already dead."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _has_pending_shot_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = 1000
        bot._ai_state["killed_tank_ids"] = {"50": 1500}

        assert _has_pending_shot_feedback(bot, 2000) is False

    def test_has_pending_feedback_false_when_single_shot_response(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_has_pending_shot_feedback ends when weapon_byte=0 response arrives."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _has_pending_shot_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state
        from tankpit_bot.sniffer.world_state_combat import mark_combat_hit

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = 1000
        mark_combat_hit(get_world_service(), weapon_byte=0, victim_id=-1)

        assert _has_pending_shot_feedback(bot, 2000) is False

    def test_feedback_single_shot_with_dual_available_is_miss(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """weapon_byte=0 with dual available is a miss (target was empty)."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
        )

        reset_world_state()
        update_inventory_from_protocol(
            get_world_service(),
            [0, 10, 0, 0, 0],
            [False, True, False, False, False],
        )
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        mark_combat_hit(get_world_service(), weapon_byte=0, victim_id=-1)
        result = _get_combat_feedback(bot)
        assert result == "miss"

    def test_feedback_single_shot_empty_tile_is_miss(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """weapon_byte=0 with empty target tile is 'miss' via tile-occupancy.

        Old heuristic couldn't tell single from miss (weapon_byte=0 was
        ambiguous). Tile-occupancy resolves it: empty tile = miss,
        regardless of weapon type or inventory state.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
        )

        reset_world_state()
        update_inventory_from_protocol(
            get_world_service(),
            [0, 0, 0, 0, 0],
            [False, False, False, False, False],
        )
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        mark_combat_hit(get_world_service(), weapon_byte=0, victim_id=-1)
        result = _get_combat_feedback(bot)
        assert result == "miss"

    def test_feedback_hit_decrements_dual_then_miss_on_empty(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """First hit decrements dual; next shot on empty tile is miss."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
        )

        reset_world_state()
        update_inventory_from_protocol(
            get_world_service(),
            [0, 1, 0, 0, 0],
            [False, True, False, False, False],
        )
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"

        # First shot: hit with dual, depletes to 0
        mark_combat_hit(get_world_service(), weapon_byte=1, victim_id=999)
        result = _get_combat_feedback(bot)
        assert result == "hit"
        assert get_inventory_state(get_world_service())["dual_shots"]["count"] == 0

        # Second shot: empty tile -> miss via tile-occupancy
        bot._ai_state["last_shot_target_id"] = 50
        mark_combat_hit(get_world_service(), weapon_byte=0, victim_id=-1)
        result = _get_combat_feedback(bot)
        assert result == "miss"
