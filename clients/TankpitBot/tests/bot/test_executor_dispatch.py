"""Tests for executor dispatch_command (split from test_executor, 2026-08-01)."""

from __future__ import annotations

from tankpit_bot.bot.ai.equipment import hostile_mines
from tankpit_bot.bot.executor import (
    dispatch_command,
)
from tankpit_bot.bot.types import (
    make_hold_command,
    make_map_open_command,
    make_mine_drop_command,
    make_move_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_radar_command,
    make_shoot_command,
    make_teleport_command,
)
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.physics.capacity import fuel_capacity, inventory_capacity
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total
from tankpit_bot.state.types import WorldStateDict, make_container_state, make_mine_state
from tankpit_bot.state.types.coord import coord_key
from tests.bot._executor_support import (
    _make_bot,
    _make_snapshot,
    _make_world,
    _WorldOnlyBot,
)
from tests.conftest import FakeEnv


def _believe_container(ws: WorldService, x: int, y: int, *, is_fuel: bool, volume: int) -> None:
    """Install one believed container into the given world service."""
    ws.world_state = WorldStateDict(
        **{
            **ws.world_state,
            "containers": {
                coord_key(x, y): make_container_state(
                    x=x, y=y, is_fuel=is_fuel, volume=volume, timestamp_ms=1000
                )
            },
        }
    )


def _believe_hostile_mine(ws: WorldService, x: int, y: int) -> None:
    """Install one enemy-team mine into the given world service.

    Self is team 0 in these fixtures (``update_self_position`` seeds a
    fresh self at team 0), so team 1 is hostile under
    :func:`~tankpit_bot.bot.ai.equipment.hostile_mines`.
    """
    ws.world_state = WorldStateDict(
        **{
            **ws.world_state,
            "mines": {
                coord_key(x, y): make_mine_state(
                    x=x, y=y, mine_type=1, tank_id=99, team=1, timestamp_ms=1000
                )
            },
        }
    )


def _fill_inventory_to(ws: WorldService, count: int) -> None:
    """Set every believed inventory slot to one count."""
    item = InventoryItem(count=count, enabled=True)
    ws.inventory_state = InventoryState(
        armor_shields=item,
        dual_shots=item,
        missile_shots=item,
        homing_shots=item,
        extra_radars=item,
    )


class TestDispatchCommand:
    """Tests for dispatch_command."""

    def test_dispatch_move(self, fake_env: FakeEnv) -> None:
        """Dispatches move command via bot.move_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_move_command(150, 160), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_pickup_fuel(self, fake_env: FakeEnv) -> None:
        """Dispatches pickup_fuel command via bot.pickup_fuel_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_pickup_fuel_command(80, 90), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_pickup_equipment(self, fake_env: FakeEnv) -> None:
        """Dispatches pickup_equipment command via bot.pickup_equipment_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_pickup_equipment_command(80, 90), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_shoot(self, fake_env: FakeEnv) -> None:
        """Dispatches shoot command via bot.shoot_at."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_shoot_command(105, 103), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_radar(self, fake_env: FakeEnv) -> None:
        """Dispatches radar command via bot.use_radar."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_radar_command(), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_mine_drop_books_the_flat_press(self, fake_env: FakeEnv) -> None:
        """The mine press reaches the wire and bills exactly -10 fuel.

        No decision-ledger entry exists for the press (the 0x4B answer
        is self-serving), but the fuel forensics layer's ``mine_press``
        entry must land ([[mine-mechanics]], [[game-economy]]).
        """
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_mine_drop_command(), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods
        entries = [e for e in bot.world.fuel_book["entries"] if e["kind"] == "mine_press"]
        assert [(e["lo"], e["hi"]) for e in entries] == [(-10, -10)]
        assert bot.world.last_wire_command_name == "mine_drop"

    def test_dispatch_mine_drop_without_cdp_books_nothing(self, fake_env: FakeEnv) -> None:
        """A failed send bills no fuel and reports False."""
        bot, _fake_cdp = _make_bot(fake_env)
        bot._cdp = None
        result = dispatch_command(bot, make_mine_drop_command(), _make_snapshot())
        assert result is False
        assert [e for e in bot.world.fuel_book["entries"] if e["kind"] == "mine_press"] == []

    def test_teleport_send_failure_books_nothing(self, fake_env: FakeEnv) -> None:
        """A wire-refused teleport bills no fuel and records no dispatch.

        The map is certified (overlay + receipt) but the CDP session is
        gone, so ``teleport_to`` reports False — the arc the map-receipt
        gate re-routed the old cover away from.
        """
        bot, _fake_cdp = _make_bot(fake_env)
        bot.world.last_wire_command_name = "map_open"
        bot._cdp = None
        result = dispatch_command(
            bot, make_teleport_command(120, 120), _make_snapshot(map_visible=True)
        )
        assert result is False
        assert [e for e in bot.world.fuel_book["entries"] if e["kind"] == "teleport"] == []

    def test_dispatch_map_open(self, fake_env: FakeEnv) -> None:
        """Dispatches map_open command via bot.open_map."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_map_open_command(), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_map_open_clears_stale_data_mark(self, fake_env: FakeEnv) -> None:
        """An orphan map-data flag cannot complete THIS open instantly.

        Run bot-20260825-212920's ending: a stalled open's late
        response left the ``map_data_processed`` flag set, the next
        open "completed" in 12 ms with no fresh dots, and the phantom
        fresh-empty snapshot exited the session under 27 live
        enemies. The dispatch now clears any stale mark (the scope
        pan's 2026-08-20 discipline) so the flag means "a MAP_DATA
        arrived since THIS open".
        """
        bot, _fake_cdp = _make_bot(fake_env)
        bot.world.mark_map_data_processed()
        result = dispatch_command(bot, make_map_open_command(), _make_snapshot())
        assert result is True
        assert bot.world.check_and_clear_map_data_processed() is False

    def test_dispatch_chat(self, fake_env: FakeEnv) -> None:
        """Dispatches chat command via bot.send_chat."""
        from tankpit_bot.bot.types import make_chat_command

        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_chat_command(41, 100, 100), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_chat_without_cdp_fails(self, fake_env: FakeEnv) -> None:
        """Chat dispatch reports False when no CDP session is attached."""
        from tankpit_bot.bot.types import make_chat_command

        bot, _fake_cdp = _make_bot(fake_env)
        bot._cdp = None
        result = dispatch_command(bot, make_chat_command(41, 100, 100), _make_snapshot())
        assert result is False

    def test_dispatch_scope_shift(self, fake_env: FakeEnv) -> None:
        """Dispatches scope_shift via bot.scope_shift onto the wire."""
        from tankpit_bot.bot.types import make_scope_shift_command
        from tankpit_bot.protocol.commands import SCOPE_EAST

        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_scope_shift_command(SCOPE_EAST), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_scope_shift_without_cdp_fails(self, fake_env: FakeEnv) -> None:
        """Scope dispatch reports False when no CDP session is attached."""
        from tankpit_bot.bot.types import make_scope_shift_command
        from tankpit_bot.protocol.commands import SCOPE_WEST

        bot, _fake_cdp = _make_bot(fake_env)
        bot._cdp = None
        result = dispatch_command(bot, make_scope_shift_command(SCOPE_WEST), _make_snapshot())
        assert result is False

    def test_dispatch_hold_sends_nothing(self, fake_env: FakeEnv) -> None:
        """Hold command returns True and does not touch the wire.

        The SPA-driven idle tick must not dispatch any CDP command;
        the fake CDP session confirms no ``Runtime.evaluate`` reached
        it while ``dispatch_command`` still reports success (the
        desired effect — do nothing — was achieved).
        """
        bot, fake_cdp = _make_bot(fake_env)
        assert "Runtime.evaluate" not in fake_cdp._sent_methods
        result = dispatch_command(bot, make_hold_command(), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" not in fake_cdp._sent_methods

    def test_dispatch_map_open_sends_wire_only_when_already_visible(
        self, fake_env: FakeEnv
    ) -> None:
        """A visible map dispatches CMD_MAP_OPEN with no client-side keypress.

        Regression guard for capture 20260620-183916: CMD_MAP_OPEN is
        idempotent on the server -- every wire dispatch produces a fresh
        MAP_DATA payload regardless of overlay visibility. The previous
        close-then-reopen hack added a synthetic 'm' keypress before
        every redundant intel refresh; the wire dispatch alone is what
        the server actually needs.
        """
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(
            bot,
            make_map_open_command(),
            _make_snapshot(map_visible=True),
        )
        assert result is True
        # No synthetic 'm' keypress dispatched -- only the wire command.
        key_events = [m for m in fake_cdp._sent_methods if m == "Input.dispatchKeyEvent"]
        assert key_events == []
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_teleport(self, fake_env: FakeEnv) -> None:
        """Dispatches teleport command via bot.teleport_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_teleport_command(200, 200), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_teleport_records_no_attempt_when_send_fails(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A failed teleport send leaves no pending attempt to mislabel later."""
        from tankpit_bot.ledger.outcome.teleport import emit_teleport_landed

        ws = WorldService()
        world = _make_world()
        result = dispatch_command(
            _WorldOnlyBot(world),
            make_teleport_command(200, 200),
            _make_snapshot(map_visible=True),
        )

        assert result is False
        landed = emit_teleport_landed(
            ws.ledger,
            duration_ms=0,
            target_x=200,
            target_y=200,
            landed_x=200,
            landed_y=200,
            messages=[],
        )
        assert landed["detail"]["sent_window"] == "(none)"

    def test_dispatch_teleport_skips_open_map_when_map_already_visible(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Teleport skips the precondition map_open when the map is already open.

        The hop must be affordable — (120,120) from (100,100) costs 169
        against 800 fuel; an unaffordable target is now suppressed by
        the ``physics/supervisor.py`` refusal prediction before the send.
        """
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(
            bot,
            make_teleport_command(120, 120),
            _make_snapshot(map_visible=True),
        )
        assert result is True
        sent_methods = fake_cdp._sent_methods
        runtime_calls = [m for m in sent_methods if m == "Runtime.evaluate"]
        assert len(runtime_calls) == 1

    def test_teleport_to_mined_tile_is_dispatchable(self, fake_env: FakeEnv) -> None:
        """A teleport aimed at a hostile-mined tile still dispatches.

        Landing legality is a different question from walkability: the
        server displaces the landing off a mined tile and charges the
        plain cost, so a mine at the target is no reason to withhold
        the send ([[terrain-composition]], [[teleport-mechanics]]
        Placement). The executor's mine veto was deleted 2026-07-20
        (commit 6d2afdbe) because it was wrong physics and looped the
        pursuit; this test is the standing proof it stays deleted.

        Target (120,120) from (100,100) costs 169 against 800 fuel, so
        the refusal predictor does not suppress the send for cost.
        """
        bot, fake_cdp = _make_bot(fake_env)
        _believe_hostile_mine(bot.world, 120, 120)
        assert coord_key(120, 120) in hostile_mines(bot.world.world_state)

        result = dispatch_command(
            bot,
            make_teleport_command(120, 120),
            _make_snapshot(map_visible=True),
        )

        assert result is True
        runtime_calls = [m for m in fake_cdp._sent_methods if m == "Runtime.evaluate"]
        assert len(runtime_calls) == 1


class TestDispatchRefusalPrediction:
    """Belief-refuted commands are suppressed at the chokepoint.

    The ``physics/supervisor.py`` laws applied to live belief — the
    20-kill soak bot-20260802-205105 sent 48 provably-refusable fuel
    pickups (code 5) that this seam now suppresses.
    """

    def test_fuel_pickup_at_capacity_is_suppressed(self, fake_env: FakeEnv) -> None:
        """Full tank + stocked believed container: no wire traffic."""
        bot, fake_cdp = _make_bot(fake_env)
        self_state = bot.world.world_state["self_state"]
        if self_state is None:
            raise AssertionError("fixture bot must have a self state")
        update_world_state_from_fuel_total(bot.world, fuel_capacity(self_state["rank"]))
        _believe_container(bot.world, 80, 90, is_fuel=True, volume=508)
        result = dispatch_command(bot, make_pickup_fuel_command(80, 90), _make_snapshot())
        assert result is False
        assert "Runtime.evaluate" not in fake_cdp._sent_methods

    def test_fuel_pickup_on_equipment_belief_dispatches(self, fake_env: FakeEnv) -> None:
        """A non-fuel record at the target proves nothing about fuel."""
        bot, fake_cdp = _make_bot(fake_env)
        self_state = bot.world.world_state["self_state"]
        if self_state is None:
            raise AssertionError("fixture bot must have a self state")
        update_world_state_from_fuel_total(bot.world, fuel_capacity(self_state["rank"]))
        _believe_container(bot.world, 80, 90, is_fuel=False, volume=0)
        result = dispatch_command(bot, make_pickup_fuel_command(80, 90), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_equipment_pickup_all_slots_full_is_suppressed(self, fake_env: FakeEnv) -> None:
        """Every slot at the rank cap: code 7 predicted, nothing sent."""
        bot, fake_cdp = _make_bot(fake_env)
        self_state = bot.world.world_state["self_state"]
        if self_state is None:
            raise AssertionError("fixture bot must have a self state")
        _fill_inventory_to(bot.world, inventory_capacity(self_state["rank"]))
        result = dispatch_command(bot, make_pickup_equipment_command(80, 90), _make_snapshot())
        assert result is False
        assert "Runtime.evaluate" not in fake_cdp._sent_methods

    def test_unaffordable_teleport_is_suppressed(self, fake_env: FakeEnv) -> None:
        """(200,200) from (100,100) costs 848 against 800 fuel: even the
        cheapest ring-1 landing is out of reach, so nothing is sent."""
        bot, fake_cdp = _make_bot(fake_env)
        bot.world.last_wire_command_name = "map_open"
        result = dispatch_command(
            bot, make_teleport_command(200, 200), _make_snapshot(map_visible=True)
        )
        assert result is False
        assert "Runtime.evaluate" not in fake_cdp._sent_methods

    def test_missing_self_state_stays_optimistic(self, fake_env: FakeEnv) -> None:
        """No self belief proves nothing: pickup and teleport dispatch.

        The bot gets a FRESH world rather than a global reset.
        ``reset_world_state()`` rebinds the module global, and the bot
        now holds its service as an attribute — so the reset left
        ``bot.world`` pointing at the pre-reset object with
        ``_make_bot``'s seeded position still in it, and the test
        silently stopped exercising the missing-self-state path
        ([[session-state-deglobalisation]] step 8).
        """
        bot, fake_cdp = _make_bot(fake_env)
        bot.world = WorldService()
        picked = dispatch_command(bot, make_pickup_fuel_command(80, 90), _make_snapshot())
        bot.world.last_wire_command_name = "map_open"
        hopped = dispatch_command(
            bot, make_teleport_command(200, 200), _make_snapshot(map_visible=True)
        )
        assert picked is True
        assert hopped is True
        assert fake_cdp._sent_methods.count("Runtime.evaluate") == 2

    def test_overlay_without_the_open_receipt_defers_the_teleport(self, fake_env: FakeEnv) -> None:
        """A rendered overlay after another action re-opens, never teleports.

        The map-modal invalidation law (run bot-20260901-032936
        03:34:21-25): a scope pan dispatched between the deferral's
        open and the teleport closed the map server-side while the
        client overlay still rendered, and the fast-path teleport drew
        cant_do code 0. With the last-wire-command receipt naming the
        pan instead of the open, the executor must spend the tick on a
        fresh map open.
        """
        bot, _fake_cdp = _make_bot(fake_env)
        bot.world.last_wire_command_name = "scope(5)"

        result = dispatch_command(
            bot, make_teleport_command(120, 120), _make_snapshot(map_visible=True)
        )

        assert result is True
        # The dispatch that went out is the map open: the receipt now
        # names it, so the NEXT tick's fast path is legitimate.
        assert bot.world.last_wire_command_name == "map_open"
