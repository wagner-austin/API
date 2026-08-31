"""Pursuit fire at departed targets: homing trace, human cap, never-engaged resume."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tankpit_bot.types.modes import AIModeState
from tests.bot.ai._support import (
    make_inventory,
    make_pursuit_target,
    make_scanned_ai_state,
    make_world,
)


def test_hunt_engage_fires_homing_when_locked_target_left_viewport() -> None:
    """Locked target out of viewport but wire-fresh -> fire at last known pos.

    Behavior-contract guard (2026-06-22 user directive): when the
    locked target teleports out of view we DO NOT enter CONFIRM_KILL
    and DO NOT chase. The bot stays put and fires at the world-known
    position; the server picks a homing shot when the target is
    distant or in motion, and homing tracks. The lock holds until a
    deactivation signal arrives.
    """
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
            "last_shot_target_id": 50,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["behavior"]["reason_kind"] == "shoot_target"


def test_hunt_close_fires_homing_when_locked_target_left_viewport() -> None:
    """CLOSE state pursues via homing fire when target leaves viewport after engagement."""
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "CLOSE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
            "last_shot_target_id": 50,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"


def test_hunt_refresh_fires_homing_when_locked_target_left_viewport() -> None:
    """REFRESH state pursues via homing fire when target leaves viewport after engagement."""
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "REFRESH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
            "last_shot_target_id": 50,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"


class TestPursuitHomingCapVsHumans:
    """One pursuit homing per departure window against humans.

    User ruling 2026-07-31: "restrict it to one homing shot when
    against humans, cuz its cheating to use all 7 that we can do
    during ttl." The window is delimited by the target's
    ``last_viewport_observation_ms`` (92000 in the fixture): a stamp
    at or after it means this departure's shot is spent and pursuit
    ticks chase via the map instead. Practice bots stay uncapped.
    """

    def _ctx(
        self,
        *,
        name: str = "Yuppler",
        mode_state: AIModeState = "ENGAGE",
        pursuit_shot_target_id: int = -1,
        pursuit_shot_ms: int = 0,
    ) -> DecideCtx:
        ws = WorldService()
        tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=150, y=150, name=name)}
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": mode_state,
                "mode_started_ms": 90000,
                "combat_target_id": 50,
                "combat_target_x": 150,
                "combat_target_y": 150,
                "last_shot_target_id": 50,
                "pursuit_shot_target_id": pursuit_shot_target_id,
                "pursuit_shot_ms": pursuit_shot_ms,
            }
        )
        return DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "", ws=ws)

    def test_first_pursuit_shot_fires_and_stamps_the_window(self) -> None:
        decision = decide_hunt_mode(self._ctx())

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["updated_ai_state"]["pursuit_shot_target_id"] == 50
        assert decision["updated_ai_state"]["pursuit_shot_ms"] == 100000

    def test_spent_window_chases_via_map_instead_of_second_homing(self) -> None:
        decision = decide_hunt_mode(self._ctx(pursuit_shot_target_id=50, pursuit_shot_ms=95000))

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "find_target"
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_viewport_reentry_rearms_the_budget(self) -> None:
        # The stamp (91000) predates the target's last in-viewport
        # observation (92000): they were SEEN since the last pursuit
        # shot, so this departure gets its one homing.
        decision = decide_hunt_mode(self._ctx(pursuit_shot_target_id=50, pursuit_shot_ms=91000))

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["updated_ai_state"]["pursuit_shot_ms"] == 100000

    def test_a_different_target_gets_a_fresh_budget(self) -> None:
        decision = decide_hunt_mode(self._ctx(pursuit_shot_target_id=49, pursuit_shot_ms=95000))

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["updated_ai_state"]["pursuit_shot_target_id"] == 50

    def test_practice_bot_pursuit_is_uncapped(self) -> None:
        decision = decide_hunt_mode(
            self._ctx(name="red-9", pursuit_shot_target_id=50, pursuit_shot_ms=95000)
        )

        assert decision["command"]["cmd_type"] == "shoot"

    def test_scan_on_landing_pursuit_honors_the_cap(self) -> None:
        decision = decide_hunt_mode(
            self._ctx(
                mode_state="SCAN_ON_LANDING",
                pursuit_shot_target_id=50,
                pursuit_shot_ms=95000,
            )
        )

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "find_target"


class TestPursuitFramesNearTargets:
    """A shift-revealable cached position is framed, never sniped.

    The visibility law (flag s11-2, 2026-08-13): within one anchor-law
    shift the server refuses to homing-track ("close enough that a
    viewport shift would reveal them" -- confirmed live by artax's six
    ``weapon=0`` edge misses at Arterial one row off-window), so
    pursuit fire there is a structural miss that would also wrongly
    spend the human homing budget. The free framing shift comes first.
    """

    def _near_ctx(self) -> DecideCtx:
        """Build an engaged pursuit ctx with the target one row off-window."""
        ws = WorldService()
        tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=100, y=91, name="Yuppler")}
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "ENGAGE",
                "mode_started_ms": 90000,
                "combat_target_id": 50,
                "combat_target_x": 100,
                "combat_target_y": 91,
                "last_shot_target_id": 50,
            }
        )
        return DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "", ws=ws)

    def test_near_pursuit_frames_and_keeps_the_homing_budget(self) -> None:
        decision = decide_hunt_mode(self._near_ctx())

        assert decision["command"]["cmd_type"] == "scope_shift"
        assert decision["behavior"]["reason_kind"] == "combat_frame_shift"
        # The budget stamp belongs to a fired pursuit homing only --
        # a framing shift must leave this departure's shot unspent.
        assert decision["updated_ai_state"]["pursuit_shot_ms"] == 0


def test_hunt_close_re_teleports_when_lock_was_never_engaged() -> None:
    """CLOSE state re-teleports when lock was set but no shot ever fired at it.

    Regression guard for live run 2026-06-23 21:36:31: bot was at
    (46,100), planner emitted teleport to red-4 at (56,177); the
    executor swapped the teleport for a pre-teleport ``map_open``
    (precondition not met). The map_open took 6116ms and the next
    decision ran in HUNT/CLOSE -- which previously fell straight to
    pursuit and fired ``shoot`` at (56,177) from dist=87, looping 19
    times. ``last_shot_target_id != combat_target_id`` is the
    discriminator: we set the lock but never fired, so the lock is
    the pre-engagement intent, not a mid-fight chase. Re-issue the
    teleport instead of firing into the void.
    """
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=1090, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "CLOSE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"


def test_hunt_engage_re_teleports_when_lock_was_never_engaged() -> None:
    """ENGAGE substate re-teleports when lock was set but no shot ever fired."""
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=1090, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"


def test_hunt_refresh_re_teleports_when_lock_was_never_engaged() -> None:
    """REFRESH substate re-teleports when lock was set but no shot ever fired."""
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=1090, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "REFRESH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"


def test_pursuit_fire_stops_when_the_homing_trace_expires() -> None:
    """A departed target past the ~12 s trace gets the map chase, not a shot.

    Flags 4 and 5 of run bot-20260730-01xx: seven pursuit homings hit,
    then one shot always resolved after the reroute wall as a booked
    miss and a wasted tick ("couldnt we avoid the missed shot
    entirely? and save a tick"). Past ``PURSUIT_TRACE_TTL_MS`` the
    pursuit goes straight to the map refresh the miss would have
    bought anyway, with the lock held.
    """
    ws = WorldService()
    stale_target = make_tank_state(
        tank_id=50,
        x=150,
        y=150,
        team=2,
        rank=1,
        name="red-9",
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=100000,
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
        last_viewport_observation_ms=80000,
    )
    world, self_state = make_world(fuel=800, tanks={"50": stale_target})
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
            "last_shot_target_id": 50,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


class TestMapReplacedInWindowTarget:
    """A map-fresh position inside the window is engaged, never re-chased.

    Regression for run bot-20260831-152132 15:22:02-08 (operator flag
    1, "opening the map then firing then opening the map then
    firing"): red-3 arrived via a map answer at an in-window tile the
    wire had not yet confirmed, so the pursuit branch read the zero
    viewport stamp as an expired trace and spent a map_open
    rediscovering the very dot it already held. The map re-places
    every alive tank at its CURRENT tile on each open — a placement
    newer than the last viewport sighting that lands inside the
    visible window is the in-view firing law's case, not a reroute
    gamble.
    """

    def _ctx(
        self,
        target: TankStateDict,
        *,
        pursuit_shot_target_id: int = -1,
        pursuit_shot_ms: int = 0,
    ) -> DecideCtx:
        ws = WorldService()
        world, self_state = make_world(fuel=800, tanks={"50": target})
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "ENGAGE",
                "mode_started_ms": 90000,
                "combat_target_id": 50,
                "combat_target_x": target["x"],
                "combat_target_y": target["y"],
                "last_shot_target_id": 50,
                "pursuit_shot_target_id": pursuit_shot_target_id,
                "pursuit_shot_ms": pursuit_shot_ms,
            }
        )
        return DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "", ws=ws)

    def test_never_viewport_confirmed_in_window_dot_is_shot_not_map_chased(self) -> None:
        target = make_pursuit_target(x=104, y=100)
        target["last_viewport_observation_ms"] = 0

        decision = decide_hunt_mode(self._ctx(target))

        assert decision["command"]["cmd_type"] == "shoot"
        # Not a pursuit-budget shot: the target is here, not departed.
        assert decision["updated_ai_state"]["pursuit_shot_ms"] == 0

    def test_stale_map_placement_falls_back_to_the_map_chase(self) -> None:
        # The placement aged past the kill-shot horizon (8 s > 7 s) and
        # the trace wall is long gone (20 s > 12 s): refresh via map.
        target = make_pursuit_target(x=104, y=100)
        target["last_viewport_observation_ms"] = 80000
        target["last_position_update_ms"] = 92000

        decision = decide_hunt_mode(self._ctx(target))

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "find_target"

    def test_human_cap_spares_the_in_window_live_position(self) -> None:
        # The one-homing-per-departure cap governs reroute pursuit at
        # DEPARTED humans; a human the map places inside the window is
        # the normal in-view fight and keeps taking fire.
        target = make_pursuit_target(x=104, y=100, name="Yuppler")

        decision = decide_hunt_mode(
            self._ctx(target, pursuit_shot_target_id=50, pursuit_shot_ms=95000)
        )

        assert decision["command"]["cmd_type"] == "shoot"
        # The departure budget stamp is untouched by an in-view shot.
        assert decision["updated_ai_state"]["pursuit_shot_ms"] == 95000


def test_scan_on_landing_fires_homing_when_locked_target_left_viewport() -> None:
    """SCAN_ON_LANDING state pursues via homing fire when target leaves viewport."""
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": make_pursuit_target(x=150, y=150)}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "SCAN_ON_LANDING",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "shoot"


def test_scan_on_landing_pursuit_past_the_trace_wall_chases_via_map() -> None:
    """SCAN_ON_LANDING's pursuit branch also respects the homing-trace wall.

    The ENGAGE-path wall has its own pin; this covers the second
    call-site (audit 2026-07-30): a locked target that left the
    viewport more than PURSUIT_TRACE_TTL_MS ago gets the map chase
    instead of a guaranteed-miss shot.
    """
    ws = WorldService()
    stale = make_pursuit_target(x=150, y=150)
    stale["last_viewport_observation_ms"] = 87000  # 13 s > the 12 s wall
    tanks: dict[str, TankStateDict] = {"50": stale}
    world, self_state = make_world(fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "SCAN_ON_LANDING",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["behavior"]["reason_kind"] == "find_target"
