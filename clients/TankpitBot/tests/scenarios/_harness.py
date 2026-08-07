"""Scenario harness for the TankPit bot.

A :class:`BotScenario` is one bot-decision test case: a programmatic
description of the world the bot believes it's in, ready to feed into
the production :func:`tankpit_bot.bot.ai_strategy.decide` function.

Construction principle: **drive typed protocol messages through the
real dispatcher**. Never construct ``WorldStateDict`` by hand, never
mock the mutators, never fork the decision pipeline. The only
test-specific code is input message construction (in this module)
and output assertions (in the individual ``test_*.py`` files).

Typical use::

    def test_bot_targets_adjacent_enemy() -> None:
        scenario = BotScenario()
        scenario.place_self(x=100, y=100, fuel=800)
        scenario.place_enemy(tank_id=5, x=99, y=100, name="orange-3")

        decision = scenario.decide()

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["behavior"]["target_id"] == 5

The harness intentionally keeps its surface small. If a test needs a
state the constructors don't expose, the answer is almost always
"ingest another typed message", not "reach into ``world_state``
directly". When a needed message kind has no helper, add it to
:mod:`tests.scenarios._wire_builders` rather than inlining the
construction in a test.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Sequence

from tankpit_bot import _test_hooks
from tankpit_bot.bot.ai.types import (
    AIConfigDict,
    AIStateDict,
    make_initial_ai_state,
)
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.inventory import InventoryState
from tankpit_bot.protocol import (
    BinaryMessage,
    FuelGainDict,
    InventoryDict,
    MovementResponseDict,
    TankInfoDict,
    ViewportUpdateDict,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state import (
    get_world_service,
)
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.state.types import SelfStateDict, WorldStateDict

#: Default tank id used for the bot's own tank in scenarios that don't
#: care about the value. Matches the production capture for Artax so
#: replays and synthetic scenarios share an id space.
DEFAULT_SELF_TANK_ID: int = 1301

#: Default team for the bot's own tank in scenarios that don't care.
DEFAULT_SELF_TEAM: int = 2  # blue

#: Default starting fuel.
DEFAULT_SELF_FUEL: int = 800

#: Default tick clock the harness starts from. Wall-clock-independent
#: so scenarios are deterministic across machines.
DEFAULT_START_TIMESTAMP_MS: int = 100_000

#: Per-tick advance used by ``BotScenario.advance_clock`` when no
#: explicit delta is given. Mirrors the bot's nominal 1 tick/s cadence.
DEFAULT_TICK_DELTA_MS: int = 1000

#: Inventory slot counts established by :meth:`place_self` so a
#: baseline scenario looks like a real session right after the
#: server's 0x49 Inventory sync (every slot near full). Tests that
#: want to study low-inventory behaviour ingest a fresh
#: :class:`InventoryDict` after ``place_self`` and the new counts
#: overwrite these.
DEFAULT_ARMOR_SHIELDS: int = 25
DEFAULT_DUAL_SHOTS: int = 25
DEFAULT_MISSILE_SHOTS: int = 25
DEFAULT_HOMING_SHOTS: int = 25
DEFAULT_EXTRA_RADARS: int = 25


class BotScenario:
    """One bot-decision scenario, built imperatively then decided.

    Constructed plain (not as a TypedDict or dataclass) because the
    scenario carries methods over the bot's stateful dispatcher
    surface. Field types are explicit and assigned only in
    :meth:`__init__` to keep the shape easy to reason about.

    Attributes:
        timestamp_ms: Current wall-clock-equivalent the scenario will
            pass into ``decide()``. Advance with
            :meth:`advance_clock` or implicitly via :meth:`decide_many`.
        combat_feedback: The previous tick's shot feedback. Default is
            empty (no in-flight shot).
        config: Optional AI config override. ``None`` resolves to
            :func:`make_default_ai_config` inside the AI state factory.
        ai_state: Durable AI state. Initialised via
            :func:`make_initial_ai_state` unless overridden.
    """

    def __init__(
        self,
        timestamp_ms: int = DEFAULT_START_TIMESTAMP_MS,
        combat_feedback: CombatFeedback = "",
        config: AIConfigDict | None = None,
    ) -> None:
        """Reset the global world service and install the scenario clock.

        The world service is a singleton in production; the scenario
        owns it for the duration of one test. Tests MUST construct
        scenarios in isolated fixtures (see the ``isolate_world``
        fixture in the scenarios test modules) so they don't leak
        state between cases.

        The scenario also installs itself as the :mod:`_test_hooks`
        clock so the bot's wire-stamping mutators see the scenario's
        ``timestamp_ms`` instead of the real wall clock. Call
        :meth:`close` (or use the scenario as a context manager) to
        restore the real clock; ``isolate_world`` fixtures should
        delegate to one of those.

        Args:
            timestamp_ms: Initial scenario clock value.
            combat_feedback: Initial combat feedback channel.
            config: Optional AI config override.
        """
        self.timestamp_ms: int = timestamp_ms
        self.combat_feedback: CombatFeedback = combat_feedback
        self.config: AIConfigDict | None = config
        self.ai_state: AIStateDict = make_initial_ai_state(config)
        self._restore_clock: Callable[[], int] | None = _test_hooks.get_current_time_ms
        _test_hooks.get_current_time_ms = self._clock

    def _clock(self) -> int:
        """Return the scenario's current timestamp in milliseconds.

        Installed as the :mod:`_test_hooks` clock so production code
        that calls ``browser.get_current_time_ms()`` (the dispatcher's
        wire-stamping path, mode-controller cooldown checks) sees the
        scenario's clock instead of the real wall clock.

        Returns:
            Scenario clock value in milliseconds.
        """
        return self.timestamp_ms

    def close(self) -> None:
        """Restore the real wall-clock hook saved at construction.

        Idempotent; safe to call from a test's teardown or fixture
        cleanup even if the scenario already had ``close`` called on
        it elsewhere.
        """
        if self._restore_clock is None:
            return
        _test_hooks.get_current_time_ms = self._restore_clock
        self._restore_clock = None

    def __enter__(self) -> BotScenario:
        """Return ``self`` for ``with BotScenario() as scenario:`` use.

        Returns:
            The same scenario instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Restore the real wall-clock hook on context-manager exit.

        Args:
            exc_type: Exception type if one was raised in the ``with``
                block; unused (cleanup is unconditional).
            exc_val: Exception instance if one was raised; unused.
            exc_tb: Traceback object if one was raised; unused.
        """
        del exc_type, exc_val, exc_tb
        self.close()

    # -----------------------------------------------------------------
    # Live state accessors -- read-only views into the world service.
    # -----------------------------------------------------------------

    @property
    def ws(self) -> WorldService:
        """Return the live world service the dispatcher writes into.

        Returns:
            The global :class:`WorldService` instance that the
            production dispatcher mutates in place.
        """
        return get_world_service()

    @property
    def world(self) -> WorldStateDict:
        """Return the current ``WorldStateDict`` snapshot.

        Returns:
            Current world-state TypedDict.
        """
        return self.ws.world_state

    @property
    def self_state(self) -> SelfStateDict | None:
        """Return the bot's ``SelfStateDict`` or ``None``.

        Returns:
            The bot's own self-state TypedDict if established,
            otherwise ``None``.
        """
        return self.world["self_state"]

    @property
    def inventory(self) -> InventoryState:
        """Return the live inventory.

        Returns:
            Current inventory TypedDict.
        """
        return self.ws.inventory_state

    # -----------------------------------------------------------------
    # Imperative construction -- ingest typed messages.
    # -----------------------------------------------------------------

    def ingest(self, message: BinaryMessage) -> None:
        """Drive a typed protocol message through the real dispatcher.

        Keeps ``world_state["timestamp_ms"]`` in sync with the
        scenario clock so mutators that key off it see consistent
        values for the duration of the test.

        Args:
            message: A :data:`BinaryMessage` -- one of the
                ``protocol`` package's TypedDicts. The dispatcher
                routes it through the production mutator chain, so
                world-state changes match what a live session would
                produce for the same bytes.
        """
        self.ws.world_state["timestamp_ms"] = self.timestamp_ms
        dispatch_world_state_update(self.ws, message)

    def ingest_many(self, messages: Sequence[BinaryMessage]) -> None:
        """Ingest a sequence of typed messages in order.

        Args:
            messages: Sequence of :data:`BinaryMessage` values to
                feed through the dispatcher.
        """
        for message in messages:
            self.ingest(message)

    def advance_clock(self, delta_ms: int = DEFAULT_TICK_DELTA_MS) -> None:
        """Advance the scenario clock by ``delta_ms``.

        Args:
            delta_ms: Number of milliseconds to add. Default mirrors
                the bot's nominal tick cadence.
        """
        self.timestamp_ms += delta_ms

    # -----------------------------------------------------------------
    # High-level placement helpers. Each is a thin wrapper over
    # one or two ``ingest`` calls -- the wire path stays primary.
    # -----------------------------------------------------------------

    def place_self(
        self,
        x: int,
        y: int,
        fuel: int = DEFAULT_SELF_FUEL,
        tank_id: int = DEFAULT_SELF_TANK_ID,
        team: int = DEFAULT_SELF_TEAM,
        rank: int = 1,
        lb_score: int = 0,
        armor_shields: int = DEFAULT_ARMOR_SHIELDS,
        dual_shots: int = DEFAULT_DUAL_SHOTS,
        missile_shots: int = DEFAULT_MISSILE_SHOTS,
        homing_shots: int = DEFAULT_HOMING_SHOTS,
        extra_radars: int = DEFAULT_EXTRA_RADARS,
    ) -> None:
        """Establish ``self_state`` and a baseline inventory.

        Drives the three messages a real session always sees at
        startup -- 0x3D MovementResponse to register own position,
        0x44 FuelGain to set fuel, 0x49 Inventory to establish slot
        counts. Without the inventory the bot's mode controller will
        prioritise ``COLLECT`` over combat because every
        slot reads zero, which is not how a real session starts.

        Tests that want to study low-inventory behaviour ingest a
        fresh :class:`InventoryDict` after ``place_self`` and the
        new counts overwrite the defaults.

        Args:
            x: Tile X coordinate.
            y: Tile Y coordinate.
            fuel: Starting fuel level.
            tank_id: Own tank id.
            team: Team id (0-3).
            rank: Military rank.
            lb_score: Leaderboard score.
            armor_shields: Armor-shield slot count.
            dual_shots: Dual-shot slot count.
            missile_shots: Missile-shot slot count.
            homing_shots: Homing-shot slot count.
            extra_radars: Extra-radar slot count.
        """
        self.ingest(
            MovementResponseDict(
                msg_type=0x3D,
                team=team,
                tank_id=tank_id,
                x=x,
                y=y,
                direction=0,
                damage_state=0,
                rank=rank,
                lb_score=lb_score,
                carrying=0,
            )
        )
        self.ingest(FuelGainDict(msg_type=0x44, fuel_total=fuel, is_free=False, flag=1))
        # The join 0x5A: every real session's first viewport patch
        # establishes the authoritative viewport record centered on
        # the tank (left/top = pos - 8, the live client's anchor).
        # Consumers like the aim clamp and the greeting encounter
        # gate refuse to act on an unestablished record, so a
        # scenario without this models a pre-join limbo no decision
        # tick ever runs in.
        self.ingest(
            ViewportUpdateDict(
                msg_type=0x5A,
                viewport_left=x - 8,
                viewport_top=y - 8,
                entities=[],
            )
        )
        self.ingest(
            InventoryDict(
                msg_type=0x49,
                show=True,
                alternate=False,
                counts=[
                    armor_shields,
                    dual_shots,
                    missile_shots,
                    homing_shots,
                    extra_radars,
                ],
                enabled=[False, True, False, True, True],
            )
        )

    def place_enemy(
        self,
        tank_id: int,
        x: int,
        y: int,
        team: int = 1,
        rank: int = 1,
        name: str = "Enemy",
    ) -> None:
        """Register an enemy tank and wire-confirm its position.

        Drives a 0x21 TankInfo to register the tank, then a 0x3D
        MovementResponse to attach a wire-confirmed position so
        ``analyze_threats`` will surface it as an acquisition
        candidate.

        Args:
            tank_id: Enemy tank id.
            x: Enemy tile X.
            y: Enemy tile Y.
            team: Enemy team id (must differ from self team for the
                threat pipeline to register them as an enemy).
            rank: Military rank.
            name: Player name (logged in OUR_SHOT and diagnostics).
        """
        self.ingest(
            TankInfoDict(
                msg_type=0x21,
                tank_id=tank_id,
                team=team,
                name=name,
                decoration_state=b"",
                persistent_tank_id=0,
            )
        )
        self.ingest(
            MovementResponseDict(
                msg_type=0x3D,
                team=team,
                tank_id=tank_id,
                x=x,
                y=y,
                direction=0,
                damage_state=0,
                rank=rank,
                lb_score=0,
                carrying=0,
            )
        )

    # -----------------------------------------------------------------
    # The actual decision step.
    # -----------------------------------------------------------------

    def decide(self) -> TickDecisionDict:
        """Run one production ``decide()`` cycle and return the decision.

        Returns:
            The :class:`TickDecisionDict` the live bot would produce
            on this tick given the current world state, AI state,
            inventory, and clock.

        Raises:
            RuntimeError: If ``self_state`` has not been established
                yet (call :meth:`place_self` or equivalent first).
        """
        self_state = self.self_state
        if self_state is None:
            raise RuntimeError(
                "BotScenario.decide() called before self_state was established. "
                "Use scenario.place_self(...) or ingest a MovementResponse "
                "for the bot's own tank first."
            )
        decision = decide(
            world=self.world,
            self_state=self_state,
            ai_state=self.ai_state,
            inventory=self.inventory,
            timestamp_ms=self.timestamp_ms,
            terrain=None,
            combat_feedback=self.combat_feedback,
        )
        # The production tick loop advances ``ai_state`` from the
        # decision; mirror that so consecutive ``decide()`` calls see
        # the durable mode/substate the live bot would.
        self.ai_state = decision["updated_ai_state"]
        return decision

    def decide_many(self, ticks: int) -> list[TickDecisionDict]:
        """Run ``ticks`` consecutive decisions, advancing the clock between.

        Each tick advances ``self.timestamp_ms`` by
        :data:`DEFAULT_TICK_DELTA_MS` after the decision.

        Args:
            ticks: Number of decisions to run.

        Returns:
            One :class:`TickDecisionDict` per tick, in order.
        """
        decisions: list[TickDecisionDict] = []
        for _ in range(ticks):
            decisions.append(self.decide())
            self.advance_clock()
        return decisions


__all__ = [
    "DEFAULT_SELF_FUEL",
    "DEFAULT_SELF_TANK_ID",
    "DEFAULT_SELF_TEAM",
    "DEFAULT_START_TIMESTAMP_MS",
    "DEFAULT_TICK_DELTA_MS",
    "BotScenario",
]
