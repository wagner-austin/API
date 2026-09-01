"""Typed world state decoded from the agent's NDJSON stream.

The stream is a sequence of records discriminated by ``kind``. A ``frame``
record opens a sample and declares how many ``entity`` and ``pool`` records
follow; the entity records carry the visible roster and the pool records the
visible resource pools. Folding them back into whole samples is this module's
job.

The declared counts are checked rather than trusted. A sample that promises
three entities and delivers two is a truncated capture — the ordinary result of
reading a stream while the agent is still writing it — and silently yielding the
short sample would let a planner make decisions on a roster it cannot see all
of.

Nothing here reads a file. Decoding is a pure function of the lines it is
given, which is what lets the same code serve a live tail and an archived
replay corpus without a branch between them.
"""

from __future__ import annotations

from typing import Final, TypedDict

KIND_FRAME: Final = "frame"
"""``kind`` value opening a sample."""

KIND_ENTITY: Final = "entity"
"""``kind`` value of an entity record inside a sample."""

KIND_POOL: Final = "pool"
"""``kind`` value of a resource-pool record inside a sample."""

KIND_OPTION: Final = "option"
"""``kind`` value of a build-option record inside a sample."""

KIND_PLAYER: Final = "player"
"""``kind`` value of a per-player scoreboard record inside a sample."""

KIND_REFUSED: Final = "refused"
"""``kind`` value of a refused-build record inside a sample.

The engine refuses a blocked placement silently -- the waypoint is dropped
after one in-range attempt and nothing at the shipped log level says so. The
agent watches every dispatched build order and reports the engine's verdict
here the sample after it becomes visible, so a refusal costs the planner one
sample instead of a 45-sample quiet clock (wiki log 2026-08-31,
verdict-withheld; 2026-09-01, the detection design).
"""

CHILD_COUNT_FIELDS: Final = ("visible", "pools", "options", "players", "refused")
"""Frame-record fields declaring how many child records follow.

Declared once because two readers need the total and they must not disagree:
this module folds records into samples, and
:mod:`rw_bot.control.channel` decides when a sample has finished arriving. When
the channel carried its own copy of this list, adding a record kind left it
reading a sample one short — the decoder then rejected it as truncated, which
is a true statement about the wrong thing.
"""


class Entity(TypedDict):
    """One entity the local player owns at a given frame.

    Attributes:
        index: Position in the owned roster, as the agent enumerated it. Useful
            for reading a single sample and nothing else: it renumbers whenever
            anything is built or dies, so it is not an addressing handle.
        unit_id: The engine's own object identity, assigned once at construction
            and used by the engine for network identity. This is the handle an
            order is dispatched against.
        type_name: Readable unit-type name, e.g. ``"builder"``. The same string
            the type registry accepts when building.
        class_name: Engine class of the entity, obfuscated and pinned to the
            recorded build.
        x: World x coordinate.
        y: World y coordinate.
        team: Owning team number. Present for every visible entity, friend or
            not.
        mine: Whether the local player owns it. The stream carries enemies too,
            so a consumer that skips this check would credit an opponent's
            buildings to itself.
        hostile: Whether the engine considers this entity's owner an enemy of
            the local player. Not the negation of ``mine``: an ally's units are
            neither, and so are the neutral team's. Read from the engine's own
            alliance comparison rather than derived here, because a planner that
            treats every unowned unit as a threat cannot cross its own ally's
            territory.
        movement: Engine name of the layer this entity travels on, e.g.
            ``"LAND"``, ``"AIR"``, ``"HOVER"``. The engine keeps a separate
            connectivity grid per layer, so this says which grid ``group``
            belongs to.
        group: Connectivity component this entity stands in, on its own layer.
            Two things can reach each other exactly when their components match
            and both are non-negative — a negative is the engine's way of
            saying the point has no component at all
            ([[mechanics-movement-layers]]).
        flying: Whether it is airborne at this moment. State rather than type:
            a gunship that has landed reports false and can then be shot by
            units that cannot hit aircraft, which is why this is read per
            sample rather than derived once from ``movement``.
        submerged: Whether it is below the surface at this moment. A surfaced
            submarine is an ordinary target and reports false.
        touching_water: Whether it is standing in water at this moment. Only
            matters to weapons that cannot strike a target clear of it — a
            torpedo — and is carried so that rule needs no special case
            ([[mechanics-combat-profile]]).
        hp: Current health.
        max_hp: Health at full.
        complete: Whether construction has finished. A building joins the roster
            the moment construction starts, so presence is not completion — and
            an unfinished factory never advances its production queue.
        damaged_by: Type name of the unit that last damaged this one, empty
            when nothing has. The engine maintains the reference natively
            (``y.bt``); the sampler publishes its TYPE so the trace's loss
            diff can name the killer off the previous sample exactly as it
            already names the position ([[policy-trace]]). A damager that
            has since died still answers -- attribution wants the type, not
            a live reference.
        queued: Units this entity has queued for production, zero for anything
            that makes nothing. A production order changes no roster until the
            unit is finished, so this is the only immediate evidence that the
            engine accepted one.
    """

    index: int
    unit_id: int
    type_name: str
    class_name: str
    x: float
    y: float
    team: int
    mine: bool
    hostile: bool
    movement: str
    group: int
    flying: bool
    submerged: bool
    touching_water: bool
    hp: float
    max_hp: float
    complete: bool
    queued: int
    damaged_by: str


class ResourcePool(TypedDict):
    """One resource pool the local player can currently see.

    Pools are terrain rather than units: they appear in no entity list, and a
    planner reading only the roster cannot see them at all. They matter because
    an extractor is the one structure the engine refuses to place anywhere else.

    Both coordinate systems are carried because they answer different
    questions. The tile coordinate identifies the pool — integral, fixed for the
    life of the map, and the unit the engine's own placement check works in. The
    world point is where a build order has to be addressed.

    Attributes:
        index: Position in the sample's pool list. Enumeration order only.
        tile_x: Tile column.
        tile_y: Tile row.
        x: World x of the tile's centre.
        y: World y of the tile's centre.
        group_land: Connectivity component of this tile on the **land** layer,
            or a negative when it has none. Compare against a land unit's
            ``group`` to decide whether it can walk here at all. Land
            specifically: every builder in the base game travels on land, and
            naming the layer keeps a mismatched comparison from looking like an
            answer ([[mechanics-movement-layers]]).
    """

    index: int
    tile_x: int
    tile_y: int
    x: float
    y: float
    group_land: int


class BuildOption(TypedDict):
    """One thing an owned unit can make.

    The engine treats placing a building and producing a unit as the same
    mechanism: a builder's actions yield buildings, a factory's yield units, and
    one command verb dispatches either. What differs is only which unit has the
    action, so this is the table that answers "who can make X" — a question no
    stat dump answers and that the bot has twice guessed wrong.

    Attributes:
        index: Position in the sample's option list. Enumeration order only.
        unit_id: Engine identity of the unit that can make it. This is what an
            order is addressed to, so no second lookup is needed.
        produces: Type name it makes, in the same vocabulary a plan uses.
        key: The engine's interned key name for the action (``u_builder``,
            ``c_1``, ...). The dispatch handle an ability order carries. The
            engine also exposes a per-action index, and it is not a
            selector -- every action on a unit answers the same figure, so
            dispatching by it resolved the rally point four probes running.
            The key is what the engine's own executor resolves actions by
            ([[mechanics-build-actions]]). Empty when the action has no key,
            which also marks it undispatchable.
        placed: Whether the thing is put at a position the planner chooses. A
            structure is; a unit rolls out of the building that made it. This
            decides which verb orders it, and it is the engine's own
            distinction rather than a guess from the type's speed.
        available: Whether the unit may use it right now. An action that exists
            but is unavailable is a wait; one that does not exist at all is a
            dead plan entry, and the two need different answers.
        makes_something: Whether the engine calls this an action that produces
            a thing, as opposed to a stop or a rally.

            **Judged here rather than in the agent, and that distinction cost a
            whole category of the game.** The agent used to drop any action
            that neither placed something nor answered true here. An upgrade is
            neither: the asset declares it as ``convertTo``, and the engine
            models conversions separately from builds. So an owned extractor
            published no options at all while opponents were observed holding
            twelve upgraded ones ([[policy-holding-ground]]). Whether an action
            is worth taking is a decision, and decisions belong to this layer.
        price: What the action costs in credits, from the engine's own
            accessor. The only reading that tells a factory's tier upgrade
            from its rally point -- both concern no type -- and the first
            live tech probe spent four unlock budgets setting rally points
            for want of it ([[mechanics-build-actions]]).
    """

    index: int
    unit_id: int
    produces: str
    key: str
    placed: bool
    available: bool
    makes_something: bool
    price: int


class PlayerStat(TypedDict):
    """One player's scoreboard, as the engine keeps it.

    The bot measured its own income by regressing a credit balance against the
    clock across windows in which it deliberately bought nothing
    ([[policy-economy]]). The engine keeps the figure, keeps it for every player
    rather than only ours, charts all three of these and writes them to its own
    save file. So this is not a better estimate; it is the number the estimate
    was approximating.

    Carried per player because the useful form is comparative. "Our army is
    worth 500" says nothing on its own; "500 against four opponents on 1,000"
    is the whole of the match report — and unlike the visible-enemy count it
    cannot be inflated by our own scouting.

    Attributes:
        index: Position in the sample's player list. Enumeration order only.
        team: The engine's team number, which is what joins a player to the
            ``team`` field of a visible entity.
        local: Whether this is the player the bot is playing.
        hostile: Whether the engine considers this player an enemy of the local
            one. Read from the engine's own alliance comparison, so an ally and
            the neutral team are both false rather than "not local".
        defeated: Whether the engine has set this player's defeat flag.
        wiped: Whether the engine has set the stronger wiped-out flag.
        income: Credits earned per second.
        army_value: Total value of everything mobile the player holds.
        building_value: Total value of everything standing.
    """

    index: int
    team: int
    local: bool
    hostile: bool
    defeated: bool
    wiped: bool
    income: int
    army_value: int
    building_value: int


class Refusal(TypedDict):
    """One build order the engine refused silently, caught by the agent.

    Attributes:
        unit_id: The builder the refused order was addressed to.
        type_name: What the order tried to place.
        x: Placement world x the order asked for.
        y: Placement world y the order asked for.
    """

    unit_id: int
    type_name: str
    x: float
    y: float


class Sample(TypedDict):
    """One coherent observation of the world.

    Attributes:
        frame: The engine's frame counter at the moment of the read.
        clock_ms: The engine's millisecond clock at the same moment.
        credits: The current player's credits, floored to whole currency. The
            engine spends in whole units, so a planner comparing against a unit
            price wants the floor: 99 credits does not buy a 100-credit
            structure.
        entities: Every visible entity, in the order the agent enumerated it.
            Includes entities the local player does not own; check
            :attr:`Entity.mine`.
        pools: Every resource pool currently visible, in scan order. Fog-filtered
            by the engine's own per-tile test, so this grows as the map is
            explored rather than listing the whole map from the first frame.
        options: Everything the player's own units can currently make, one entry
            per producible type per unit.
        players: One scoreboard per occupied player slot, in slot order. Not
            fog-filtered: these are the engine's own bookkeeping rather than
            anything observed, so an opponent's army value is known even when
            none of it is in sight.
        refusals: Build orders the engine refused silently since the previous
            sample, each reported once. The planner feeds every site straight
            into the workforce's refusal ledger.
    """

    frame: int
    clock_ms: int
    credits: int
    defeated: bool
    wiped: bool
    players_left: int
    entities: tuple[Entity, ...]
    pools: tuple[ResourcePool, ...]
    options: tuple[BuildOption, ...]
    players: tuple[PlayerStat, ...]
    refusals: tuple[Refusal, ...]
