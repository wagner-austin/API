"""Typed world fixtures, built once and shared by every test that needs one.

Adding one field to the wire used to be a thirteen-file change. Every test that
wanted a world hand-wrote its own NDJSON, so ``flying`` appearing on the entity
record broke each of them separately and in the same way — and a fixture that
drifts from the encoder is worse than no fixture, because it passes while the
real stream would not.

So there is one builder per record, defaults for everything a given test does not
care about, and **the lines come from the production encoder**. A fixture cannot
disagree with :func:`~rw_bot.wire.state.encode_sample` about what a sample looks
like, because it is what wrote it.

These are constructors, not mocks. What comes out is the same TypedDict the
decoder produces from live bytes, so a test that exercises a policy against one
is exercising it against the real shape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import TracebackType

from rw_bot.control import _test_hooks
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.wire.codec import encode_sample
from rw_bot.wire.state import BuildOption, Entity, PlayerStat, ResourcePool, Sample


def entity(
    unit_id: int,
    type_name: str,
    *,
    index: int = 0,
    class_name: str = "com.corrodinggames.rts.game.units.x",
    x: float = 0.0,
    y: float = 0.0,
    team: int = 0,
    mine: bool = True,
    hostile: bool = False,
    movement: str = "LAND",
    group: int = 1,
    flying: bool = False,
    submerged: bool = False,
    touching_water: bool = False,
    hp: float = 100.0,
    max_hp: float = 100.0,
    complete: bool = True,
    queued: int = 0,
) -> Entity:
    """Build one entity.

    Args:
        unit_id: Engine identity.
        type_name: Engine type name.
        index: Position in the roster.
        class_name: Engine class name.
        x: World x.
        y: World y.
        team: Owning team number.
        mine: Whether the local player owns it.
        hostile: Whether the engine calls it an enemy.
        movement: Movement-layer name.
        group: Connectivity component on that layer.
        flying: Whether it is airborne right now.
        submerged: Whether it is below the surface right now.
        touching_water: Whether it is standing in water right now.
        hp: Current health.
        max_hp: Health at full.
        complete: Whether construction has finished.
        queued: Units queued for production.

    Returns:
        The entity.
    """
    return Entity(
        index=index,
        unit_id=unit_id,
        type_name=type_name,
        class_name=class_name,
        x=x,
        y=y,
        team=team,
        mine=mine,
        hostile=hostile,
        movement=movement,
        group=group,
        flying=flying,
        submerged=submerged,
        touching_water=touching_water,
        hp=hp,
        max_hp=max_hp,
        complete=complete,
        queued=queued,
    )


def enemy(
    unit_id: int,
    type_name: str,
    *,
    x: float = 0.0,
    y: float = 0.0,
    team: int = 1,
    movement: str = "LAND",
    group: int = 1,
    flying: bool = False,
    submerged: bool = False,
    touching_water: bool = False,
    hp: float = 100.0,
    max_hp: float = 100.0,
    complete: bool = True,
) -> Entity:
    """Build one hostile entity.

    Hostility and ownership are the engine's separate answers rather than
    negations of each other, so a fixture that wants an enemy has to set both,
    and every test doing that by hand is a place to get it half right.

    Spelled out rather than forwarded through ``**kwargs``: a forwarding
    wrapper cannot be typed without widening every field to a union and casting
    it back, and this package does not permit the cast.

    Args:
        unit_id: Engine identity.
        type_name: Engine type name.
        x: World x.
        y: World y.
        team: Owning team number.
        movement: Movement-layer name.
        group: Connectivity component on that layer.
        flying: Whether it is airborne right now.
        submerged: Whether it is below the surface right now.
        touching_water: Whether it is standing in water right now.
        hp: Current health.
        max_hp: Health at full.
        complete: Whether construction has finished.

    Returns:
        The entity, unowned and hostile.
    """
    return entity(
        unit_id,
        type_name,
        x=x,
        y=y,
        team=team,
        mine=False,
        hostile=True,
        movement=movement,
        group=group,
        flying=flying,
        submerged=submerged,
        touching_water=touching_water,
        hp=hp,
        max_hp=max_hp,
        complete=complete,
    )


def profile(
    type_name: str,
    attack_range: float,
    *,
    index: int = 0,
    land: bool = True,
    air: bool = False,
    underwater: bool = False,
    out_of_water: bool = True,
) -> CombatProfile:
    """Build one combat profile.

    Defaults describe the common case the live dump shows: a ground weapon that
    cannot reach aircraft or submarines. Anything else is stated.

    Args:
        type_name: Engine type name.
        attack_range: Reach in world units, zero for the unarmed.
        index: Position in the dump.
        land: Whether its fire reaches a ground target.
        air: Whether its fire reaches an airborne target.
        underwater: Whether its fire reaches a submerged target.
        out_of_water: Whether its ground fire reaches a target clear of water.

    Returns:
        The profile.
    """
    return CombatProfile(
        index=index,
        type_name=type_name,
        attack_range=attack_range,
        hits_land=land,
        hits_air=air,
        hits_underwater=underwater,
        hits_land_out_of_water=out_of_water,
    )


def profiles_for(catalogue: Mapping[str, UnitStats]) -> dict[str, CombatProfile]:
    """Build a profile table agreeing with a test's own catalogue.

    Armament has one owner in production — the registry dump — and a test that
    states it twice can state it two ways: a unit armed in the catalogue and
    unarmed in the profiles is a world the game cannot produce, and a test built
    on one proves nothing. Deriving the table from the catalogue keeps the two
    consistent by construction.

    Every type is ground-only, which is what the base game's buildable land
    units are. A test about layers overrides the entries it cares about.

    Args:
        catalogue: Unit stats by type name.

    Returns:
        Combat profiles by type name, complete for every type the catalogue
        describes.
    """
    table: dict[str, CombatProfile] = {}
    for index, (type_name, stats) in enumerate(catalogue.items()):
        weapon = stats["weapon"]
        reach = 0.0 if weapon is None else weapon["attack_range"]
        # An unarmed type reaches no layer at all, which is what the agent
        # writes: the engine's base predicates answer true for air and land
        # regardless of armament, so it forces them false rather than putting
        # "a Builder can shoot aircraft" on the wire
        # ([[mechanics-combat-profile]]). A fixture that said otherwise would
        # describe a world the game cannot produce -- and did: a zero-reach
        # enemy standing on a route was read as covering it.
        table[type_name] = profile(type_name, reach, index=index, land=reach > 0.0)
    return table


def pool(*, index: int = 0, x: float = 0.0, y: float = 0.0, group_land: int = 1) -> ResourcePool:
    """Build one resource pool.

    Tile coordinates are derived from the world point at the engine's 20-unit
    tile pitch, so a fixture cannot describe a pool whose two coordinate systems
    disagree.

    Args:
        index: Position in the pool list.
        x: World x of the tile centre.
        y: World y of the tile centre.
        group_land: Connectivity component on the land layer.

    Returns:
        The pool.
    """
    return ResourcePool(
        index=index,
        tile_x=int(x) // 20,
        tile_y=int(y) // 20,
        x=x,
        y=y,
        group_land=group_land,
    )


def option(
    unit_id: int,
    produces: str,
    *,
    index: int = 0,
    key: str = "u_x",
    placed: bool = False,
    available: bool = True,
    makes_something: bool = True,
    price: int = 0,
) -> BuildOption:
    """Build one build option.

    Args:
        unit_id: Engine identity of the unit that can make it.
        produces: Type name it makes.
        index: Position in the option list.
        key: The engine's interned key name for the action.
        placed: Whether it is sited by the planner.
        available: Whether it may be used right now.
        price: The engine's charge for the action. What tells a tier
            unlock from a rally point among no-type actions.

    Returns:
        The option.
    """
    return BuildOption(
        index=index,
        unit_id=unit_id,
        produces=produces,
        key=key,
        placed=placed,
        available=available,
        makes_something=makes_something,
        price=price,
    )


def player(
    team: int,
    *,
    index: int = 0,
    local: bool = False,
    hostile: bool = True,
    defeated: bool = False,
    wiped: bool = False,
    income: int = 18,
    army_value: int = 0,
    building_value: int = 0,
) -> PlayerStat:
    """Build one player scoreboard entry.

    Args:
        team: Engine team number.
        index: Position in the player list.
        local: Whether this is the player the bot is playing.
        hostile: Whether the engine calls them an enemy.
        defeated: Whether the defeat flag is set.
        wiped: Whether the wiped-out flag is set.
        income: Credits per second.
        army_value: Value of everything mobile.
        building_value: Value of everything standing.

    Returns:
        The scoreboard entry.
    """
    return PlayerStat(
        index=index,
        team=team,
        local=local,
        hostile=hostile,
        defeated=defeated,
        wiped=wiped,
        income=income,
        army_value=army_value,
        building_value=building_value,
    )


def sample(
    *entities: Entity,
    frame: int = 1,
    clock_ms: int = 0,
    credits: int = 4000,
    defeated: bool = False,
    wiped: bool = False,
    players_left: int = 6,
    pools: Sequence[ResourcePool] = (),
    options: Sequence[BuildOption] = (),
    players: Sequence[PlayerStat] = (),
) -> Sample:
    """Build one whole observation.

    Indices are assigned from position, so a fixture cannot carry two entities
    claiming the same slot.

    Args:
        *entities: The visible roster, in order.
        frame: Engine frame counter.
        clock_ms: Engine millisecond clock.
        credits: Credits the local player holds.
        defeated: Whether the local player has been defeated.
        wiped: Whether the local player has been wiped out.
        players_left: Players still in the match.
        pools: Visible resource pools.
        options: What the player's units can make.
        players: Per-player scoreboards.

    Returns:
        The sample.
    """
    return Sample(
        frame=frame,
        clock_ms=clock_ms,
        credits=credits,
        defeated=defeated,
        wiped=wiped,
        players_left=players_left,
        entities=tuple(_reindex(entities)),
        pools=tuple(pools),
        options=tuple(options),
        players=tuple(players),
    )


def _reindex(entities: Sequence[Entity]) -> list[Entity]:
    """Renumber entities by position.

    Args:
        entities: The roster, in order.

    Returns:
        The same entities, each carrying its own position as ``index``.
    """
    return [{**e, "index": position} for position, e in enumerate(entities)]


def lines(*samples: Sample) -> list[str]:
    """Render whole samples as the NDJSON the agent would write.

    Delegates to the production encoder rather than formatting here, so a
    fixture cannot describe a record shape the real stream does not produce.

    Args:
        *samples: The samples to render, in order.

    Returns:
        Every line, in stream order, without newline terminators.
    """
    out: list[str] = []
    for one in samples:
        out.extend(encode_sample(one))
    return out


def repeated(one: Sample, times: int) -> list[str]:
    """Render one sample as many identical observations.

    A loop under test reads until its budget runs out, so most tests need the
    same world several times rather than several worlds.

    Args:
        one: The sample to repeat.
        times: How many observations to produce.

    Returns:
        The lines, in stream order.
    """
    return lines(*(one for _ in range(times)))


class ScriptedPeer:
    """Serves prepared lines and records what was sent back.

    Attributes:
        sent: Every line written, in order.
    """

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.sent: list[str] = []

    def send_line(self, line: str) -> None:
        """Record one written line.

        Args:
            line: Line content, without a newline.
        """
        self.sent.append(line)

    def read_line(self) -> str:
        """Serve the next prepared line, or end of stream.

        Returns:
            The next line, or an empty string once exhausted.
        """
        if not self._lines:
            return ""
        return self._lines.pop(0)

    def close(self) -> None:
        """Release the connection."""


class StubbedConnect:
    """Binds the connect hook to a scripted peer for one block.

    Attributes:
        peer: The peer every connection returns.
    """

    def __init__(self, peer: ScriptedPeer) -> None:
        self.peer = peer
        self._original: _test_hooks.ConnectProto = _test_hooks.connect

    def __call__(self, host: str, port: int, timeout_s: float) -> _test_hooks.Connection:
        """Return the scripted peer.

        Args:
            host: Ignored.
            port: Ignored.
            timeout_s: Ignored.

        Returns:
            The scripted peer.
        """
        return self.peer

    def __enter__(self) -> StubbedConnect:
        """Install the stub.

        Returns:
            This stub.
        """
        self._original = _test_hooks.connect
        _test_hooks.connect = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the original hook.

        Args:
            exc_type: Exception class raised in the block, if any.
            exc: Exception raised in the block, if any.
            traceback: Traceback of the raised exception, if any.
        """
        _test_hooks.connect = self._original


__all__ = [
    "enemy",
    "entity",
    "lines",
    "option",
    "player",
    "pool",
    "profile",
    "profiles_for",
    "repeated",
    "sample",
]
