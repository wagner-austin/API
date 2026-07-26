"""The policy loop, driven against a scripted world.

The deciding is tested in ``test_policy_build_order``; what is tested here is
the loop around it — when it stops, what it counts, and the one behaviour a
pure decision function cannot express on its own: not re-ordering a structure
that is still being built.
"""

from __future__ import annotations

from collections.abc import Mapping

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.runner import format_scorecard, run


def _unit(type_name: str, price: int, speed: float = 0.0) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=price,
        hp=100,
        speed=speed,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


def _place(type_name: str, needs_pool: bool = False) -> TypePlacement:
    return TypePlacement(index=0, type_name=type_name, needs_pool=needs_pool)


_CATALOGUE = {
    "landFactory": _unit("landFactory", 300),
    "laboratory": _unit("laboratory", 900),
    "extractorT1": _unit("extractorT1", 700),
    # Mobile, which is load-bearing rather than decorative: the builder ends
    # every build standing on the site it just used, and a mobile unit must not
    # make that site read as occupied.
    "builder": _unit("builder", 500, speed=0.6),
    # Priced as the live catalogue prices it, because the produce window is
    # derived from the price rather than fixed -- a made-up figure here would
    # make the window tests assert nothing.
    "scout": _unit("scout", 700, speed=1.0),
}

_PLACEMENTS = {
    "landFactory": _place("landFactory"),
    "laboratory": _place("laboratory"),
    "extractorT1": _place("extractorT1", needs_pool=True),
    "scout": _place("scout"),
}

#: Attack range by type name, as the registry dump gives it.
#:
#: Every type these worlds can name appears, armed or not, mirroring the real
#: dump's coverage of all 173 registered types ([[policy-threat]]).
_REACHES: dict[str, float] = dict.fromkeys(_CATALOGUE, 0.0)


#: Where an entity stands unless a test says otherwise. Arbitrary, but fixed:
#: most tests are about counting rather than geometry, and a builder that never
#: moves is what keeps the stall clock running in those.
_DEFAULT_AT = (100.0, 200.0)


def _entity_line(
    frame: int,
    index: int,
    unit_id: int,
    type_name: str,
    mine: bool,
    at: tuple[float, float],
    queued: int = 0,
    complete: bool = True,
) -> str:
    return (
        f'{{"kind":"entity","frame":{frame},"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":{at[0]},"y":{at[1]},'
        f'"team":{0 if mine else 1},"mine":{str(mine).lower()},'
        f'"hostile":{str(not mine).lower()},"movement":"LAND","group":1,'
        f'"hp":100.0,"max_hp":100.0,"complete":{str(complete).lower()},"queued":{queued}}}'
    )


def _pool_line(frame: int, index: int, tile_x: int, tile_y: int) -> str:
    return (
        f'{{"kind":"pool","frame":{frame},"index":{index},'
        f'"tile_x":{tile_x},"tile_y":{tile_y},'
        f'"x":{tile_x * 20 + 10}.0,"y":{tile_y * 20 + 10}.0,"group_land":1}}'
    )


#: What the Builder offers by default in these worlds, mirroring the capture.
_BUILDER_OFFERS = ("landFactory", "airFactory", "extractorT1", "laboratory")


def _option_line(frame: int, index: int, unit_id: int, produces: str, placed: bool) -> str:
    return (
        f'{{"kind":"option","frame":{frame},"index":{index},"unit_id":{unit_id},'
        f'"produces":"{produces}","action":1,'
        f'"placed":{str(placed).lower()},"available":true}}'
    )


def _sample_lines(
    frame: int,
    credits: int,
    *entities: tuple[int, str, bool],
    pools: tuple[tuple[int, int], ...] = (),
    at: Mapping[int, tuple[float, float]] | None = None,
    options: tuple[tuple[int, str, bool], ...] | None = None,
    queued: Mapping[int, int] | None = None,
    complete: Mapping[int, bool] | None = None,
) -> list[str]:
    """Render one sample.

    Args:
        frame: Engine frame counter.
        credits: Credits held.
        entities: ``(unit_id, type_name, mine)`` per visible entity.
        pools: Tile coordinates of the visible resource pools.
        at: Positions by unit id, for the entities whose position matters.
            Anything absent stands at :data:`_DEFAULT_AT`.
        options: ``(unit_id, produces, placed)`` per build option the player's
            units offer. Defaults to the Builder offering the structures these
            runs order, which is what the live capture shows it offering.
        queued: Units queued for production, by unit id. Anything absent has an
            empty queue, which is what the live capture shows for a building
            that is not making anything.
        complete: Construction state by unit id. Anything absent is finished,
            since most tests are not about the construction window.

    Returns:
        The sample's NDJSON lines.
    """
    if options is None:
        options = tuple((214, name, True) for name in _BUILDER_OFFERS)
    positions = at if at is not None else {}
    queues = queued if queued is not None else {}
    done = complete if complete is not None else {}
    lines = [
        f'{{"kind":"frame","frame":{frame},"clock_ms":{frame * 3},'
        f'"visible":{len(entities)},"pools":{len(pools)},'
        f'"options":{len(options)},"credits":{credits}}}'
    ]
    for index, (unit_id, type_name, mine) in enumerate(entities):
        lines.append(
            _entity_line(
                frame,
                index,
                unit_id,
                type_name,
                mine,
                positions.get(unit_id, _DEFAULT_AT),
                queues.get(unit_id, 0),
                done.get(unit_id, True),
            )
        )
    for index, (tile_x, tile_y) in enumerate(pools):
        lines.append(_pool_line(frame, index, tile_x, tile_y))
    for index, (unit_id, produces, placed) in enumerate(options):
        lines.append(_option_line(frame, index, unit_id, produces, placed))
    return lines


_BUILDER = (214, "builder", True)


class _ScriptedPeer:
    """Serves prepared lines and records what was sent back.

    Attributes:
        sent: Every line the runner wrote, in order.
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


def _orders(peer: _ScriptedPeer) -> list[str]:
    """Everything the loop sent except the per-sample acknowledgements.

    The ack is protocol rather than policy -- it tells the agent the sample is
    finished with, and in lockstep it is what releases the simulation
    ([[policy-determinism]]). Assertions here are about what the bot decided,
    so the acks are filtered out rather than woven into every expectation.
    """
    return [line for line in peer.sent if '"kind":"ack"' not in line]


def test_a_plan_already_satisfied_finishes_without_ordering() -> None:
    peer = _ScriptedPeer(_sample_lines(1, 4000, _BUILDER, (300, "landFactory", True)))
    card = run(
        AgentChannel(peer), ("landFactory",), _CATALOGUE, _PLACEMENTS, _REACHES, max_samples=5
    )
    assert card["outcome"] == "done"
    assert card["completed"] == 1
    assert _orders(peer) == []


def test_one_order_is_sent_and_the_structure_ends_the_plan() -> None:
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(10, 3700, _BUILDER, (300, "landFactory", True))
    )
    card = run(
        AgentChannel(peer), ("landFactory",), _CATALOGUE, _PLACEMENTS, _REACHES, max_samples=5
    )
    assert card["outcome"] == "done"
    assert card["completed"] == 1
    assert card["orders_sent"] == 1
    assert _orders(peer) == [
        '{"kind":"build","unit_id":214,"x":300.0,"y":320.0,"type":"landFactory"}'
    ]


def test_a_structure_still_being_built_is_not_re_ordered() -> None:
    """Three samples pass before it appears; only one order may be sent."""
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(2, 3700, _BUILDER)
        + _sample_lines(3, 3700, _BUILDER)
        + _sample_lines(4, 3700, _BUILDER, (300, "landFactory", True))
    )
    card = run(
        AgentChannel(peer), ("landFactory",), _CATALOGUE, _PLACEMENTS, _REACHES, max_samples=8
    )
    assert card["orders_sent"] == 1
    assert card["completed"] == 1
    assert card["outcome"] == "done"


def test_waiting_for_credits_sends_nothing_and_keeps_reading() -> None:
    peer = _ScriptedPeer(
        _sample_lines(1, 100, _BUILDER)
        + _sample_lines(2, 500, _BUILDER)
        + _sample_lines(3, 900, _BUILDER)
        + _sample_lines(4, 900, _BUILDER, (300, "laboratory", True))
    )
    card = run(
        AgentChannel(peer), ("laboratory",), _CATALOGUE, _PLACEMENTS, _REACHES, max_samples=8
    )
    assert card["orders_sent"] == 1
    assert card["outcome"] == "done"


def test_a_blocked_plan_stops_immediately() -> None:
    peer = _ScriptedPeer(_sample_lines(1, 4000, (213, "commandCenter", True)))
    card = run(
        AgentChannel(peer), ("landFactory",), _CATALOGUE, _PLACEMENTS, _REACHES, max_samples=9
    )
    assert card["outcome"] == "blocked"
    assert card["last_reason"] == "the player owns no builder"
    assert card["samples_seen"] == 1


def test_the_sample_budget_bounds_a_run_that_never_finishes() -> None:
    peer = _ScriptedPeer(
        _sample_lines(1, 10, _BUILDER)
        + _sample_lines(2, 10, _BUILDER)
        + _sample_lines(3, 10, _BUILDER)
    )
    card = run(
        AgentChannel(peer), ("laboratory",), _CATALOGUE, _PLACEMENTS, _REACHES, max_samples=3
    )
    assert card["outcome"] == "sample_limit"
    assert card["samples_seen"] == 3
    assert card["orders_sent"] == 0


def test_frames_elapsed_spans_the_run() -> None:
    peer = _ScriptedPeer(
        _sample_lines(100, 4000, _BUILDER)
        + _sample_lines(460, 3700, _BUILDER, (300, "landFactory", True))
    )
    card = run(
        AgentChannel(peer), ("landFactory",), _CATALOGUE, _PLACEMENTS, _REACHES, max_samples=5
    )
    assert card["frames_elapsed"] == 360
    assert card["credits_at_end"] == 3700


def test_a_zero_sample_budget_reports_that_nothing_was_read() -> None:
    card = run(
        AgentChannel(_ScriptedPeer([])),
        ("landFactory",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=0,
    )
    assert card["outcome"] == "sample_limit"
    assert card["samples_seen"] == 0
    assert card["last_reason"] == "no sample was read"


def test_two_structures_are_ordered_in_plan_sequence() -> None:
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(2, 3700, _BUILDER, (300, "landFactory", True))
        + _sample_lines(3, 2800, _BUILDER, (300, "landFactory", True), (301, "laboratory", True))
    )
    card = run(
        AgentChannel(peer),
        ("landFactory", "laboratory"),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=6,
    )
    assert card["completed"] == 2
    assert card["orders_sent"] == 2
    assert [line.split('"type":"')[1].rstrip('"}') for line in _orders(peer)] == [
        "landFactory",
        "laboratory",
    ]


def test_the_scorecard_renders_every_figure() -> None:
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(10, 3700, _BUILDER, (300, "landFactory", True))
    )
    card = run(
        AgentChannel(peer), ("landFactory",), _CATALOGUE, _PLACEMENTS, _REACHES, max_samples=5
    )
    assert format_scorecard(card) == (
        "outcome        done (all 1 plan entries satisfied)",
        "completed      1/1",
        "orders sent    1",
        "samples seen   2",
        "frames elapsed 9",
        "credits left   3700",
    )


def test_an_order_the_engine_refuses_is_reported_as_stalled() -> None:
    """Observed for real: a builder cannot build a laboratory, and the engine
    says so only in its own log. Without this the run reports "building
    laboratory" forever while nothing happens."""
    peer = _ScriptedPeer(_sample_lines(1, 4000, _BUILDER) * 10)
    card = run(
        AgentChannel(peer),
        ("laboratory",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=20,
        stall_samples=4,
    )
    assert card["outcome"] == "stalled"
    assert card["orders_sent"] == 1
    assert card["completed"] == 0
    assert "laboratory was ordered but never appeared after 4 samples" in card["last_reason"]


def test_a_produced_unit_leaves_as_a_produce_order_carrying_no_position() -> None:
    """Two verbs, because the engine has two.

    A structure is placed where the planner chooses; a unit rolls out of the
    building that made it. Sending a build order for the second would offer the
    engine a coordinate it does not want.
    """
    centre = (213, "commandCenter", True)
    peer = _ScriptedPeer(_sample_lines(1, 4000, centre, options=((213, "scout", False),)))
    card = run(
        AgentChannel(peer),
        ("scout",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=1,
    )
    assert _orders(peer) == ['{"kind":"produce","unit_id":213,"type":"scout"}']
    assert card["orders_sent"] == 1


def test_a_building_with_the_order_in_its_queue_is_not_called_stalled() -> None:
    """The production counterpart of a builder still walking to its site.

    A factory never moves, so the movement test alone would call a working one
    refused after the window expired. The building reports what it is holding,
    and that is the signal -- measured live, a Command Center read ``queued: 1``
    for all forty-five samples a Scout took and dropped to zero on the sample it
    appeared.
    """
    centre = (213, "commandCenter", True)
    peer = _ScriptedPeer(
        _sample_lines(
            1,
            9000,
            centre,
            options=((213, "scout", False),),
            queued={213: 1},
        )
        * 40
    )
    card = run(
        AgentChannel(peer),
        ("scout",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=30,
        stall_samples=4,
    )
    assert card["outcome"] == "sample_limit"
    assert card["orders_sent"] == 1


def test_a_structure_going_up_is_not_called_stalled() -> None:
    """The regression that not counting unfinished structures would introduce.

    The builder stops moving the moment it arrives, and the structure joins the
    roster unfinished at about the same time. So movement stops being evidence
    exactly when construction starts, and without the second half of the
    in-flight test the run would call a rising factory refused.
    """
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(2, 4000, _BUILDER, (300, "landFactory", True), complete={300: False}) * 40
    )
    card = run(
        AgentChannel(peer),
        ("landFactory",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=20,
        stall_samples=3,
    )
    assert card["outcome"] == "sample_limit"
    assert card["orders_sent"] == 1
    assert card["completed"] == 0


def test_a_structure_that_finishes_ends_the_plan() -> None:
    """The same run, with the flag flipping, completes rather than timing out."""
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(2, 4000, _BUILDER, (300, "landFactory", True), complete={300: False})
        + _sample_lines(3, 4000, _BUILDER, (300, "landFactory", True))
    )
    card = run(
        AgentChannel(peer),
        ("landFactory",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=20,
        stall_samples=3,
    )
    assert card["outcome"] == "done"
    assert card["completed"] == 1
    assert card["orders_sent"] == 1


def test_an_opponents_half_built_structure_does_not_keep_our_clock_alive() -> None:
    """Ownership is checked, or an enemy site in view would suspend the stall."""
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(2, 4000, _BUILDER, (900, "landFactory", False), complete={900: False}) * 40
    )
    card = run(
        AgentChannel(peer),
        ("landFactory",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=20,
        stall_samples=3,
    )
    assert card["outcome"] == "stalled"


def test_the_producer_is_found_wherever_it_sits_in_the_roster() -> None:
    """Roster order is enumeration order and renumbers constantly.

    The queue has to be read off the unit the order was addressed to, so a
    producer standing behind other entities must still be found.
    """
    centre = (213, "commandCenter", True)
    peer = _ScriptedPeer(
        _sample_lines(
            1,
            9000,
            _BUILDER,
            centre,
            options=((213, "scout", False),),
            queued={213: 1},
        )
        * 40
    )
    card = run(
        AgentChannel(peer),
        ("scout",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=30,
        stall_samples=4,
    )
    assert card["outcome"] == "sample_limit"


def test_a_producer_holding_nothing_is_called_stalled() -> None:
    """An empty queue after an order means the engine refused it."""
    centre = (213, "commandCenter", True)
    peer = _ScriptedPeer(_sample_lines(1, 9000, centre, options=((213, "scout", False),)) * 40)
    card = run(
        AgentChannel(peer),
        ("scout",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=30,
        stall_samples=4,
    )
    assert card["outcome"] == "stalled"
    assert "scout was ordered but never appeared" in card["last_reason"]


def test_a_producer_destroyed_mid_order_stalls_rather_than_waiting_forever() -> None:
    """No producer in the roster is no evidence of progress.

    Treating a missing building as "still working" would hang the run on
    something that no longer exists, which is the same trap a missing builder
    sets for the movement test.
    """
    centre = (213, "commandCenter", True)
    peer = _ScriptedPeer(
        _sample_lines(1, 9000, centre, options=((213, "scout", False),))
        + _sample_lines(2, 9000, options=((213, "scout", False),)) * 40
    )
    card = run(
        AgentChannel(peer),
        ("scout",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=30,
        stall_samples=4,
    )
    assert card["outcome"] == "stalled"


def test_an_extractor_is_ordered_onto_a_pool_and_the_next_onto_a_different_one() -> None:
    """Two extractors must not both be sent to the nearest pool.

    The first order puts an extractor on the pool, and from the next sample on
    that pool reads as occupied, so the second order has to move to the other
    one. This is the whole reason occupancy is judged from the roster rather
    than tracked in the planner.
    """
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER, pools=((5, 10), (30, 10)))
        + _sample_lines(2, 3300, _BUILDER, (300, "extractorT1", True), pools=((5, 10), (30, 10)))
        + _sample_lines(
            3,
            2600,
            _BUILDER,
            (300, "extractorT1", True),
            (301, "extractorT1", True),
            pools=((5, 10), (30, 10)),
        )
    )
    card = run(
        AgentChannel(peer),
        ("extractorT1", "extractorT1"),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=6,
    )
    assert card["outcome"] == "done"
    assert card["orders_sent"] == 2
    # The builder sits at (100, 200) and both pools are equally unoccupied at
    # the first sample, so the nearer one is chosen: tile (5, 10) centres on
    # (110, 210). The second extractor then stands on it, pushing the next
    # order out to tile (30, 10) at (610, 210).
    assert _orders(peer) == [
        '{"kind":"build","unit_id":214,"x":110.0,"y":210.0,"type":"extractorT1"}',
        '{"kind":"build","unit_id":214,"x":610.0,"y":210.0,"type":"extractorT1"}',
    ]


def test_an_extractor_with_no_pool_in_sight_waits_rather_than_ordering() -> None:
    """Fog lifts as units move, so this is a wait the world can resolve."""
    peer = _ScriptedPeer(_sample_lines(1, 4000, _BUILDER) * 3)
    card = run(
        AgentChannel(peer),
        ("extractorT1",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=3,
    )
    assert card["outcome"] == "sample_limit"
    assert card["orders_sent"] == 0
    assert card["last_reason"] == "extractorT1 needs a resource pool and none is visible yet"


def test_a_slow_structure_inside_the_stall_window_is_not_called_stalled() -> None:
    """The window has to tolerate a build that is merely slow."""
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(2, 3700, _BUILDER)
        + _sample_lines(3, 3700, _BUILDER)
        + _sample_lines(4, 3700, _BUILDER, (300, "landFactory", True))
    )
    card = run(
        AgentChannel(peer),
        ("landFactory",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=20,
        stall_samples=5,
    )
    assert card["outcome"] == "done"


def test_a_builder_still_walking_to_a_far_site_is_not_called_stalled() -> None:
    """The bug this replaced silently capped how far the bot could build.

    A stall window counted from the moment of the order reaches only as far as
    a builder can walk within it. Observed live: an order to a resource pool 762
    world units away was declared refused while the builder was still walking,
    and the extractor finished seconds after the run gave up. Movement is what
    distinguishes an order in flight from one the engine threw away.
    """
    walking = [
        line
        for step in range(12)
        for line in _sample_lines(
            step + 2,
            4000,
            _BUILDER,
            pools=((5, 10),),
            at={214: (100.0 + 12.0 * step, 200.0)},
        )
    ]
    arrived = _sample_lines(
        20,
        3300,
        _BUILDER,
        (300, "extractorT1", True),
        pools=((5, 10),),
        at={214: (232.0, 200.0)},
    )
    peer = _ScriptedPeer(_sample_lines(1, 4000, _BUILDER, pools=((5, 10),)) + walking + arrived)
    card = run(
        AgentChannel(peer),
        ("extractorT1",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=30,
        stall_samples=4,
    )
    assert card["outcome"] == "done"
    assert card["orders_sent"] == 1


def test_a_builder_that_stops_without_building_is_still_called_stalled() -> None:
    """Movement resets the clock; standing still must not reset it forever."""
    walking = [
        line
        for step in range(3)
        for line in _sample_lines(
            step + 2, 4000, _BUILDER, pools=((5, 10),), at={214: (100.0 + 12.0 * step, 200.0)}
        )
    ]
    parked = [
        line
        for step in range(12)
        for line in _sample_lines(
            step + 10, 4000, _BUILDER, pools=((5, 10),), at={214: (124.0, 200.0)}
        )
    ]
    peer = _ScriptedPeer(_sample_lines(1, 4000, _BUILDER, pools=((5, 10),)) + walking + parked)
    card = run(
        AgentChannel(peer),
        ("extractorT1",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=30,
        stall_samples=4,
    )
    assert card["outcome"] == "stalled"
    assert "standing still" in card["last_reason"]


def test_a_builder_lost_mid_order_blocks_rather_than_waiting_forever() -> None:
    """A missing builder is not movement, or a lost builder would wait forever."""
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER, pools=((5, 10),))
        + _sample_lines(2, 4000, (213, "commandCenter", True), pools=((5, 10),))
    )
    card = run(
        AgentChannel(peer),
        ("extractorT1",),
        _CATALOGUE,
        _PLACEMENTS,
        _REACHES,
        max_samples=30,
        stall_samples=4,
    )
    assert card["outcome"] == "blocked"
    assert card["last_reason"] == "the player owns no builder"
