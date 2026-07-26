"""The fight phase, driven against a scripted world.

What to attack is tested in ``test_policy_combat``; what is tested here is the
loop around it — when it stops, what it counts, and the one behaviour a pure
decision cannot express: not re-issuing an order that is already being carried
out.
"""

from __future__ import annotations

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.policy.campaign import EXPAND_RETRY_SAMPLES, fight, format_battle


def _unit(type_name: str, *, speed: float = 1.0, armed: bool = True, price: int = 350) -> UnitStats:
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
        weapon=(
            Weapon(
                shoot_delay=50.0,
                attack_range=110.0,
                direct_damage=17.0,
                direct_damage_volley=17.0,
                area_damage=0.0,
                area_damage_volley=0.0,
            )
            if armed
            else None
        ),
    )


_CATALOGUE = {
    "c_tank": _unit("c_tank"),
    "builder": _unit("builder", speed=0.6, armed=False),
    "commandCenter": _unit("commandCenter", speed=0.0, armed=False),
    # Priced as the engine's own dump prices it, so the reserve arithmetic
    # below is the arithmetic a real match does.
    "extractorT1": _unit("extractorT1", speed=0.0, armed=False, price=700),
    "landFactory": _unit("landFactory", speed=0.0, armed=False, price=1000),
}

#: Attack range by type name. Complete by contract, the unarmed at zero.
_REACHES = dict.fromkeys(_CATALOGUE, 0.0)


def _entity_line(
    frame: int,
    index: int,
    unit_id: int,
    type_name: str,
    *,
    mine: bool,
    x: float = 0.0,
    queued: int = 0,
) -> str:
    return (
        f'{{"kind":"entity","frame":{frame},"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":{x},"y":0.0,'
        f'"team":{0 if mine else 1},"mine":{str(mine).lower()},'
        f'"hostile":{str(not mine).lower()},"movement":"LAND","group":1,'
        f'"hp":100.0,"max_hp":100.0,"complete":true,"queued":{queued}}}'
    )


def _pool_line(frame: int, index: int, x: float) -> str:
    return (
        f'{{"kind":"pool","frame":{frame},"index":{index},'
        f'"tile_x":{int(x) // 20},"tile_y":0,"x":{x},"y":0.0,"group_land":1}}'
    )


def _option_line(frame: int, index: int, unit_id: int, produces: str, *, placed: bool) -> str:
    return (
        f'{{"kind":"option","frame":{frame},"index":{index},"unit_id":{unit_id},'
        f'"produces":"{produces}","action":1,"placed":{str(placed).lower()},'
        f'"available":true}}'
    )


def _sample_lines(
    frame: int,
    *entities: tuple[int, str, bool, float],
    pools: tuple[float, ...] = (),
    options: tuple[tuple[int, str, bool], ...] = (),
    credits_held: int = 4000,
    busy: tuple[int, ...] = (),
) -> list[str]:
    lines = [
        f'{{"kind":"frame","frame":{frame},"clock_ms":{frame * 3},'
        f'"visible":{len(entities)},"pools":{len(pools)},"options":{len(options)},'
        f'"credits":{credits_held},"defeated":false,"wiped":false,"players_left":6}}'
    ]
    for index, (unit_id, type_name, mine, x) in enumerate(entities):
        lines.append(
            _entity_line(
                frame,
                index,
                unit_id,
                type_name,
                mine=mine,
                x=x,
                queued=1 if unit_id in busy else 0,
            )
        )
    for index, pool_x in enumerate(pools):
        lines.append(_pool_line(frame, index, pool_x))
    for index, (unit_id, produces, placed) in enumerate(options):
        lines.append(_option_line(frame, index, unit_id, produces, placed=placed))
    return lines


class _ScriptedPeer:
    """Serves prepared lines and records what was sent back.

    Attributes:
        sent: Every line the loop wrote, in order.
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


_TANK = (1, "c_tank", True, 0.0)
_ENEMY = (9, "c_tank", False, 100.0)

#: The rest of a first wave, so an attack is allowed to happen at all.
#:
#: Nothing is sent until the army reaches :data:`~rw_bot.policy.combat.FIRST_WAVE`
#: ([[engine-ai-triggers]]). Every test below is about *which* target is chosen
#: or *when* an order repeats, not about the threshold, so they field a full
#: wave and let the one test that is about the threshold stand on its own.
_WAVE = (_TANK, (2, "c_tank", True, 0.0), (3, "c_tank", True, 0.0))


def test_the_army_is_sent_at_the_enemy() -> None:
    peer = _ScriptedPeer(_sample_lines(1, *_WAVE, _ENEMY) * 3)
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=3)
    assert _orders(peer) == [
        '{"kind":"attack","unit_id":1,"target_id":9}',
        '{"kind":"attack","unit_id":2,"target_id":9}',
        '{"kind":"attack","unit_id":3,"target_id":9}',
    ]
    assert battle["orders_sent"] == len(_WAVE)


def test_the_reserve_is_sent_to_gather_at_the_base() -> None:
    """A wave that starts scattered arrives scattered, gate or no gate.

    Units roll out wherever the factory is and then stand there. Rallying them
    is what makes the released wave one force rather than a queue of arrivals,
    and it doubles as the only defensive posture the bot has: units waiting at
    the base stand between an attacker and it.
    """
    base = (99, "commandCenter", True, 0.0)
    far = (1, "c_tank", True, 900.0)
    peer = _ScriptedPeer(_sample_lines(1, base, far, _ENEMY) * 3)
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=3)
    assert '{"kind":"move","unit_id":1,"x":0.0,"y":0.0}' in peer.sent
    assert battle["rallied"] == 1


def test_a_unit_already_at_the_base_is_not_told_to_go_there() -> None:
    base = (99, "commandCenter", True, 0.0)
    home = (1, "c_tank", True, 10.0)
    peer = _ScriptedPeer(_sample_lines(1, base, home, _ENEMY) * 3)
    assert fight(AgentChannel(peer), _CATALOGUE, max_samples=3)["rallied"] == 0


def test_a_rally_order_is_sent_once_rather_than_every_sample() -> None:
    """Re-issuing would reset the walk at the sampling rate."""
    base = (99, "commandCenter", True, 0.0)
    far = (1, "c_tank", True, 900.0)
    peer = _ScriptedPeer(_sample_lines(1, base, far, _ENEMY) * 20)
    assert fight(AgentChannel(peer), _CATALOGUE, max_samples=20)["rallied"] == 1


def test_a_force_short_of_a_wave_gathers_instead_of_attacking() -> None:
    """One tank in sight of an enemy sends nothing at all.

    Attacking with whatever exists feeds units in one at a time and loses each
    of them separately. The threshold is the shipped AI's: its combat groups are
    created empty with a target size and do not move until full
    ([[engine-ai-triggers]]).
    """
    peer = _ScriptedPeer(_sample_lines(1, _TANK, _ENEMY) * 5)
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=5)
    assert _orders(peer) == []
    assert battle["orders_sent"] == 0
    assert battle["samples_seen"] == 5


def test_a_committed_force_keeps_going_after_losses() -> None:
    """The latch. Survivors do not turn round to wait for reinforcements.

    Without it a wave that took casualties would break off mid-engagement while
    still standing in range of the thing that shot it.
    """
    peer = _ScriptedPeer(
        _sample_lines(1, *_WAVE, _ENEMY) + _sample_lines(2, _TANK, (10, "c_tank", False, 50.0)) * 3
    )
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=4)
    # The survivor is re-tasked onto the new target rather than idling, which
    # only happens because commitment outlived the wave that earned it.
    assert '{"kind":"attack","unit_id":1,"target_id":10}' in _orders(peer)
    assert battle["orders_sent"] > len(_WAVE)


def test_an_order_already_in_flight_is_not_re_issued() -> None:
    """The engine runs a waypoint until it is replaced.

    Re-sending the same attack every sample would replace an in-progress order
    with an identical one at the sampling rate, and the unit would never close
    the distance.
    """
    peer = _ScriptedPeer(_sample_lines(1, *_WAVE, _ENEMY) * 20)
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=20)
    assert battle["orders_sent"] == len(_WAVE)
    assert battle["samples_seen"] == 20


def test_a_nearer_enemy_does_not_pull_the_army_off_its_target() -> None:
    """Commitment across samples, which is what stopped the churn.

    A closer enemy appearing is not a reason to abandon a target already being
    shot at; re-choosing on every sample spent 743 orders on 24 targets.
    """
    peer = _ScriptedPeer(
        _sample_lines(1, *_WAVE, _ENEMY)
        + _sample_lines(2, *_WAVE, _ENEMY, (10, "c_tank", False, 5.0)) * 5
    )
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=6)
    assert {order.split('"target_id":')[1] for order in _orders(peer)} == {"9}"}
    assert battle["orders_sent"] == len(_WAVE)


def test_a_new_target_earns_a_new_order() -> None:
    """The first enemy dies, so the army is re-committed rather than idling."""
    peer = _ScriptedPeer(
        _sample_lines(1, *_WAVE, _ENEMY) + _sample_lines(2, *_WAVE, (10, "c_tank", False, 50.0))
    )
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=2)
    assert _orders(peer) == [
        '{"kind":"attack","unit_id":1,"target_id":9}',
        '{"kind":"attack","unit_id":2,"target_id":9}',
        '{"kind":"attack","unit_id":3,"target_id":9}',
        '{"kind":"attack","unit_id":1,"target_id":10}',
        '{"kind":"attack","unit_id":2,"target_id":10}',
        '{"kind":"attack","unit_id":3,"target_id":10}',
    ]
    assert battle["orders_sent"] == 2 * len(_WAVE)


def test_no_enemy_left_is_cleared() -> None:
    peer = _ScriptedPeer(_sample_lines(1, *_WAVE))
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=5)
    assert battle["outcome"] == "cleared"
    assert _orders(peer) == []


def test_no_army_left_is_reported_apart_from_clearing_the_field() -> None:
    """Losing and winning must not read the same in the run log."""
    peer = _ScriptedPeer(_sample_lines(1, _ENEMY))
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=5)
    assert battle["outcome"] == "no_army"
    assert battle["army_start"] == 0


def test_an_unarmed_roster_is_no_army() -> None:
    peer = _ScriptedPeer(_sample_lines(1, (2, "builder", True, 0.0), _ENEMY))
    assert fight(AgentChannel(peer), _CATALOGUE, max_samples=5)["outcome"] == "no_army"


def test_the_sample_budget_bounds_a_fight_that_never_resolves() -> None:
    peer = _ScriptedPeer(_sample_lines(1, *_WAVE, _ENEMY) * 10)
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=4)
    assert battle["outcome"] == "sample_limit"
    assert battle["samples_seen"] == 4


def test_an_engaged_target_that_disappears_is_counted() -> None:
    """Counted as gone rather than as killed.

    A target that retreated into fog reads identically from here, and the
    scorecard must not claim a kill it cannot see.
    """
    peer = _ScriptedPeer(_sample_lines(1, *_WAVE, _ENEMY) + _sample_lines(2, *_WAVE))
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=5)
    assert battle["killed"] == 1
    assert battle["outcome"] == "cleared"


def test_losses_are_replaced_while_the_fight_runs() -> None:
    """Without this the bot commits a fixed force and is finished when it is.

    The opponents replace losses continuously; a sortie that cannot be
    reinforced is how four tanks were lost to nothing.
    """
    factory = (300, "landFactory", True, 0.0)
    option = (
        '{"kind":"option","frame":1,"index":0,"unit_id":300,'
        '"produces":"c_tank","action":1,"placed":false,"available":true}'
    )
    lines = _sample_lines(1, *_WAVE, _ENEMY, factory)
    lines.insert(1, option)
    lines[0] = lines[0].replace('"options":0', '"options":1')
    peer = _ScriptedPeer(lines * 2)
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=2, reinforce=("c_tank",))
    assert battle["produced"] == 2
    assert '{"kind":"produce","unit_id":300,"type":"c_tank"}' in _orders(peer)


def test_nothing_is_reinforced_when_nothing_is_wanted() -> None:
    peer = _ScriptedPeer(_sample_lines(1, *_WAVE, _ENEMY) * 2)
    assert fight(AgentChannel(peer), _CATALOGUE, max_samples=2)["produced"] == 0


_BUILDER = (214, "builder", True, 0.0)
_CENTRE = (213, "commandCenter", True, 0.0)

#: The engine's own answer to "who can place an extractor", as it rides on the
#: wire. Placement is a build order rather than a produce order, which is why
#: continuous production could never have taken a pool by itself.
_CAN_PLACE: tuple[tuple[int, str, bool], ...] = (
    (214, "extractorT1", True),
    (214, "landFactory", True),
)

#: A finished Land Factory, and the produce option that makes it a producer.
#:
#: Placement and production are different questions to the engine and to us: a
#: unit rolls out (``placed`` false), a structure is sited (``placed`` true).
#: Only the former counts as queue capacity ([[mechanics-build-actions]]).
_FACTORY = (300, "landFactory", True, 0.0)
_MAKES_TANKS: tuple[tuple[int, str, bool], ...] = ((300, "c_tank", False),)

_CLAIMED = '{"kind":"build","unit_id":214,"x":300.0,"y":0.0,"type":"extractorT1"}'


def _builds(peer: _ScriptedPeer) -> list[str]:
    """Only the build orders, so attack traffic does not drown the economy."""
    return [line for line in _orders(peer) if '"kind":"build"' in line]


def test_a_free_pool_is_claimed_while_the_fight_runs() -> None:
    """The whole point: the economy no longer stops when the plan does.

    Three extractors funded the entire match on a map carrying 46 pools, and
    nothing in the bot could take a fourth -- production refuses placed
    structures and reinforcement drops anything needing a pool, both correctly
    ([[policy-economy]]).
    """
    peer = _ScriptedPeer(
        _sample_lines(1, *_WAVE, _BUILDER, _ENEMY, pools=(300.0,), options=_CAN_PLACE) * 2
    )
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=2, reaches=_REACHES, reserve=0)
    assert _builds(peer) == [_CLAIMED]
    assert battle["expanded"] == 1
    assert battle["expand_reason"] == "extractorT1 #1 at (300, 0)"


def test_a_builder_already_walking_to_a_site_is_not_re_tasked() -> None:
    """A builder in transit is an order still being carried out.

    Re-sending would replace the waypoint with a copy of itself and the builder
    would never arrive, which is the same defect commitment fixed on the combat
    side.
    """
    walking = (214, "builder", True, 60.0)
    peer = _ScriptedPeer(
        _sample_lines(1, *_WAVE, _BUILDER, _ENEMY, pools=(300.0,), options=_CAN_PLACE)
        + _sample_lines(2, *_WAVE, walking, _ENEMY, pools=(300.0,), options=_CAN_PLACE)
    )
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=2, reaches=_REACHES, reserve=0)
    assert _builds(peer) == [_CLAIMED]
    assert battle["expand_reason"] == "builder still walking to its site"


def test_the_same_site_is_not_ordered_every_sample() -> None:
    peer = _ScriptedPeer(
        _sample_lines(1, *_WAVE, _BUILDER, _ENEMY, pools=(300.0,), options=_CAN_PLACE) * 5
    )
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=5, reaches=_REACHES, reserve=0)
    assert battle["expanded"] == 1


def test_an_order_that_never_took_is_eventually_sent_again() -> None:
    """A pool that is never retried is an economy that stops for good.

    The builder here is ordered, then stands still and starts nothing -- a
    refused or lost order. Suppressing the repeat forever would be the safe
    read and the wrong one.
    """
    samples = EXPAND_RETRY_SAMPLES + 2
    peer = _ScriptedPeer(
        _sample_lines(1, *_WAVE, _BUILDER, _ENEMY, pools=(300.0,), options=_CAN_PLACE) * samples
    )
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=samples, reaches=_REACHES, reserve=0)
    assert _builds(peer) == [_CLAIMED, _CLAIMED]
    assert battle["expanded"] == 2


def test_a_reserve_that_outruns_the_purse_leaves_the_pool_alone() -> None:
    peer = _ScriptedPeer(
        _sample_lines(
            1, *_WAVE, _BUILDER, _ENEMY, pools=(300.0,), options=_CAN_PLACE, credits_held=800
        )
    )
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=1, reaches=_REACHES, reserve=350)
    assert _builds(peer) == []
    assert battle["expand_reason"] == "800 credits, need 1050 to expand past a 350 reserve"


def test_a_lost_builder_is_replaced_so_the_economy_can_restart() -> None:
    """Without this a dead builder ends expansion for the rest of the match.

    The command centre is asked because it is the only producer that cannot
    make a tank, so the builder sits last in the preference order and the
    factories are never diverted onto one.
    """
    peer = _ScriptedPeer(
        _sample_lines(1, *_WAVE, _CENTRE, _ENEMY, options=((213, "builder", False),)) * 2
    )
    battle = fight(
        AgentChannel(peer),
        _CATALOGUE,
        max_samples=2,
        reinforce=("c_tank",),
        reaches=_REACHES,
        reserve=0,
    )
    assert '{"kind":"produce","unit_id":213,"type":"builder"}' in _orders(peer)
    assert battle["expand_reason"] == "nothing owned can place extractorT1"


def test_no_builder_is_made_while_one_is_alive() -> None:
    """The replacement is conditional, or the command centre makes them forever."""
    peer = _ScriptedPeer(
        _sample_lines(1, *_WAVE, _BUILDER, _CENTRE, _ENEMY, options=((213, "builder", False),)) * 2
    )
    fight(
        AgentChannel(peer),
        _CATALOGUE,
        max_samples=2,
        reinforce=("c_tank",),
        reaches=_REACHES,
        reserve=0,
    )
    assert [line for line in _orders(peer) if '"type":"builder"' in line] == []


def test_throughput_is_bought_before_more_income() -> None:
    """A pool is worthless while the credits it earns have nowhere to go.

    Every producer here is busy and the bank covers a factory, which is the
    exact state that banked 7,013 credits behind a single Land Factory. The
    free pool beside it is deliberately left alone: another extractor would
    earn credits the player already cannot spend ([[policy-economy]]).
    """
    peer = _ScriptedPeer(
        _sample_lines(
            1,
            *_WAVE,
            _BUILDER,
            _FACTORY,
            _ENEMY,
            pools=(300.0,),
            options=_CAN_PLACE + _MAKES_TANKS,
            credits_held=4000,
            busy=(300,),
        )
    )
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=1, reaches=_REACHES, reserve=0)
    assert _builds(peer) == [
        '{"kind":"build","unit_id":214,"x":-200.0,"y":120.0,"type":"landFactory"}'
    ]
    assert battle["expand_reason"] == "every producer busy on 4000 credits; adding a landFactory"


def test_a_free_producer_means_the_pool_wins() -> None:
    """The fall-through. Spare queue capacity makes another factory pointless.

    Same world, except the factory's queue is empty. Another factory would idle
    beside the one already idling, so the credits go into income instead.
    """
    peer = _ScriptedPeer(
        _sample_lines(
            1,
            *_WAVE,
            _BUILDER,
            _FACTORY,
            _ENEMY,
            pools=(300.0,),
            options=_CAN_PLACE + _MAKES_TANKS,
            credits_held=4000,
        )
    )
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=1, reaches=_REACHES, reserve=0)
    assert _builds(peer) == [_CLAIMED]
    assert battle["expanded"] == 1


def test_extractors_standing_are_counted_at_both_ends() -> None:
    peer = _ScriptedPeer(_sample_lines(1, *_WAVE, (5, "extractorT1", True, 500.0), _ENEMY) * 2)
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=2, reaches=_REACHES, reserve=0)
    assert battle["extractors_start"] == 1
    assert battle["extractors_end"] == 1


def test_the_report_renders_every_figure() -> None:
    peer = _ScriptedPeer(_sample_lines(1, *_WAVE, _ENEMY) * 2)
    lines = format_battle(fight(AgentChannel(peer), _CATALOGUE, max_samples=2))
    assert lines == (
        "fight outcome  sample_limit",
        "attack orders  3",
        "reinforced     0",
        "army           3 -> 3",
        "enemies seen   1 -> 1",
        "engaged gone   0",
        # A caller that passes no reach table is not playing an economy, and the
        # report says so rather than showing a zero that could be read as
        # "expansion was tried and found nothing".
        "extractors     0 -> 0",
        "expansions     0",
        "expand note    expansion disabled",
        "samples seen   2",
    )
