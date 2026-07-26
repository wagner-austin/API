"""The fight phase, driven against a scripted world.

What to attack is tested in ``test_policy_combat``; what is tested here is the
loop around it — when it stops, what it counts, and the one behaviour a pure
decision cannot express: not re-issuing an order that is already being carried
out.
"""

from __future__ import annotations

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.policy.campaign import fight, format_battle


def _unit(type_name: str, *, speed: float = 1.0, armed: bool = True) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=350,
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
}


def _entity_line(
    frame: int, index: int, unit_id: int, type_name: str, *, mine: bool, x: float = 0.0
) -> str:
    return (
        f'{{"kind":"entity","frame":{frame},"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":{x},"y":0.0,'
        f'"team":{0 if mine else 1},"mine":{str(mine).lower()},'
        f'"hostile":{str(not mine).lower()},'
        f'"hp":100.0,"max_hp":100.0,"complete":true,"queued":0}}'
    )


def _sample_lines(frame: int, *entities: tuple[int, str, bool, float]) -> list[str]:
    lines = [
        f'{{"kind":"frame","frame":{frame},"clock_ms":{frame * 3},'
        f'"visible":{len(entities)},"pools":0,"options":0,"credits":4000}}'
    ]
    for index, (unit_id, type_name, mine, x) in enumerate(entities):
        lines.append(_entity_line(frame, index, unit_id, type_name, mine=mine, x=x))
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


_TANK = (1, "c_tank", True, 0.0)
_ENEMY = (9, "c_tank", False, 100.0)


def test_the_army_is_sent_at_the_enemy() -> None:
    peer = _ScriptedPeer(_sample_lines(1, _TANK, _ENEMY) * 3)
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=3)
    assert peer.sent == ['{"kind":"attack","unit_id":1,"target_id":9}']
    assert battle["orders_sent"] == 1


def test_an_order_already_in_flight_is_not_re_issued() -> None:
    """The engine runs a waypoint until it is replaced.

    Re-sending the same attack every sample would replace an in-progress order
    with an identical one at the sampling rate, and the unit would never close
    the distance.
    """
    peer = _ScriptedPeer(_sample_lines(1, _TANK, _ENEMY) * 20)
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=20)
    assert battle["orders_sent"] == 1
    assert battle["samples_seen"] == 20


def test_a_new_target_earns_a_new_order() -> None:
    """The first enemy dies, so the army is re-committed rather than idling."""
    peer = _ScriptedPeer(
        _sample_lines(1, _TANK, _ENEMY) + _sample_lines(2, _TANK, (10, "c_tank", False, 50.0))
    )
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=2)
    assert peer.sent == [
        '{"kind":"attack","unit_id":1,"target_id":9}',
        '{"kind":"attack","unit_id":1,"target_id":10}',
    ]
    assert battle["orders_sent"] == 2


def test_no_enemy_left_is_cleared() -> None:
    peer = _ScriptedPeer(_sample_lines(1, _TANK))
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=5)
    assert battle["outcome"] == "cleared"
    assert peer.sent == []


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
    peer = _ScriptedPeer(_sample_lines(1, _TANK, _ENEMY) * 10)
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=4)
    assert battle["outcome"] == "sample_limit"
    assert battle["samples_seen"] == 4


def test_an_engaged_target_that_disappears_is_counted() -> None:
    """Counted as gone rather than as killed.

    A target that retreated into fog reads identically from here, and the
    scorecard must not claim a kill it cannot see.
    """
    peer = _ScriptedPeer(_sample_lines(1, _TANK, _ENEMY) + _sample_lines(2, _TANK))
    battle = fight(AgentChannel(peer), _CATALOGUE, max_samples=5)
    assert battle["killed"] == 1
    assert battle["outcome"] == "cleared"


def test_the_report_renders_every_figure() -> None:
    peer = _ScriptedPeer(_sample_lines(1, _TANK, _ENEMY) * 2)
    lines = format_battle(fight(AgentChannel(peer), _CATALOGUE, max_samples=2))
    assert lines == (
        "fight outcome  sample_limit",
        "attack orders  1",
        "army           1 -> 1",
        "enemies seen   1 -> 1",
        "engaged gone   0",
        "samples seen   2",
    )
