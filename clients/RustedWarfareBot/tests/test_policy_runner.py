"""The policy loop, driven against a scripted world.

The deciding is tested in ``test_policy_build_order``; what is tested here is
the loop around it — when it stops, what it counts, and the one behaviour a
pure decision function cannot express on its own: not re-ordering a structure
that is still being built.
"""

from __future__ import annotations

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.runner import format_scorecard, run


def _unit(type_name: str, price: int) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=price,
        hp=100,
        speed=0.0,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


_CATALOGUE = {"landFactory": _unit("landFactory", 300), "laboratory": _unit("laboratory", 900)}


def _entity_line(frame: int, index: int, unit_id: int, type_name: str, mine: bool) -> str:
    return (
        f'{{"kind":"entity","frame":{frame},"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":100.0,"y":200.0,'
        f'"team":{0 if mine else 1},"mine":{str(mine).lower()},'
        f'"hp":100.0,"max_hp":100.0}}'
    )


def _sample_lines(frame: int, credits: int, *entities: tuple[int, str, bool]) -> list[str]:
    lines = [
        f'{{"kind":"frame","frame":{frame},"clock_ms":{frame * 3},'
        f'"visible":{len(entities)},"credits":{credits}}}'
    ]
    for index, (unit_id, type_name, mine) in enumerate(entities):
        lines.append(_entity_line(frame, index, unit_id, type_name, mine))
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


def test_a_plan_already_satisfied_finishes_without_ordering() -> None:
    peer = _ScriptedPeer(_sample_lines(1, 4000, _BUILDER, (300, "landFactory", True)))
    card = run(AgentChannel(peer), ("landFactory",), _CATALOGUE, max_samples=5)
    assert card["outcome"] == "done"
    assert card["completed"] == 1
    assert peer.sent == []


def test_one_order_is_sent_and_the_structure_ends_the_plan() -> None:
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(10, 3700, _BUILDER, (300, "landFactory", True))
    )
    card = run(AgentChannel(peer), ("landFactory",), _CATALOGUE, max_samples=5)
    assert card["outcome"] == "done"
    assert card["completed"] == 1
    assert card["orders_sent"] == 1
    assert peer.sent == ['{"kind":"build","unit_id":214,"x":300.0,"y":320.0,"type":"landFactory"}']


def test_a_structure_still_being_built_is_not_re_ordered() -> None:
    """Three samples pass before it appears; only one order may be sent."""
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(2, 3700, _BUILDER)
        + _sample_lines(3, 3700, _BUILDER)
        + _sample_lines(4, 3700, _BUILDER, (300, "landFactory", True))
    )
    card = run(AgentChannel(peer), ("landFactory",), _CATALOGUE, max_samples=8)
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
    card = run(AgentChannel(peer), ("laboratory",), _CATALOGUE, max_samples=8)
    assert card["orders_sent"] == 1
    assert card["outcome"] == "done"


def test_a_blocked_plan_stops_immediately() -> None:
    peer = _ScriptedPeer(_sample_lines(1, 4000, (213, "commandCenter", True)))
    card = run(AgentChannel(peer), ("landFactory",), _CATALOGUE, max_samples=9)
    assert card["outcome"] == "blocked"
    assert card["last_reason"] == "the player owns no builder"
    assert card["samples_seen"] == 1


def test_the_sample_budget_bounds_a_run_that_never_finishes() -> None:
    peer = _ScriptedPeer(
        _sample_lines(1, 10, _BUILDER)
        + _sample_lines(2, 10, _BUILDER)
        + _sample_lines(3, 10, _BUILDER)
    )
    card = run(AgentChannel(peer), ("laboratory",), _CATALOGUE, max_samples=3)
    assert card["outcome"] == "sample_limit"
    assert card["samples_seen"] == 3
    assert card["orders_sent"] == 0


def test_frames_elapsed_spans_the_run() -> None:
    peer = _ScriptedPeer(
        _sample_lines(100, 4000, _BUILDER)
        + _sample_lines(460, 3700, _BUILDER, (300, "landFactory", True))
    )
    card = run(AgentChannel(peer), ("landFactory",), _CATALOGUE, max_samples=5)
    assert card["frames_elapsed"] == 360
    assert card["credits_at_end"] == 3700


def test_a_zero_sample_budget_reports_that_nothing_was_read() -> None:
    card = run(AgentChannel(_ScriptedPeer([])), ("landFactory",), _CATALOGUE, max_samples=0)
    assert card["outcome"] == "sample_limit"
    assert card["samples_seen"] == 0
    assert card["last_reason"] == "no sample was read"


def test_two_structures_are_ordered_in_plan_sequence() -> None:
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(2, 3700, _BUILDER, (300, "landFactory", True))
        + _sample_lines(3, 2800, _BUILDER, (300, "landFactory", True), (301, "laboratory", True))
    )
    card = run(AgentChannel(peer), ("landFactory", "laboratory"), _CATALOGUE, max_samples=6)
    assert card["completed"] == 2
    assert card["orders_sent"] == 2
    assert [line.split('"type":"')[1].rstrip('"}') for line in peer.sent] == [
        "landFactory",
        "laboratory",
    ]


def test_the_scorecard_renders_every_figure() -> None:
    peer = _ScriptedPeer(
        _sample_lines(1, 4000, _BUILDER)
        + _sample_lines(10, 3700, _BUILDER, (300, "landFactory", True))
    )
    card = run(AgentChannel(peer), ("landFactory",), _CATALOGUE, max_samples=5)
    assert format_scorecard(card) == (
        "outcome        done (all 1 structures built)",
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
        max_samples=20,
        stall_samples=4,
    )
    assert card["outcome"] == "stalled"
    assert card["orders_sent"] == 1
    assert card["completed"] == 0
    assert "laboratory was ordered but never appeared after 4 samples" in card["last_reason"]


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
        max_samples=20,
        stall_samples=5,
    )
    assert card["outcome"] == "done"
