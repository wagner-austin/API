"""The scoreboard read as a situation, and the fall that opens the window.

The engine broadcasts every player's army value unfogged; what is pinned
here is that the reader picks the right rows (ours, the strongest SURVIVING
rival), that momentum measures the fall from the recent peak exactly, and
that a peak older than the window is forgotten ([[policy-situation]]).
"""

from __future__ import annotations

from rw_bot.policy.situation import (
    CLOSE_HOLD,
    MOMENTUM_WINDOW,
    Closer,
    Momentum,
    closing_window,
    read_situation,
    strike_window,
)
from rw_bot.wire.state import Sample
from tests.wire_fixtures import entity, player, sample


def _world(*, ours: int, rivals: tuple[tuple[int, bool], ...]) -> Sample:
    """A sample whose scoreboard carries our army value and the rivals'."""
    players = [player(0, index=0, local=True, hostile=False, army_value=ours, income=50)]
    for index, (value, defeated) in enumerate(rivals):
        players.append(
            player(
                index + 1,
                index=index + 1,
                hostile=True,
                defeated=defeated,
                army_value=value,
                income=90,
            )
        )
    return sample(entity(213, "commandCenter"), credits=4000, players=tuple(players))


def test_the_situation_reads_ours_and_the_strongest_surviving_rival() -> None:
    world = _world(ours=12_000, rivals=((8_000, False), (30_000, True), (15_000, False)))
    situation = read_situation(world)
    assert situation == {
        "our_army": 12_000,
        "rival_army": 15_000,
        "our_income": 50,
        "rival_income": 90,
    }
    # The 30,000 rival is defeated and does not count; the strongest
    # SURVIVING one is the 15,000.


def test_a_world_with_no_scoreboard_reads_as_no_situation() -> None:
    """Absence of data is not a zero: a zero would fake a peak-sized drop."""
    bare = sample(entity(213, "commandCenter"), credits=4000)
    assert read_situation(bare) is None
    momentum = Momentum()
    momentum.observe(_world(ours=1_000, rivals=((30_000, False),)))
    momentum.observe(bare)
    assert momentum.drop() == 0


def test_the_window_opens_on_the_fall_from_the_recent_peak() -> None:
    """Their wave dies on our line: 30,000 peak, 12,000 left, drop 18,000."""
    momentum = Momentum()
    momentum.observe(_world(ours=9_000, rivals=((22_000, False),)))
    momentum.observe(_world(ours=9_000, rivals=((30_000, False),)))
    assert momentum.drop() == 0
    assert strike_window(momentum, 15_000) is False
    momentum.observe(_world(ours=9_000, rivals=((12_000, False),)))
    assert momentum.drop() == 18_000
    assert strike_window(momentum, 15_000) is True
    assert strike_window(momentum, 18_001) is False
    # Zero is off, whatever the fall says.
    assert strike_window(momentum, 0) is False


def test_a_peak_older_than_the_window_is_forgotten() -> None:
    """A different wave's peak must not hold the door open forever."""
    momentum = Momentum()
    momentum.observe(_world(ours=9_000, rivals=((50_000, False),)))
    for _ in range(MOMENTUM_WINDOW):
        momentum.observe(_world(ours=9_000, rivals=((20_000, False),)))
    assert momentum.drop() == 0


def test_momentum_before_any_reading_is_flat() -> None:
    assert Momentum().drop() == 0


def test_the_closing_window_opens_on_the_doctrine_dominance() -> None:
    """The closer's trigger: nineteen Very Hard matches stood dominant at
    the 4,000-sample cap and eleven of them LOST at 10,000 -- a decided
    match is won by ending it while it is decided (log 2026-08-01)."""
    assert closing_window(_world(ours=9_000, rivals=((3_000, False),)), 3) is True
    assert closing_window(_world(ours=8_999, rivals=((3_000, False),)), 3) is False
    assert closing_window(_world(ours=9_000, rivals=((3_000, False),)), 0) is False


def test_a_disarmed_rival_opens_the_window_at_any_ratio() -> None:
    """Buildings do not chase; a rival with no army is the one to finish."""
    assert closing_window(_world(ours=500, rivals=((0, False),)), 3) is True


def test_the_strongest_survivor_is_the_dominance_yardstick() -> None:
    """A dead rival's ghost value must not keep the door shut."""
    world = _world(ours=9_000, rivals=((50_000, True), (3_000, False)))
    assert closing_window(world, 3) is True


def test_a_scoreboard_free_world_never_closes() -> None:
    """Scripted worlds and old captures carry no player records."""
    assert closing_window(sample(entity(213, "commandCenter")), 3) is False


def test_the_closer_commits_only_on_sustained_dominance() -> None:
    """Persistence as evidence: the raw latch committed on one open sample
    and turned early-game ratio noise into lifelong premature all-ins --
    9 won / 13 lost, four former wins wiped (`runs/sweeps/vh-latch`,
    log 2026-08-01)."""
    closer = Closer(3)
    dominant = _world(ours=9_000, rivals=((3_000, False),))
    for _ in range(CLOSE_HOLD - 1):
        assert closer.observe(dominant) is False
    assert closer.observe(dominant) is True


def test_a_transient_spike_resets_the_debounce() -> None:
    """Three tanks against a builder reads as dominance for a few samples;
    an interruption starts the count over."""
    closer = Closer(3)
    dominant = _world(ours=9_000, rivals=((3_000, False),))
    contested = _world(ours=5_000, rivals=((3_000, False),))
    for _ in range(CLOSE_HOLD - 1):
        closer.observe(dominant)
    assert closer.observe(contested) is False
    for _ in range(CLOSE_HOLD - 1):
        assert closer.observe(dominant) is False
    assert closer.observe(dominant) is True


def test_a_committed_closer_never_stands_down() -> None:
    """Forward memory: re-reading the window every tick closed piecemeal --
    9, 3 and 6 marches dying in dribbles (`runs/sweeps/vh-closer`)."""
    closer = Closer(3)
    dominant = _world(ours=9_000, rivals=((3_000, False),))
    for _ in range(CLOSE_HOLD):
        closer.observe(dominant)
    assert closer.observe(_world(ours=500, rivals=((90_000, False),))) is True


def test_a_zero_close_never_commits() -> None:
    closer = Closer(0)
    dominant = _world(ours=9_000, rivals=((3_000, False),))
    for _ in range(CLOSE_HOLD * 2):
        assert closer.observe(dominant) is False
