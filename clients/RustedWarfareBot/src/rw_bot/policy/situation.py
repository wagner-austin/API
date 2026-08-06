"""The scoreboard, read as a situation instead of filed as a report.

The engine broadcasts every player's army value, building value and income
in every sample, unfogged -- the bot has carried the figures on the wire
all campaign and used them for exactly one thing: the ``best rival`` line
of the match report. Meanwhile every release decision it ever made counted
OUR units against a fixed ladder rung, and every all-in fired on a clock
([[policy-verdict]], [[policy-combat]]).

This module is the information layer those decisions were missing: pure
reads of the sample's player records, shaped as answers to the questions a
release decision actually asks. The first consumer is the strike window --
release the horde when the army-value ratio says their force is broken --
which replaces the clock the all-in fired on and the intrusion edge the
riposte guessed with, both of them approximations of the number that was
on the wire the whole time.

The reads are pure; the one stateful thing here is :class:`Momentum`, which
remembers the rival's recent peak the way every other controller remembers
exactly what its decision needs and nothing else ([[policy-loop]]).
"""

from __future__ import annotations

from collections import deque
from typing import Final, TypedDict

from rw_bot.wire.state import Sample

#: Observations of rival history the momentum window holds.
#:
#: Sized to a wave's lifecycle: their group stages, crosses and dies on our
#: line inside roughly this many samples, so a peak older than the window
#: is a different wave's peak and must not keep the release open forever.
MOMENTUM_WINDOW: Final = 40


class Situation(TypedDict):
    """The strategic picture one sample carries.

    Attributes:
        our_army: The local player's army value, the engine's own figure.
        rival_army: The strongest surviving hostile's army value.
        our_income: The local player's income per second.
        rival_income: The strongest surviving hostile's income per second.
    """

    our_army: int
    rival_army: int
    our_income: int
    rival_income: int


def read_situation(sample: Sample) -> Situation | None:
    """Read the scoreboard into a situation.

    Args:
        sample: One observation of the world.

    Returns:
        The situation, or None when the sample carries no player records --
        a scripted world, or a capture taken before the scoreboard existed.
    """
    ours: tuple[int, int] | None = None
    rival: tuple[int, int] | None = None
    for player in sample["players"]:
        if player["local"]:
            ours = (player["army_value"], player["income"])
        elif (
            player["hostile"]
            and not player["defeated"]
            and (rival is None or player["army_value"] > rival[0])
        ):
            rival = (player["army_value"], player["income"])
    if ours is None or rival is None:
        return None
    return Situation(
        our_army=ours[0],
        rival_army=rival[0],
        our_income=ours[1],
        rival_income=rival[1],
    )


class Momentum:
    """Watches the strongest rival's army value for the drop that is a door.

    The first strike window was an absolute ratio and was refuted in one
    Impossible probe: their army is always a multiple of ours at that rung,
    so the window never opened and the horde died holding (log 2026-07-31).
    What actually marks their broken wave is RELATIVE -- the army value
    falling tens of thousands as the wave dies on our line -- and reading a
    fall needs a memory of the recent peak, which is all this class holds.

    Attributes are internal: a bounded window of the rival's recent army
    values, oldest first.
    """

    def __init__(self) -> None:
        self._history: deque[int] = deque(maxlen=MOMENTUM_WINDOW)

    def observe(self, sample: Sample) -> None:
        """Record this observation's rival army value.

        A sample with no scoreboard records nothing: absence of data is not
        a zero, and a zero here would fake a peak-sized drop.

        Args:
            sample: One observation of the world.
        """
        situation = read_situation(sample)
        if situation is not None:
            self._history.append(situation["rival_army"])

    def drop(self) -> int:
        """Return how far the rival's army value sits below its recent peak.

        Returns:
            The recent peak minus the latest reading, never below zero, and
            zero before any reading exists.
        """
        if not self._history:
            return 0
        return max(0, max(self._history) - self._history[-1])


def closing_window(sample: Sample, close: int) -> bool:
    """Report whether the match is decided enough to go end it.

    Open while our army value stands at least ``close`` times the strongest
    surviving rival's. Zero is off.

    The measurement this exists for: nineteen Very Hard matches stood
    dominant at the 4,000-sample cap -- median worth 30,100, rivals ground
    low -- and replayed at 10,000 samples, eleven of them LOST. The AI
    compounds too, and mid-game dominance decays into a losing long game
    unless it is converted while the window is open (`runs/sweeps/vh-close`,
    log 2026-08-01). The refuted absolute ratio was the same test pointed
    the other way: at Impossible their army is always the multiple, so a
    window on OUR multiple never opened there and never will -- this is a
    rung-where-we-compound verb by construction.

    A rival with no army at all opens the window at any ratio, deliberately:
    buildings do not chase, and a match against a disarmed opponent is
    exactly the one to go finish.

    Args:
        sample: One observation of the world.
        close: Our army as a multiple of theirs that means decided, zero off.

    Returns:
        True when the doctrine's dominance is met.
    """
    if close <= 0:
        return False
    situation = read_situation(sample)
    if situation is None:
        return False
    return situation["our_army"] >= close * situation["rival_army"]


#: Consecutive open-window samples before the closer commits.
#:
#: Persistence as evidence, instead of a magic credit floor. The raw latch
#: committed on the first open sample and went 9 won / 13 LOST: at match
#: start three tanks against a builder read as dominance for a few samples,
#: and a permanent latch turned that transient into a lifelong all-in --
#: four matches won or survived by earlier arms were wiped
#: (`runs/sweeps/vh-latch`, log 2026-08-01). A spike lasts a handful of
#: samples; genuine mid-game dominance persists for hundreds, so a hold of
#: twenty-five filters the first without meaningfully delaying the second.
CLOSE_HOLD: Final = 25


class Closer:
    """Latches the decision to end the match, on sustained dominance only.

    The window itself is :func:`closing_window`; what this adds is memory in
    both directions. Forward: once committed, always committed -- re-reading
    the window every tick closed piecemeal, 9, 3 and 6 marches dying in
    dribbles across three lost matches (`runs/sweeps/vh-closer`). Backward:
    commitment needs the window held open :data:`CLOSE_HOLD` samples running,
    because committing on one open sample turned early-game ratio noise into
    a lifelong premature all-in (`runs/sweeps/vh-latch`).
    """

    def __init__(self, close: int) -> None:
        """Open the closer.

        Args:
            close: The doctrine's dominance multiple, zero for never.
        """
        self._close = close
        self._held = 0
        self._committed = False

    def observe(self, sample: Sample) -> bool:
        """Advance the debounce and report whether the match is being ended.

        Args:
            sample: One observation of the world.

        Returns:
            True from the sample the commitment latches, forever after.
        """
        if self._committed:
            return True
        if closing_window(sample, self._close):
            self._held += 1
            self._committed = self._held >= CLOSE_HOLD
        else:
            self._held = 0
        return self._committed


def strike_window(momentum: Momentum, strike: int) -> bool:
    """Report whether the rival's fall opens the release window.

    Open while the strongest rival's army value sits at least ``strike``
    credits below its recent peak -- their wave died somewhere, most often
    on our line, and the credits it cost them are the width of the door.
    Zero is off.

    Args:
        momentum: The rival's tracked history.
        strike: Credits of rival army value that must have fallen, zero off.

    Returns:
        True when the doctrine's drop is met.
    """
    return strike > 0 and momentum.drop() >= strike


__all__ = [
    "CLOSE_HOLD",
    "MOMENTUM_WINDOW",
    "Closer",
    "Momentum",
    "Situation",
    "closing_window",
    "read_situation",
    "strike_window",
]
