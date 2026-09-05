"""Feeding the corpus-trained heads from the loop's own figures.

The loop computes one set of figures per observation and three readers
consume them: the recorder writes the trace, the doom latch photographs
the opening, the brace latch photographs the sliding present. Each
head's feed order is that head's training contract -- the exact columns
its exporter fitted on -- so building the tuples is a translation
boundary, not loop logic, and it lives here so the loop reads as
strategy rather than plumbing ([[policy-loop]]).

One owner for both latches keeps the loop's interface to prediction
small: feed the sentries every sample, read two booleans. The brace
arming EDGE is returned rather than acted on, because what the arm
does to the match (stand expansion down, zero the reserve) is the
loop's policy to apply, and applying it here would put a spender's
decision inside a reader ([[impossible-step-three-design]]).
"""

from __future__ import annotations

from collections.abc import Mapping

from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.doom import DoomLatch
from rw_bot.policy.field import coverage
from rw_bot.policy.head import HeadModel
from rw_bot.policy.raze import BraceLatch, RazeWatch
from rw_bot.policy.reclaim import EXTRACTOR_TYPES
from rw_bot.policy.scoreboard import local_player, rival_income
from rw_bot.policy.situation import read_situation
from rw_bot.wire.state import Sample


class Sentries:
    """Both prediction latches, fed together, read as two booleans.

    Attributes are internal; the readable surface is :attr:`predicted`
    (the doom latch's verdict) and :attr:`braced` (the brace latch's).
    """

    def __init__(
        self,
        doom: HeadModel | None,
        brace: HeadModel | None,
        profiles: Mapping[str, CombatProfile],
        gate: HeadModel | None = None,
    ) -> None:
        """Open the sentries.

        Args:
            doom: The fleet-doom model, or None when the doctrine does not
                play the predicted mode. The caller applies that gate --
                which modes exist is the doctrine's business.
            brace: The razing model, or None when the doctrine plays
                unbraced.
            profiles: Combat profiles by type name, for the brace feed's
                own coverage pass -- the recorder computes one only when
                tracing, and the latch must see the same figures either
                way.
            gate: The razing model AGAIN, or None when nothing gates on
                it. The brace latch arms once and stops reading; the hunt
                gate is a continuous verdict -- stand down while the
                razing is predicted, hunt again when the score clears --
                so it keeps its own window over the same feed
                ([[impossible-step-three-design]]).
        """
        self._doom = DoomLatch(doom) if doom is not None else None
        self._brace = BraceLatch(brace) if brace is not None else None
        self._gate = gate
        self._gate_watch = RazeWatch(gate["window"]) if gate is not None else None
        self._profiles = profiles

    @property
    def predicted(self) -> bool:
        """Whether the doom latch has armed."""
        return self._doom is not None and self._doom.armed

    @property
    def braced(self) -> bool:
        """Whether the brace latch has armed."""
        return self._brace is not None and self._brace.armed

    @property
    def hunted_down(self) -> bool:
        """Whether the hunt gate holds the party home right now.

        False without a gate, while the window is still filling, and
        whenever the score sits below the model's own threshold --
        exactly the samples on which hunting is allowed.
        """
        if self._gate is None or self._gate_watch is None or not self._gate_watch.full():
            return False
        return self._gate_watch.score(self._gate) >= self._gate["threshold"]

    def observe(
        self,
        sample: Sample,
        *,
        army: int,
        enemies: int,
        extractors: int,
        losses: int,
        producers: int,
        idle: int,
        orders: int,
        refused: int,
        worth: int,
        rival_worth: int,
        workers: int,
        navy_seen: int,
        air_seen: int,
        navy_blood: int,
    ) -> bool:
        """Feed both latches this observation's figures.

        Args:
            sample: One observation of the world.
            army: Units able to fight.
            enemies: Hostile entities visible.
            extractors: Finished extractors owned.
            losses: Units that went missing since the previous sample.
            producers: Owned units able to make something wanted.
            idle: How many of those held nothing in their queue.
            orders: Produce orders issued this observation.
            refused: Credit claims the budget turned down this observation.
            worth: Everything the player holds.
            rival_worth: The strongest hostile player's total.
            workers: Builders owned, as the workforce counts them.
            navy_seen: Hostile WATER-movers visible this observation.
            air_seen: Hostile fliers visible this observation.
            navy_blood: Cumulative kills on us by fleet types seen so far.

        Returns:
            True exactly once, on the observation the brace armed -- the
            edge the loop's policy responds to. False every other time.
        """
        pilot = local_player(sample)
        income = 0 if pilot is None else pilot["income"]
        if self._doom is not None:
            # The trace's numeric columns, in doom.COLUMNS order: what the
            # model was fitted on is what the latch is fed.
            self._doom.feed(
                (
                    army,
                    sample["credits"],
                    enemies,
                    extractors,
                    losses,
                    producers,
                    idle,
                    orders,
                    refused,
                    worth,
                    rival_worth,
                    income,
                    rival_income(sample),
                    workers,
                    navy_seen,
                    air_seen,
                    navy_blood,
                )
            )
        brace_open = self._brace is not None and not self._brace.armed
        if not brace_open and self._gate_watch is None:
            return False
        situation = read_situation(sample)
        covered = coverage(sample, self._profiles, EXTRACTOR_TYPES)
        row = (
            army,
            sample["credits"],
            extractors,
            losses,
            workers,
            income,
            worth,
            rival_worth,
            rival_income(sample),
            0 if situation is None else situation["rival_army"],
            covered["eco_covered"],
            covered["own_covered"],
            covered["foe_covered"],
        )
        if self._gate_watch is not None:
            self._gate_watch.feed(row)
        if self._brace is None or self._brace.armed:
            return False
        self._brace.feed(row)
        return self._brace.armed


__all__ = ["Sentries"]
