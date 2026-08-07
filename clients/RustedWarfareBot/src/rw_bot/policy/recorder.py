"""Collecting the per-sample record a run is read back from.

The loop sees everything and the scorecard keeps about two dozen numbers. What
is dropped between those two is every question of the form "when did it turn",
and that is the question a pair of runs always ends up posing
([[policy-trace]]).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rw_bot.policy.scoreboard import local_player, rival_income
from rw_bot.policy.trace import (
    Loss,
    Tick,
    format_trace,
    losses_between,
    owned_by_id,
    world_digest,
)
from rw_bot.wire.state import Entity, Sample


class Recorder:
    """Keeps one row per sample so a run can be read back afterwards.

    Off unless a path is given, because most runs are tests and a trace file
    per test would be noise. Accumulates in memory and writes once at the end:
    the loop is holding the simulation between samples in lockstep, and a
    flush per observation would pace the match by disk.

    Attributes:
        ticks: One entry per sample, in order.
        losses: Every inferred loss, in order.
    """

    def __init__(self, path: Path | None) -> None:
        """Open a recorder.

        Args:
            path: Where to write the trace, or None to keep none.
        """
        self._path = path
        self.ticks: list[Tick] = []
        self.losses: list[Loss] = []
        self._previous: Mapping[int, Entity] = {}

    def step(
        self,
        sample: Sample,
        army: int,
        enemies: int,
        extractors: int,
        producers: int,
        idle: int,
        orders: int,
        refused: int,
        worth: int,
        rival: int,
        plan: str,
        workers: int,
    ) -> None:
        """Record one observation.

        Args:
            sample: One observation of the world.
            army: Units able to fight.
            enemies: Hostile entities visible.
            extractors: Finished extractors owned.
            producers: Owned units able to make something wanted.
            idle: How many of those held nothing in their queue.
            orders: Produce orders issued this observation.
            refused: Credit claims the budget turned down this observation.
            worth: Everything the player holds.
            rival: The strongest hostile player's total.
            plan: The opening plan's outcome this observation.
            workers: Builders owned, as the workforce counts them.
        """
        if self._path is None:
            return
        current = owned_by_id(sample)
        gone = losses_between(self._previous, current, sample["frame"])
        self._previous = current
        self.losses.extend(gone)
        # The income pair is read off the sample here rather than passed in,
        # like the losses and the digest: no other pass wants these numbers,
        # and the scoreboard rows ride every sample already. ``local`` is None
        # only on a stream that predates the player record, where zero is the
        # honest column ([[policy-economy]]).
        local = local_player(sample)
        self.ticks.append(
            Tick(
                frame=sample["frame"],
                army=army,
                credits=sample["credits"],
                enemies=enemies,
                extractors=extractors,
                lost=len(gone),
                producers=producers,
                idle=idle,
                orders=orders,
                refused=refused,
                worth=worth,
                rival=rival,
                income=0 if local is None else local["income"],
                rival_income=rival_income(sample),
                world=world_digest(sample),
                plan=plan,
                workers=workers,
            )
        )

    def write(self) -> None:
        """Write both tables, if a path was given.

        Frames are rebased to **match age** -- the first sample is age zero --
        because the absolute counter runs from engine boot and carries the
        wall-clock length of the menu in it. Two runs of one seed now produce
        the same match one or two boot frames apart, and rebasing is what
        makes their traces byte-comparable: age is the coordinate the match
        actually evolves in ([[policy-determinism]]).
        """
        if self._path is None:
            return
        base = self.ticks[0]["frame"] if self.ticks else 0
        ticks = [Tick(**{**t, "frame": t["frame"] - base}) for t in self.ticks]
        losses = [Loss(**{**one, "frame": one["frame"] - base}) for one in self.losses]
        lines = format_trace(ticks, losses)
        self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["Recorder"]
