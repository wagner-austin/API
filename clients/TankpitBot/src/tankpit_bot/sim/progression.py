"""Law 10 — the client's rank, and the 0x2B frames that announce it.

The sim's client held one rank for a whole session, so the archived
0x2B Promotion family had no counterpart at all
([[session-state-deglobalisation]]).

The cycle, measured across all 285 archived sessions (7 frames — rare,
but every one fits the same shape):

* ``new_rank=0, was_promoted=False`` lands ZERO seconds after the
  CLIENT'S OWN deactivation, three times out of three. Dying demotes
  you to recruit, silently — the flag says no banner.
* ``new_rank=1, was_promoted=True`` follows, with the banner, once the
  tank has recovered. Three of the four archived promotions have no
  deactivation inside their capture at all (the death predates the
  recording); the one that does sits 196 s — 98 ticks — after it.
* The ``promo_state`` bar in every TankStatusSync tells the same
  story from the other side: 62,209 of 62,528 own-tank frames carry
  10, and the only frames carrying 0-9 are the recovery windows
  between a demotion and its promotion.

What is NOT claimed: the recovery delay's distribution. One timed
sample is a number, not a law, and it is labelled as such below.

``promo_state`` is not a fuel gauge, incidentally — value 10 spans the
whole fuel range. It is a progression counter, and the sim emits the
steady-state 10 rather than the 0 it used to invent.
"""

from __future__ import annotations

from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.protocol.types import BinaryMessage, PromotionDict
from tankpit_bot.sim.world import SimWorldDict

DEMOTED_RANK = 0
"""Recruit — the rank a deactivation drops the client to."""

PROMOTED_RANK = 1
"""Private — the rank recovery restores, and the sim's starting rank."""

PROMOTION_RECOVERY_TICKS = 98
"""Ticks between the demotion and the promotion that undoes it.

ONE measured sample (bot-20260803-180918: 196 s at the 2 s tick). The
other three archived promotions carry no deactivation inside their
capture, so the delay has a value here and not a distribution."""

STEADY_PROMO_STATE = 10
"""The promotion bar's resting value: 62,209 of 62,528 own-tank frames."""

RECOVERING_PROMO_STATE = 0
"""The bar during a recovery window — the archive's 0-9 band starts here."""


class RankProgression:
    """The client's rank across deactivation and recovery.

    Holds no world: :meth:`advance` is handed the one it changes, the
    same way the ferry drift and the room churn are.
    """

    def __init__(self, client_id: int) -> None:
        """Bind the progression to the connected client's tank.

        Args:
            client_id: The tank whose rank this tracks.
        """
        self._client_id = client_id
        self.demoted_tick: int | None = None
        """The tick the client was demoted on, while recovering."""

    @property
    def promo_state(self) -> int:
        """The bar value this tick's status syncs should carry."""
        return RECOVERING_PROMO_STATE if self.demoted_tick is not None else STEADY_PROMO_STATE

    def note_deactivation(self, world: SimWorldDict, messages: list[BinaryMessage]) -> None:
        """Demote the client on its own deactivation and say so.

        Args:
            world: Simulated world (the client's rank is lowered).
            messages: The batch carrying the 0x41 (appended).
        """
        client = world["tanks"][self._client_id]
        if client["rank"] == DEMOTED_RANK:
            return
        client["rank"] = DEMOTED_RANK
        client["fuel"] = min(client["fuel"], fuel_capacity(DEMOTED_RANK))
        self.demoted_tick = world["tick"]
        messages.append(PromotionDict(msg_type=0x2B, new_rank=DEMOTED_RANK, was_promoted=False))

    def advance(self, world: SimWorldDict, messages: list[BinaryMessage]) -> None:
        """Promote the client back once its recovery window elapses.

        Args:
            world: Simulated world (the client's rank is restored).
            messages: This tick's outgoing batch (appended).
        """
        if self.demoted_tick is None:
            return
        if world["tick"] - self.demoted_tick < PROMOTION_RECOVERY_TICKS:
            return
        self.demoted_tick = None
        world["tanks"][self._client_id]["rank"] = PROMOTED_RANK
        messages.append(PromotionDict(msg_type=0x2B, new_rank=PROMOTED_RANK, was_promoted=True))


__all__ = [
    "DEMOTED_RANK",
    "PROMOTED_RANK",
    "PROMOTION_RECOVERY_TICKS",
    "RECOVERING_PROMO_STATE",
    "STEADY_PROMO_STATE",
    "RankProgression",
]
