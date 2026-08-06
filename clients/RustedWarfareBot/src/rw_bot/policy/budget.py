"""One authority over the tick's credits, claimed in priority order.

The bot had two spenders and one balance. Inside a single observation the
production pass budgeted across every idle producer using ``sample["credits"]``,
and the expansion pass then asked the same field whether it could afford an
extractor — neither knowing what the other had just committed. Both were correct
in isolation and the pair was not: with one factory the overlap was small enough
to hide, and the moment a second producer existed the same credit was spent
twice and the engine silently refused the second order ([[policy-economy]]).

The fix is not a bigger reserve. It is that **spending is a single decision with
an order**, and this module is that decision. A tick opens one budget, every
spender claims against it in priority order, and a claim that does not fit is
refused with a reason rather than issued and dropped by the engine.

Priority is expressed by call order rather than by a number attached to each
claim, because the order *is* the policy and burying it in per-claim weights
would make it unreadable. What the loop does, in order:

1. the opening plan, because nothing else can proceed without its prerequisites,
2. replacing losses, because an army that is dying now cannot wait for income,
3. more income, which compounds over the rest of the match,
4. more throughput, which does not, and which is what the surplus buys once
   there is no income left to buy.

**The reserve is what keeps step 4 from starving step 2.** Production only
spends when a producer is idle, so on a tick where every factory is busy the
lower-priority claims would otherwise take everything and leave nothing for the
replacement queued a moment later. Holding a floor that only the army may cross
is a forward reservation for a claim that has not been made yet, which is a
different job from ordering the claims that have.

Pure: credits in, decisions out. Nothing here opens a socket or issues an order.
"""

from __future__ import annotations

from typing import TypedDict

from rw_bot import RwBotError

_NEGATIVE_CREDITS = "RW-BUDGET-001"
_NEGATIVE_RESERVE = "RW-BUDGET-002"
_NEGATIVE_CLAIM = "RW-BUDGET-003"


class BudgetError(RwBotError):
    """A budget was opened or claimed against with an impossible figure.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending figure.
    """


class Claim(TypedDict):
    """One spender's request against the tick's credits.

    Recorded whether or not it succeeded, because a refusal is the more
    informative half. "No pool was taken" has several causes that call for
    opposite responses, and a bare count of expansions cannot tell them apart
    ([[policy-economy]]).

    Attributes:
        purpose: What the credits were for, e.g. ``"expand:extractorT1"``.
        amount: Credits requested.
        granted: Whether the claim was met in full. Claims are never met in
            part — half an extractor is not a purchase.
        reason: Human-readable justification, for the run log.
    """

    purpose: str
    amount: int
    granted: bool
    reason: str


class Budget:
    """The credits available this tick, and what has been committed against them.

    One per observation. Carrying it between ticks would be a second source of
    truth about a figure the sample already reports, and the two would drift the
    moment an order was refused by the engine.

    Attributes:
        credits: Credits the player held when the tick opened.
        reserve: Credits only a protected claim may draw on.
    """

    def __init__(self, credits: int, reserve: int) -> None:
        """Open a budget for one observation.

        Args:
            credits: Credits the player holds, from the sample.
            reserve: Credits to keep back for claims that protect the army.
                A reserve larger than the balance is not an error — it means
                every credit is spoken for, which is an ordinary early-match
                state — and unprotected claims are simply all refused.

        Raises:
            BudgetError: ``RW-BUDGET-001`` when credits are negative,
                ``RW-BUDGET-002`` when the reserve is.
        """
        if credits < 0:
            raise BudgetError(
                _NEGATIVE_CREDITS,
                f"a budget cannot open on {credits} credits; the engine floors the "
                "balance at zero, so a negative is a decode fault rather than a debt",
            )
        if reserve < 0:
            raise BudgetError(
                _NEGATIVE_RESERVE,
                f"a reserve of {reserve} is not a reservation; to reserve nothing, pass 0",
            )
        self.credits = credits
        self.reserve = reserve
        self._spent = 0
        self._withheld = 0
        self._ledger: list[Claim] = []

    def spent(self) -> int:
        """Return how much has been committed so far this tick.

        Returns:
            Credits committed by granted claims.
        """
        return self._spent

    def remaining(self) -> int:
        """Return what is left for a protected claim.

        Returns:
            Credits not yet committed, never below zero.
        """
        return self.credits - self._spent

    def spendable(self) -> int:
        """Return what is left for an unprotected claim.

        Returns:
            Credits not yet committed and not held back by the reserve or by
            a withholding, never below zero.
        """
        return max(0, self.remaining() - self.reserve - self._withheld)

    def withhold(self, amount: int) -> None:
        """Keep credits back from every later claim this tick.

        **The saving mechanism, and why claim order alone could not buy the
        tier-three conversion.** A budget lives one tick, and income arrives a
        few credits per tick -- so a 4,000-credit conversion asked first still
        read ``of 0 available`` every single tick, because the previous tick's
        spenders had drained the balance to the reserve before it could grow
        (measured: asked 3,788, granted 0, log 2026-07-31). A spender that is
        refused withholds its price instead, later channels see that much
        less, and the balance climbs across ticks until the claim fits.
        Protected claims are bound too, deliberately: replacing losses is
        protected and drains the balance to zero each tick, so a saving only
        investment respected would never fill.

        Args:
            amount: Credits to keep back. Accumulates across calls.

        Raises:
            BudgetError: ``RW-BUDGET-003`` when the amount is negative.
        """
        if amount < 0:
            raise BudgetError(
                _NEGATIVE_CLAIM,
                f"a withholding of {amount} would free credits rather than save them",
            )
        self._withheld += amount

    def release(self, amount: int) -> None:
        """Hand a withholding back to the claimants that run after this call.

        The withhold-then-release pair is how a LATE claimant saves. The tech
        unlock claims first each tick, so claim-then-withhold suffices there;
        defence claims last, and a withholding it placed early in the tick
        would bind its own claim too -- ``spendable`` subtracts what is
        withheld with no notion of whose saving it is. So the deficit is
        withheld early (binding produce, upgrades and creep), and released
        here at the point the expander runs -- where income and defence, the
        two claimants the measurements ordered, arbitrate it in that order
        ([[policy-economy]], log 2026-08-01).

        Never released below zero: a release larger than what stands withheld
        frees only what was actually held, so a caller cannot mint credits by
        over-releasing.

        Args:
            amount: Credits to hand back. Negative is refused for the same
                reason a negative withholding is.

        Raises:
            BudgetError: ``RW-BUDGET-003`` when the amount is negative.
        """
        if amount < 0:
            raise BudgetError(
                _NEGATIVE_CLAIM,
                f"a release of {amount} would withhold credits rather than free them",
            )
        self._withheld = max(0, self._withheld - amount)

    def ledger(self) -> tuple[Claim, ...]:
        """Return every claim made this tick, granted or not, in order.

        Returns:
            The claims, in the order they were made.
        """
        return tuple(self._ledger)

    def claim(self, purpose: str, amount: int, *, protected: bool = False) -> Claim:
        """Ask for credits, and commit them when they are there.

        Args:
            purpose: What the credits are for, carried into the run log.
            amount: Credits wanted. Zero is a valid claim and is always granted:
                an order that costs nothing still wants a ledger entry saying it
                happened.
            protected: Whether this claim may draw on the reserve. True for
                anything that keeps the army alive, false for investment.

        Returns:
            The claim, granted or refused, with its reasoning either way.

        Raises:
            BudgetError: ``RW-BUDGET-003`` when the amount is negative, which
                would otherwise credit the budget and let a later claim spend
                money the player never had.
        """
        if amount < 0:
            raise BudgetError(
                _NEGATIVE_CLAIM,
                f"{purpose!r} claimed {amount} credits; a negative claim would refund the "
                "budget and let a later claim spend credits the player never held",
            )
        # Withheld credits are invisible to every later claim, protected or
        # not -- a saving that only investment respected would never fill,
        # because replacing losses is protected and drains the balance to
        # zero each tick. The reserve stays crossable: it protects the army
        # FROM investment, where the withholding protects a purchase FROM
        # the army, and the two floors are deliberately independent.
        available = max(0, self.remaining() - self._withheld) if protected else self.spendable()
        if amount > available:
            return self._record(
                Claim(
                    purpose=purpose,
                    amount=amount,
                    granted=False,
                    reason=(
                        f"{purpose} wanted {amount} of {available} available"
                        + ("" if protected else f" past a {self.reserve} reserve")
                        + f"; {self._spent} already committed this tick"
                    ),
                )
            )
        self._spent += amount
        return self._record(
            Claim(
                purpose=purpose,
                amount=amount,
                granted=True,
                reason=f"{purpose} took {amount}, leaving {self.remaining()}",
            )
        )

    def _record(self, claim: Claim) -> Claim:
        """Append a claim to the ledger and hand it back.

        Args:
            claim: The claim to record.

        Returns:
            The same claim, so callers can record and return in one step.
        """
        self._ledger.append(claim)
        return claim


def format_ledger(ledger: tuple[Claim, ...]) -> tuple[str, ...]:
    """Render a tick's claims as report lines.

    Args:
        ledger: The claims, in the order they were made.

    Returns:
        One line per claim, marking each granted or refused.
    """
    return tuple(f"{'took' if claim['granted'] else 'held'}  {claim['reason']}" for claim in ledger)


__all__ = ["Budget", "BudgetError", "Claim", "format_ledger"]
