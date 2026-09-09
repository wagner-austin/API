"""Refusing a question-set measurement that cannot resolve what it declares.

WHY THIS RUNS BEFORE THE ARMS DO. On 2026-09-09 this programme's headline --
"the cartridge arm beats lexical, dense and fused retrieval from ~774M" -- was
retracted. The arms differed by 0.0521 and 0.0417 accuracy at the two rungs
that carried it, which over 32 items is 1.7 and 1.3 items. The smallest net
difference 32 items can resolve under any attainable outcome is 5 items,
0.1562. The claim was roughly four times below the floor of the instrument
that produced it, and no split of the data could have supported it.

Nothing was wrong with the analysis afterwards; the analysis was excellent and
is what caught it. What was missing was a PRECONDITION. The run had already
spent its GPU hours, published a page and reached a wiki hub before anyone
asked whether the question set could resolve the difference being claimed.

So a plan now DECLARES the smallest difference it intends to resolve, and this
module refuses the run when the realised question set cannot resolve it under
any outcome. The refusal is an exception, not a warning: a warning on a
measurement that takes hours is a line in a log nobody reads until the
retraction.

WHERE THE FLOOR COMES FROM, AND THE MISTAKE THAT IS EASY TO MAKE HERE. McNemar
conditions on the DISCORDANT pairs, so the resolvable difference depends on a
count nobody knows before running. The tempting shortcut is to evaluate at
``discordant == item_count`` and call it the best case. That is the WORST
case: the rejecting margin ``discordant - 2 * minority`` GROWS with the
discordant count, roughly as its square root, so a set where the arms disagree
everywhere needs a larger absolute difference than one where they disagree
rarely.

The margin is also NOT MONOTONIC in the discordant count -- measured over
d = 5..399 under mid-p at alpha 0.05, it oscillates by one item with parity
(d=7 needs 7, d=8 needs 6) -- so the floor cannot be found by assuming a
direction. It is the minimum over every attainable d, and that minimum sits at
the SMALLEST d that can reject at all: 5 pairs under mid-p at alpha 0.05, 6
under the exact test, 7 and 8 respectively at alpha 0.01. Below that d no
split rejects however lopsided it is.

So the floor is ``smallest_rejecting_discordant / item_count``, and a plan
declaring an effect under it is asking for a number no outcome can produce.

WHAT PASSING THIS GATE DOES NOT MEAN. It is a FALSIFIABILITY check, not a
power guarantee. It says some attainable outcome would support the declared
effect; it does not say the outcome is likely. Classifying an observed result
belongs after the run, against the discordant count that actually occurred,
and is where an 80%-power minimum detectable effect belongs. Conflating the
two is the confusion the whole 2026-09-08 sweep exists to end, and this module
refuses only what is impossible so that the refusal is never arguable.

WHY NO ``_test_hooks.py``. Following the reasoning
:mod:`platform_core.minimum_detectable_effect` states for itself: every
function here is a pure transformation of its arguments, with no clock,
network, filesystem or randomness. A hooks file would be a seam with nothing
behind it. Tests call the real functions against the real power module.
"""

from __future__ import annotations

import math
from typing import Final

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.minimum_detectable_effect import mcnemar_power
from platform_core.power_distributions import McNemarTest

from model_trainer.core.services.model.cartridge_qa_plans import QaPlan

#: How far the search for a rejecting discordant count runs before giving up.
#: The smallest rejecting count is 5 to 8 across the variants and alphas this
#: programme uses, so a ceiling of 64 is two orders of margin. It exists so
#: that an alpha small enough to be unreachable fails loudly instead of
#: looping.
_DISCORDANT_SEARCH_CEILING: Final[int] = 64


def smallest_rejecting_discordant(alpha: float, test: McNemarTest) -> int:
    """Find the fewest disagreements that could ever reject.

    Args:
        alpha: Two-sided significance level the comparison is judged at.
        test: Which McNemar variant the comparison is reported under.

    Returns:
        The smallest discordant count for which some attainable split rejects.
        Measured values: 5 for mid-p at alpha 0.05, 6 for exact at 0.05, 7 and
        8 respectively at 0.01.

    Raises:
        AppError: With ``CARTRIDGE_QA_UNDERPOWERED`` when no count at or below
            :data:`_DISCORDANT_SEARCH_CEILING` rejects, which means the alpha
            asked for is not reachable by this test at any plausible size.
    """
    for discordant in range(_DISCORDANT_SEARCH_CEILING + 1):
        if mcnemar_power(discordant, alpha, test)["can_ever_reject"]:
            return discordant
    raise AppError(
        ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED,
        (
            f"no discordant count at or below {_DISCORDANT_SEARCH_CEILING} rejects at alpha "
            f"{alpha!r} under the {test.value} test, so no question set of any size can "
            f"produce a significant result at that alpha; the alpha is the thing to change"
        ),
        model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED),
    )


def resolvable_floor(item_count: int, alpha: float, test: McNemarTest) -> float:
    """State the smallest difference a question set could ever resolve.

    Args:
        item_count: How many items the question set holds. Must be positive;
            an empty set resolves nothing and has no floor to report.
        alpha: Two-sided significance level.
        test: Which McNemar variant the comparison is reported under.

    Returns:
        The smallest net accuracy difference some attainable outcome would
        report as significant, in the units the arms are scored in.

    Raises:
        AppError: With ``CARTRIDGE_QA_UNDERPOWERED`` when the question set is
            empty, or when no discordant count rejects at this alpha.
    """
    if item_count < 1:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED,
            (
                f"a question set of {item_count!r} item(s) resolves nothing and has no floor; "
                f"the corpus produced no items to score, which is a corpus failure surfacing "
                f"here rather than a power one"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED),
        )
    return smallest_rejecting_discordant(alpha, test) / item_count


def require_resolvable_question_set(plan: QaPlan, item_count: int) -> float:
    """Refuse a measurement whose question set is too small for its own claim.

    Args:
        plan: The measurement being run. Its ``smallest_effect_of_interest``
            is the difference the plan exists to resolve, and its ``alpha``
            and ``mcnemar_test`` fix the rejection region it is judged against.
        item_count: How many items the corpus actually yielded. The REALISED
            count, never the plan's ``max_items`` cap -- the cap is an upper
            bound the corpus is free to fall short of, and the 32-item set
            behind the retracted headline came from a plan whose cap said 120.

    Returns:
        The resolvable floor, for the caller to carry into its ``RunRecord``
        so every number ships beside the smallest difference that produced it
        could have been significant.

    Raises:
        AppError: With ``CARTRIDGE_QA_UNDERPOWERED`` when the floor is above
            the difference the plan declares it is looking for.
    """
    floor = resolvable_floor(item_count, plan["alpha"], plan["mcnemar_test"])
    if floor > plan["smallest_effect_of_interest"]:
        # BOTH UNITS, AND THE ITEMS COME FIRST. A rate reads like a
        # measurement whatever its size -- "+0.0417" looks like a finding --
        # while "1.3 items of 32" reads as what it is. The retracted headline
        # was legible as wrong only in hindsight for exactly that reason, so
        # the refusal states the count it would take and the count the plan
        # is asking about, and only then the rates.
        rejecting = smallest_rejecting_discordant(plan["alpha"], plan["mcnemar_test"])
        wanted = plan["smallest_effect_of_interest"] * item_count
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED,
            (
                f"this plan is hunting {wanted:.1f} item(s) of {item_count}, and the fewest "
                f"that can ever reject is {rejecting} of {item_count} -- at alpha "
                f"{plan['alpha']!r} under the {plan['mcnemar_test'].value} test, "
                f"{item_count} item(s) resolve nothing smaller than {floor:.4f} while the plan "
                f"declares {plan['smallest_effect_of_interest']:.4f}; the corpus must yield at "
                f"least {_required_items(plan)} items before this measurement can produce a "
                f"number any split of the data would support"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED),
        )
    return floor


def _required_items(plan: QaPlan) -> int:
    """Say how many items the plan would need, rather than only that it failed.

    A refusal that names a target is a next action; one that does not is an
    obstacle. The count is the smallest ``n`` whose floor reaches the declared
    effect, which inverts directly.

    Args:
        plan: The measurement being run.

    Returns:
        The fewest items whose resolvable floor is at or below the plan's
        ``smallest_effect_of_interest``.

    Raises:
        AppError: With ``CARTRIDGE_QA_UNDERPOWERED`` when no discordant count
            rejects at the plan's alpha.
    """
    rejecting = smallest_rejecting_discordant(plan["alpha"], plan["mcnemar_test"])
    return math.ceil(rejecting / plan["smallest_effect_of_interest"])


__all__ = [
    "require_resolvable_question_set",
    "resolvable_floor",
    "smallest_rejecting_discordant",
]
