"""Which trait-composition measurements exist, and what each one commits to.

WHY A TABLE AND NOT FLAGS, for the reason
:mod:`~model_trainer.core.services.model.cartridge_plans` gives: a gain means
nothing apart from the base, roster, schedule and seeds that produced it, and
a configuration assembled from a dozen flags is one nobody can reproduce
without also recovering the command line. The corpus is the deliberate
exception and stays a path, because it is data and it lives somewhere
different on every machine -- and the digest folded into the label is what
stops two different corpora being differenced against each other.

WHAT FIXES THE DECLARED EFFECT, AND WHY IT IS WHAT IT IS. The floor a paired
set can ever resolve is ``smallest_rejecting_discordant / held_out_pairs``,
and at alpha 0.05 under the exact test the numerator is 6. The committed
corpus authors 32 pairs per trait and the stride holds out half, so 16 pairs
are scored and the floor is 6/16 = 0.375. Every plan below declares exactly
that, because declaring anything smaller would be refused by
:func:`~model_trainer.core.services.model.cartridge_qa_power.require_resolvable_pairs`
before a model loaded -- correctly, and that refusal is the feature. A tighter
plan is a CORPUS decision, not a plan decision: halving the declared effect
means doubling the authored pairs, and the gate is what makes that visible
before the GPU hours rather than after the retraction.

WHY THE STEERING SITE DIFFERS BETWEEN RUNGS. It is a layer index into a
specific model and the two bases have different depths, so one value cannot
serve both. Both sit at two thirds of depth -- gpt2 has 12 blocks and
gpt2-medium 24 -- which is a CHOSEN convention rather than a measured optimum,
and the record carries the site so a later sweep over depth can be differenced
against these.
"""

from __future__ import annotations

from typing import Final

from platform_core.power_distributions import McNemarTest

from model_trainer.core.contracts.trait_plan import TraitPlan

#: The roster every plan draws from, in the order the corpus is authored.
#:
#: FOUR RATHER THAN ALL SIX ADMISSIBLE TRAITS, and the number is set by the
#: grid rather than by taste: the largest compartment count is 4, so a roster
#: of four is exactly what an n4 cell consumes. A fifth trait would be
#: authored pairs no cell reads.
_ROSTER: Final[tuple[str, ...]] = (
    "bullets",
    "step-by-step",
    "formal-tone",
    "rhetorical-questions",
)

#: The roster rotated, so roster IDENTITY can be told from roster ORDER.
#:
#: The corpus arc measured this and it mattered: two-compartment retention
#: moved by about seven points depending on which corpora partnered, which is
#: the same size as some of the effects being claimed. A rotation is the
#: cheapest control that can catch it, and it costs one more plan rather than
#: a new instrument.
_ROTATED: Final[tuple[str, ...]] = (
    "rhetorical-questions",
    "formal-tone",
    "step-by-step",
    "bullets",
)

#: The declared effect every plan commits to, in net item rate.
#:
#: CHOSEN, NOT MEASURED, and stated as such because this field's sibling on
#: the question-set plan was once averaged from real anchors and then
#: described as a floor. It is the smallest effect the committed corpus can
#: resolve at all; erring small would only refuse the run, which is the safe
#: direction.
_DECLARED_EFFECT: Final[float] = 0.375

TRAIT_SWEEP_PLANS: Final[dict[str, TraitPlan]] = {
    "gpt2-traits": TraitPlan(
        model_id="gpt2",
        traits=_ROSTER,
        held_out_stride=2,
        max_seq_len=128,
        slots=64,
        seeds=(7, 8, 9),
        epochs=12,
        learning_rate=0.01,
        compartment_counts=(2, 4),
        smallest_effect_of_interest=_DECLARED_EFFECT,
        alpha=0.05,
        mcnemar_test=McNemarTest.EXACT,
        steering_module="transformer.h.8.mlp.c_proj",
        steering_strength=1.0,
    ),
    # The roster control, and the ONLY field that differs from the plan above.
    # Anything else moving between them would make a difference in their
    # numbers attributable to two things at once.
    "gpt2-traits-rotated": TraitPlan(
        model_id="gpt2",
        traits=_ROTATED,
        held_out_stride=2,
        max_seq_len=128,
        slots=64,
        seeds=(7, 8, 9),
        epochs=12,
        learning_rate=0.01,
        compartment_counts=(2, 4),
        smallest_effect_of_interest=_DECLARED_EFFECT,
        alpha=0.05,
        mcnemar_test=McNemarTest.EXACT,
        steering_module="transformer.h.8.mlp.c_proj",
        steering_strength=1.0,
    ),
    # The depth rung. The corpus arc's findings moved between gpt2 and
    # gpt2-medium in ways nothing on the smaller model predicted -- the
    # untrained-prefix control's sign among them -- so a second rung is not a
    # confirmation run, it is where the surprises have historically been.
    "gpt2-medium-traits": TraitPlan(
        model_id="gpt2-medium",
        traits=_ROSTER,
        held_out_stride=2,
        max_seq_len=128,
        slots=64,
        seeds=(7, 8, 9),
        epochs=12,
        learning_rate=0.01,
        compartment_counts=(2, 4),
        smallest_effect_of_interest=_DECLARED_EFFECT,
        alpha=0.05,
        mcnemar_test=McNemarTest.EXACT,
        steering_module="transformer.h.16.mlp.c_proj",
        steering_strength=1.0,
    ),
}


__all__ = ["TRAIT_SWEEP_PLANS"]
