"""Which trait-composition measurements exist, and what each one commits to.

WHY A TABLE AND NOT FLAGS, for the reason
:mod:`~model_trainer.core.services.model.cartridge_plans` gives: a gain means
nothing apart from the base, roster, schedule and seeds that produced it, and
a configuration assembled from a dozen flags is one nobody can reproduce
without also recovering the command line. The corpus is the deliberate
exception and stays a path, because it is data and it lives somewhere
different on every machine -- and the digest folded into the label is what
stops two different corpora being differenced against each other.

TWO GATES, AND ONLY ONE OF THEM IS A SMALLEST EFFECT OF INTEREST.

* ``pair_test_floor`` is the committed corpus's own floor for the per-pair
  McNemar test the steering arm reports: 32 authored pairs per trait, half
  held out, and 6 of 16 is the smallest net rate that can ever reject at
  alpha 0.05 under the exact test. It says that test is falsifiable on this
  corpus and nothing more -- a threshold chosen so the corpus clears it is
  not a threshold, which is exactly why it is no longer called one.
* The smallest effect of interest is DERIVED, the way
  ``clients/RustedWarfareBot``'s power audit derives its own: from the
  smallest effect this arc has ever acted on, carried into the instrument's
  unit by the variance pilot, and enforced against the seeds by
  :func:`~model_trainer.core.services.model.trait_roster.require_resolvable_seeds`.

THE PILOT. Plan ``gpt2-traits`` at seeds 7, 8, 9 on image v56, CPU, twin jobs
57928000 and 57928006 on two nodes, which logged identical numbers: the
bullets solo arm gained +2.6144 nats of expression, and the per-seed
differences between its n2 and n4 composed arms were 2.0248, 0.5953 and
0.6317 (paired sd 0.815). At the derived 0.1375 nats that sd needs 138 seeds,
so every plan below declares 138 -- one pilot block and 45 more.

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

#: The per-pair McNemar floor of the committed corpus: 6 of 16 held-out pairs.
_PAIR_TEST_FLOOR: Final[float] = 0.375

#: The smallest composition effect this arc has acted on, in retention.
#:
#: Base-LoRA plus diverse cartridges retained 0.3326 at eight compartments
#: against diverse alone's 0.2800, and that margin is what the operating
#: point of record was adopted on. The four-compartment margin, 0.0257, is
#: smaller but was never acted on -- the adoption read the eight-compartment
#: cell -- so it is not this arc's smallest acted-on effect.
_ACTED_ON_RETENTION: Final[float] = 0.0526

#: The pilot's solo expression gain, which carries the retention into nats.
_PILOT_ALONE_GAIN: Final[float] = 2.6144235928853354

#: The pilot's per-seed n2-minus-n4 composed expression, seeds 7, 8 and 9.
#:
#: Two composed arms of one record, paired by seed: the shape of every
#: comparison a repair family is read by.
_PILOT_PAIRED_DIFFERENCES: Final[tuple[float, ...]] = (
    2.0247728675603867,
    0.595255434513092,
    0.6316859424114227,
)

#: 138 seeds, the count the pilot's spread requires; seeds 7, 8, 9 first.
_SEEDS: Final[tuple[int, ...]] = tuple(range(7, 7 + 138))

#: Candidate strengths for a UNIT steering direction.
#:
#: The published tuning searches coefficients of 1 to 5 on raw vectors; this
#: site's raw contrastive vector on gpt2 has norm about 22, so that band is
#: unit strengths of roughly 22 to 110, and the grid brackets it from both
#: sides. A local probe on the training pairs found expression rising to
#: about strength 32 and collapsing past 64 as the coherence cost grew.
_STEERING_STRENGTHS: Final[tuple[float, ...]] = (
    4.0,
    8.0,
    16.0,
    24.0,
    32.0,
    48.0,
    64.0,
    96.0,
    128.0,
)

#: The coherence cost a tuned steering strength may incur, in nats: the
#: pilot's solo cartridge's own, so the two substrates are compared at the
#: fluency the cartridge itself paid.
_STEERING_COHERENCE_BAR: Final[float] = 1.861071730653445

TRAIT_SWEEP_PLANS: Final[dict[str, TraitPlan]] = {
    "gpt2-traits": TraitPlan(
        model_id="gpt2",
        traits=_ROSTER,
        held_out_stride=2,
        max_seq_len=128,
        slots=64,
        seeds=_SEEDS,
        epochs=12,
        learning_rate=0.01,
        compartment_counts=(2, 4),
        pair_test_floor=_PAIR_TEST_FLOOR,
        acted_on_retention=_ACTED_ON_RETENTION,
        pilot_alone_gain=_PILOT_ALONE_GAIN,
        pilot_paired_differences=_PILOT_PAIRED_DIFFERENCES,
        alpha=0.05,
        mcnemar_test=McNemarTest.EXACT,
        steering_module="transformer.h.8.mlp.c_proj",
        steering_strengths=_STEERING_STRENGTHS,
        steering_coherence_bar=_STEERING_COHERENCE_BAR,
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
        seeds=_SEEDS,
        epochs=12,
        learning_rate=0.01,
        compartment_counts=(2, 4),
        pair_test_floor=_PAIR_TEST_FLOOR,
        acted_on_retention=_ACTED_ON_RETENTION,
        pilot_alone_gain=_PILOT_ALONE_GAIN,
        pilot_paired_differences=_PILOT_PAIRED_DIFFERENCES,
        alpha=0.05,
        mcnemar_test=McNemarTest.EXACT,
        steering_module="transformer.h.8.mlp.c_proj",
        steering_strengths=_STEERING_STRENGTHS,
        steering_coherence_bar=_STEERING_COHERENCE_BAR,
    ),
    # The depth rung. The corpus arc's findings moved between gpt2 and
    # gpt2-medium in ways nothing on the smaller model predicted -- the
    # untrained-prefix control's sign among them -- so a second rung is not a
    # confirmation run, it is where the surprises have historically been.
    # Its pilot is BORROWED from gpt2, stated rather than hidden: no medium
    # pilot has run, and its seeds and bar are the small rung's until one does.
    "gpt2-medium-traits": TraitPlan(
        model_id="gpt2-medium",
        traits=_ROSTER,
        held_out_stride=2,
        max_seq_len=128,
        slots=64,
        seeds=_SEEDS,
        epochs=12,
        learning_rate=0.01,
        compartment_counts=(2, 4),
        pair_test_floor=_PAIR_TEST_FLOOR,
        acted_on_retention=_ACTED_ON_RETENTION,
        pilot_alone_gain=_PILOT_ALONE_GAIN,
        pilot_paired_differences=_PILOT_PAIRED_DIFFERENCES,
        alpha=0.05,
        mcnemar_test=McNemarTest.EXACT,
        steering_module="transformer.h.16.mlp.c_proj",
        steering_strengths=_STEERING_STRENGTHS,
        steering_coherence_bar=_STEERING_COHERENCE_BAR,
    ),
}


__all__ = ["TRAIT_SWEEP_PLANS"]
