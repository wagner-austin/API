"""What one trait-composition measurement DECLARES, before it is allowed to run.

THE SAME CONTRACT AS :mod:`~model_trainer.core.contracts.qa_plan`, one
substrate over. Every field here is either an input a number cannot be
reproduced without, or a commitment the run is checked against before a model
loads. Two commitments are checked, on two different scales:

* ``pair_test_floor``, ``alpha`` and ``mcnemar_test`` are read by
  :func:`~model_trainer.core.services.model.cartridge_qa_power.require_resolvable_pairs`
  against the REALISED held-out pair count: a falsifiability check on the
  per-pair McNemar test the steering arm reports. It is NOT a smallest effect
  of interest, because this arc has never acted on a net-pair-rate effect
  and so has nothing to derive one from in that unit.
* ``acted_on_retention``, ``pilot_alone_gain`` and
  ``pilot_paired_differences`` are read by
  :func:`~model_trainer.core.services.model.trait_roster.require_resolvable_seeds`:
  the SMALLEST EFFECT OF INTEREST, derived from the smallest effect this arc
  has acted on and carried into the instrument's own unit by the pilot's
  solo gain, against the seed count the pilot's paired spread says resolves
  it. That is where composition arms are compared, so that is where the
  smallest effect of interest lives.

WHY THERE IS NO DECODER HERE, and the asymmetry with every other contract in
this package is deliberate. A plan is a constant in a committed table; it
never crosses a JSON boundary in either direction, so there is no untrusted
value to validate and a decoder would be a function with no caller. What DOES
cross that boundary -- the authored pairs, the checkpoint, the record -- has
one, in :mod:`~model_trainer.core.contracts.trait_corpus`,
:mod:`~model_trainer.core.contracts.sweep_checkpoint` and
:mod:`platform_core.run_record` respectively.

WHY THE TRAITS ARE A TUPLE AND THE ORDER IS PART OF THE MEASUREMENT. The first
trait is the one whose expression is the finding; the rest are compartments
composed in front of it, in the order the compartment counts consume them. The
same traits in another order are a different measurement, and the label says
so -- which is what stops two rosters being differenced against each other.
"""

from __future__ import annotations

import itertools

from platform_core.power_distributions import McNemarTest
from typing_extensions import TypedDict


class TraitPlan(TypedDict):
    """One complete, reproducible trait-composition measurement.

    Attributes:
        model_id: HuggingFace id of the base to measure against. A cartridge
            is a block of one model's own attention keys and values and a
            steering vector is a direction in one model's residual stream, so
            neither transfers between bases.
        traits: Which traits this plan draws, in roster order. The first is
            the trait whose expression is the finding; the rest are composed
            in front of it. Every name must be a file in the staged corpus and
            one of
            :data:`~model_trainer.core.contracts.trait_corpus.CONSISTENTLY_EFFECTIVE_TRAITS`.
        held_out_stride: One pair in this many is scored; the cartridge trains
            on the rest. Two holds out half, which is the largest held-out set
            a split can give and therefore the lowest resolvable floor this
            corpus can reach.
        max_seq_len: Token budget a single continuation may not exceed. A
            member over it is refused rather than truncated, because
            truncation removes the tokens the trait is carried by and still
            produces a number.
        slots: Prefix positions for EACH trait cartridge; a composed prefix is
            this times the compartment count.
        seeds: Initialisation seeds. Every arm runs once per seed, and the
            spread across them is what this plan's means are judged against.
        epochs: Passes over a trait's training pairs.
        learning_rate: Step size for AdamW.
        compartment_counts: How many trait cartridges are composed, in
            increasing order. ``len(traits)`` must reach the largest.
        pair_test_floor: The net pair rate the per-pair McNemar test is
            declared to resolve, in the units
            :func:`~model_trainer.core.contracts.paired_comparison.summarise_pairs`
            reports. It is the committed corpus's own floor and says so: a
            value chosen so the corpus clears it is not a threshold, which is
            why the smallest effect of interest is the next three fields.
        acted_on_retention: The smallest composition effect this arc has ever
            acted on, as a fraction of the alone gain: base-LoRA plus diverse
            cartridges retaining 0.3326 at eight compartments against diverse
            alone's 0.2800, the margin the operating point of record was
            adopted on. Derived, not chosen, the way
            ``clients/RustedWarfareBot``'s power audit derives its SEI.
        pilot_alone_gain: The solo arm's mean expression gain in the pilot
            record, which carries the retention above into nats of
            expression: a retention difference IS a composed-gain difference
            divided by the alone gain.
        pilot_paired_differences: Per-seed differences between two composed
            arms of the pilot record, the spread every composed comparison
            this plan makes is resolved against.
        alpha: Two-sided significance level the rejection region is fixed at,
            for both the per-pair test and the seed-paired one.
        mcnemar_test: Which McNemar variant the per-arm comparison is reported
            under. Carried rather than assumed because the exact and mid-p
            rejection regions differ, so a power statement computed against
            the wrong one describes a test nobody ran.
        steering_module: Dotted path of the module whose output the
            steering-vector arm reads and perturbs. Declared rather than
            derived from the depth, because the layer a contrastive direction
            is readable at is a property of the model that this programme has
            not measured, and a derived one would put an unchosen number in
            every record.
        steering_strengths: The candidate multipliers for a unit steering
            direction, in the order tried. The arm is applied at the most
            expressive one that stays within ``steering_coherence_bar`` on the
            training pairs -- the published tuning rule -- so the grid is
            declared and the choice is recorded rather than either being
            typed in as one number.
        steering_coherence_bar: Largest coherence cost, in nats, the tuned
            steering strength may incur. Set to the trait cartridge's own solo
            coherence cost in the pilot, so the two substrates are compared at
            matched fluency rather than at whatever strength was typed.
    """

    model_id: str
    traits: tuple[str, ...]
    held_out_stride: int
    max_seq_len: int
    slots: int
    seeds: tuple[int, ...]
    epochs: int
    learning_rate: float
    compartment_counts: tuple[int, ...]
    pair_test_floor: float
    acted_on_retention: float
    pilot_alone_gain: float
    pilot_paired_differences: tuple[float, ...]
    alpha: float
    mcnemar_test: McNemarTest
    steering_module: str
    steering_strengths: tuple[float, ...]
    steering_coherence_bar: float


#: Fixed rather than a flag, and distinct from every other cartridge
#: experiment's name: this one's compartments are DISPOSITIONS, and a record
#: differenced against a corpus-compartment record would be comparing two
#: different questions that happen to share an arm vocabulary.
TRAIT_SWEEP_EXPERIMENT = "cartridge-trait-composition"


def seed_token(seeds: tuple[int, ...]) -> str:
    """Spell a plan's seeds for its label, compactly when they are a run.

    A pilot's three seeds read ``7.8.9`` as they always have. A plan sized by
    its pilot can carry a hundred or more, and listing them would make the
    label longer than anything that reads it; a contiguous ascending run of
    four or more is ``first..last`` instead, which names exactly the same set
    and cannot be mistaken for a dotted list.

    Args:
        seeds: The plan's seeds, in order.

    Returns:
        The token.
    """
    contiguous = all(later == earlier + 1 for earlier, later in itertools.pairwise(seeds))
    if contiguous and len(seeds) >= 4:
        return f"{seeds[0]}..{seeds[-1]}"
    return ".".join(str(seed) for seed in seeds)


def trait_plan_label(name: str, plan: TraitPlan, *, digest: str) -> str:
    """Build the label that identifies one plan's numbers on one corpus.

    Every field that moves a number appears, including the seeds and the
    roster order: a plan re-run on a different draw or a rotated roster would
    otherwise register under an existing name and be differenced against
    numbers it cannot reproduce.

    Args:
        name: The plan's name.
        plan: The plan.
        digest: Digest of the trait corpora, from
            :func:`~model_trainer.core.contracts.trait_corpus.trait_corpus_digest`.
            Truncated into the label -- the full value is long, and twelve hex
            characters distinguish any two corpora anybody will run.

    Returns:
        The label.
    """
    traits = ".".join(plan["traits"])
    seeds = seed_token(plan["seeds"])
    counts = ".".join(str(count) for count in plan["compartment_counts"])
    strengths = ".".join(f"{strength:g}" for strength in plan["steering_strengths"])
    return (
        f"{name}"
        f"-{plan['model_id']}"
        f"-traits{traits}"
        f"-s{plan['held_out_stride']}"
        f"-e{plan['epochs']}"
        f"-lr{plan['learning_rate']}"
        f"-slots{plan['slots']}"
        f"-n{counts}"
        f"-seeds{seeds}"
        f"-steer{plan['steering_module']}x{strengths}-bar{plan['steering_coherence_bar']}"
        f"-{digest[:12]}"
    )


__all__ = [
    "TRAIT_SWEEP_EXPERIMENT",
    "TraitPlan",
    "seed_token",
    "trait_plan_label",
]
