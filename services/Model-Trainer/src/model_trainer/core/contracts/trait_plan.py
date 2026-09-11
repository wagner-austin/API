"""What one trait-composition measurement DECLARES, before it is allowed to run.

THE SAME CONTRACT AS :mod:`~model_trainer.core.contracts.qa_plan`, one
substrate over. Every field here is either an input a number cannot be
reproduced without, or a commitment the run is checked against before a model
loads. ``smallest_effect_of_interest``, ``alpha`` and ``mcnemar_test`` are the
second kind: they are read by
:func:`~model_trainer.core.services.model.cartridge_qa_power.require_resolvable_pairs`
against the REALISED held-out pair count, and a plan that cannot resolve what
it declares is refused rather than run and regretted.

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
        smallest_effect_of_interest: The smallest net rate of items the
            measurement exists to resolve, in the units
            :func:`~model_trainer.core.contracts.paired_comparison.summarise_pairs`
            reports.

            A DECLARED VALUE MUST SAY WHETHER IT WAS MEASURED OR CHOSEN, and
            the table's comments do. For this axis it is CHOSEN, because the
            arc has never measured the variance of a contrastive logprob
            difference and inheriting the published 15-40 point figures would
            import a judge scale that does not transfer. Erring small is the
            safe direction: a too-large value licenses a verdict, a too-small
            one only refuses a run.
        alpha: Two-sided significance level the rejection region is fixed at.
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
        steering_strength: Multiplier applied to a unit steering direction
            when it is added back. Declared for the same reason the BM25 knobs
            are declared on the question-set plan: it moves the arm the
            cartridge is being compared against, so an arm reported without it
            says nothing a reader can reproduce.
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
    smallest_effect_of_interest: float
    alpha: float
    mcnemar_test: McNemarTest
    steering_module: str
    steering_strength: float


#: Fixed rather than a flag, and distinct from every other cartridge
#: experiment's name: this one's compartments are DISPOSITIONS, and a record
#: differenced against a corpus-compartment record would be comparing two
#: different questions that happen to share an arm vocabulary.
TRAIT_SWEEP_EXPERIMENT = "cartridge-trait-composition"


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
    seeds = ".".join(str(seed) for seed in plan["seeds"])
    counts = ".".join(str(count) for count in plan["compartment_counts"])
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
        f"-steer{plan['steering_module']}x{plan['steering_strength']}"
        f"-{digest[:12]}"
    )


__all__ = [
    "TRAIT_SWEEP_EXPERIMENT",
    "TraitPlan",
    "trait_plan_label",
]
