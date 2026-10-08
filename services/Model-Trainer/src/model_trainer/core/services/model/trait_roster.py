"""Reading a trait roster into measurable splits, and refusing one too small.

ONE OWNER FOR BOTH TRAIT SWEEPS. The naive grid
(``cartridge_trait_sweep``) and the repair families
(``cartridge_trait_repair_sweep``) score the same primary trait on the same
held-out pairs, so they must tokenise, split and gate the roster identically:
two copies would be two definitions of which pairs are scored, and records
whose arm names match while their held-out sets differ cannot be subtracted.

THE GATE RUNS BEFORE A MODEL LOADS, on the REALISED held-out count. A cap in a
plan is an upper bound the corpus is free to fall short of, and the retracted
question-set headline came from a plan whose cap said 120 over a set of 32.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.logging import get_logger
from platform_core.minimum_detectable_effect import required_replicates
from platform_core.power_distributions import t_critical
from platform_core.run_record import Observation

from model_trainer.core.contracts.trait_corpus import TraitCorpus
from model_trainer.core.contracts.trait_plan import TraitPlan
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_qa_power import require_resolvable_pairs
from model_trainer.core.services.model.cartridge_scoring import TraitPair
from model_trainer.core.services.model.trait_corpus import (
    split_trait_pairs,
    tokenise_trait_pairs,
)

_log = get_logger(__name__)


class SplitTrait:
    """One trait's pairs, already tokenised and split.

    A class with three read attributes rather than a TypedDict because it
    never crosses a serialisation boundary and holds tensors; it is the
    in-process shape the sweeps pass between preparation and measurement.

    Attributes:
        trait: The trait's name, which names every arm it appears in.
        train: Pairs the cartridge trains on and the direction is read from.
        held_out: Pairs every arm is scored on.
    """

    def __init__(self, trait: str, train: list[TraitPair], held_out: list[TraitPair]) -> None:
        """Hold one trait's split.

        Args:
            trait: The trait's name.
            train: Training pairs.
            held_out: Held-out pairs.
        """
        self.trait = trait
        self.train = train
        self.held_out = held_out


def prepare_traits(
    corpora: Sequence[TraitCorpus], plan: TraitPlan, *, device: str
) -> list[SplitTrait]:
    """Tokenise and split every trait in the roster, in roster order.

    Args:
        corpora: The authored corpora, in roster order.
        plan: The measurement being run.
        device: Torch device string to build the tensors on.

    Returns:
        One split per trait, in roster order.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` propagated from tokenisation
            or the split.
    """
    tokenizer = hf_hooks.Hooks.load_hf_tokenizer(plan["model_id"])
    prepared: list[SplitTrait] = []
    for corpus in corpora:
        pairs = tokenise_trait_pairs(
            corpus, tokenizer, max_seq_len=plan["max_seq_len"], device=device
        )
        train, held_out = split_trait_pairs(pairs, held_out_stride=plan["held_out_stride"])
        _log.info(
            "trait %s: %d pair(s), %d train / %d held out",
            corpus["trait"],
            len(pairs),
            len(train),
            len(held_out),
        )
        prepared.append(SplitTrait(corpus["trait"], train, held_out))
    return prepared


def require_resolvable_seeds(plan: TraitPlan) -> tuple[Observation, ...]:
    """Refuse a plan whose seeds cannot resolve the smallest effect of interest.

    THE SEI IS DERIVED, NOT CHOSEN, and not inherited from the published
    15-40 point figures either: those are on a judge's 0-100 scale, which
    does not transfer to a logprob difference. It is the smallest composition
    effect this arc has acted on -- a retention margin, a fraction of the
    alone gain -- carried into nats of expression by the pilot's solo gain,
    because a retention difference between two arms that share an alone arm
    is their composed-gain difference divided by that gain.

    THE SEEDS ARE WHAT RESOLVES IT, because composition arms are compared
    seed by seed: every arm of a cell trains under the same seeds, so two
    arms' gains at one seed are two measurements of one draw. The pilot's
    paired differences supply the spread, and
    :func:`~platform_core.minimum_detectable_effect.required_replicates`
    says how many seeds bring the minimum detectable effect down to the SEI.
    A plan declaring fewer is refused before a model loads.

    Args:
        plan: The measurement being run.

    Returns:
        The SEI's provenance and the gate's verdict as rows: the acted-on
        retention, the pilot's alone gain and paired sd, the SEI in nats,
        the seeds required, and the minimum detectable effect at the seeds
        the plan declares.

    Raises:
        AppError: With ``CARTRIDGE_QA_UNDERPOWERED`` when the plan's seeds are
            fewer than the pilot's spread requires, and propagated from
            :func:`required_replicates` when the pilot carries too few
            differences, the SEI is not positive, or no seed count suffices.
    """
    smallest_effect = plan["acted_on_retention"] * plan["pilot_alone_gain"]
    needed = required_replicates(
        list(plan["pilot_paired_differences"]), plan["alpha"], smallest_effect
    )
    declared = len(plan["seeds"])
    if declared < needed["required_replicates"]:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED,
            (
                f"this plan composes over {declared} seed(s), and at the pilot's paired sd "
                f"of {needed['observed_sample_sd']:.4f} nats a seed-paired comparison needs "
                f"{needed['required_replicates']} to resolve the smallest effect of interest, "
                f"{smallest_effect:.4f} nats (a {plan['acted_on_retention']:.4f} retention "
                f"margin on a {plan['pilot_alone_gain']:.4f}-nat solo gain); declare "
                f"{needed['required_replicates']} seeds or the composed comparisons this "
                f"plan exists to make cannot be told from seed noise"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_QA_UNDERPOWERED),
        )
    detectable = (
        t_critical(declared - 1, plan["alpha"]) * needed["observed_sample_sd"] / math.sqrt(declared)
    )
    _log.info(
        "%d seed(s) resolve %.4f nats of composed expression; the SEI is %.4f",
        declared,
        detectable,
        smallest_effect,
    )
    return (
        Observation(name="acted_on_retention", value=plan["acted_on_retention"]),
        Observation(name="pilot_alone_gain", value=plan["pilot_alone_gain"]),
        Observation(name="pilot_paired_sd", value=needed["observed_sample_sd"]),
        Observation(name="smallest_effect_of_interest", value=smallest_effect),
        Observation(name="required_seeds", value=float(needed["required_replicates"])),
        Observation(name="seed_paired_mde", value=detectable),
    )


def gate_primary_trait(plan: TraitPlan, primary: SplitTrait) -> tuple[Observation, ...]:
    """Run both of a trait plan's gates and name their numbers.

    Args:
        plan: The measurement being run.
        primary: The trait whose expression is the finding.

    Returns:
        The rows every trait record opens with: the realised pair counts, the
        per-pair test's resolvable and declared floors, the slot count, and
        the seed gate's rows from :func:`require_resolvable_seeds`.

    Raises:
        AppError: With ``CARTRIDGE_QA_UNDERPOWERED`` when the realised pair set
            cannot resolve the declared pair floor, or the seeds cannot
            resolve the smallest effect of interest.
    """
    floor = require_resolvable_pairs(
        len(primary.held_out),
        alpha=plan["alpha"],
        test=plan["mcnemar_test"],
        smallest_effect_of_interest=plan["pair_test_floor"],
        subject="pair",
    )
    _log.info(
        "%d held-out pair(s) on %s resolve nothing smaller than %.4f; the plan declares %.4f",
        len(primary.held_out),
        primary.trait,
        floor,
        plan["pair_test_floor"],
    )
    return (
        Observation(name="held_out_pairs", value=float(len(primary.held_out))),
        Observation(name="training_pairs", value=float(len(primary.train))),
        Observation(name="resolvable_floor", value=floor),
        Observation(name="pair_test_floor", value=plan["pair_test_floor"]),
        Observation(name="slots_per_cartridge", value=float(plan["slots"])),
        *require_resolvable_seeds(plan),
    )


__all__ = [
    "SplitTrait",
    "gate_primary_trait",
    "prepare_traits",
    "require_resolvable_seeds",
]
