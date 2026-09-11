"""One trait, and the matched continuations that measure whether it is there.

WHAT A TRAIT CORPUS IS. Per trait, a set of prompts each continued two ways:
once exhibiting the trait, once not. The two members are the measurement --
:mod:`~model_trainer.core.services.model.cartridge_scoring` differences them,
so whatever the prefix does to both cancels -- and every way they differ
BEYOND the trait is an alternative explanation for the score. Length, topic,
vocabulary, and the position the difference falls at are all such ways, which
is why the pairs are authored and committed rather than generated at run time
from a template nobody would re-read.

WHY A FILE PER TRAIT AND NOT ONE FILE. A composition grid draws traits as
compartments, so the unit a run selects is a trait; one file per trait makes
the selection a path list, makes a trait's digest its own, and means adding a
trait cannot change the bytes of any other. It is also what
:mod:`hpc3` stages: a directory of files, each digested on both sides.

WHY THE TRAIT NAME IS IN THE FILE AND NOT ONLY IN ITS NAME. A filename is not
carried into a record. The trait names the arms, appears in the label, and is
what a later reader pairs two runs by, so it has to survive a copy -- and a
file whose declared trait disagrees with its own filename is refused rather
than silently preferring one of them.

THE TRAITS THIS PROGRAMME MAY DRAW FROM ARE NOT ARBITRARY, and the constraint
is measured rather than stylistic: Subbiah's limits-of-steering-vectors work
reports that positionally concentrated traits -- rhyme, which matters at line
ends, and screenplay or tweet formatting, which matter at the margins -- give
a null that measures trait CHOICE rather than composition. The admissible set
is :data:`CONSISTENTLY_EFFECTIVE_TRAITS`, and a corpus outside it is refused
here rather than debated in a closure.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final

from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

#: Filename extension every trait corpus file carries.
TRAIT_CORPUS_SUFFIX: Final[str] = ".json"

#: The traits a plan may draw from, and the ONLY ones.
#:
#: Taken from the published set that expressed CONSISTENTLY under steering --
#: the complement is the positionally concentrated set (rhyming structure,
#: screenplay format, tweet style, third person), where a composition null
#: cannot be told apart from the trait simply being hard to express at all.
#: Frozen rather than a plan field, because a plan free to choose its own
#: traits could make a composition result by choosing easy ones, and the
#: choice would then be the finding.
CONSISTENTLY_EFFECTIVE_TRAITS: Final[frozenset[str]] = frozenset(
    {
        "bullets",
        "step-by-step",
        "formal-tone",
        "rhetorical-questions",
        "all-caps-emphasis",
        "informal-tone",
    }
)


class TraitPairSpec(TypedDict):
    """One prompt continued two ways, as authored.

    Attributes:
        prompt: The shared opening. Identical for both members by
            construction, so no part of the score can come from the prompt.
        expressing: The continuation that exhibits the trait.
        neutral: The continuation that does not. Matched to ``expressing`` in
            content as closely as the trait allows.
    """

    prompt: str
    expressing: str
    neutral: str


class TraitCorpus(TypedDict):
    """Every pair that measures one trait.

    Attributes:
        trait: Which trait these pairs measure. Names the arms and appears in
            the label, so it must be one of
            :data:`CONSISTENTLY_EFFECTIVE_TRAITS`.
        pairs: The pairs, in the order authored. The ORDER IS PART OF THE
            CONTRACT: outcomes are indexed by it, and two arms are paired item
            by item, so a reordering makes two runs' per-item comparisons
            describe different texts under the same indices.
    """

    trait: str
    pairs: list[TraitPairSpec]


def _require_nonempty(obj: JSONObject, key: str) -> str:
    """Read a required string field that may not be blank.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing or is not a string.
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if the value is empty or
            only whitespace. An empty continuation scores a loss over no
            tokens, which is not a small measurement but an undefined one.
    """
    value = require_str(obj, key)
    if value.strip():
        return value
    raise AppError(
        ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE,
        (
            f"trait pair field {key!r} is empty; both members of a pair are scored as "
            f"token sequences and an empty one has no loss to compare, so the pair "
            f"would contribute an undefined number rather than a small one"
        ),
        model_trainer_status_for(ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE),
    )


def decode_trait_pair_spec(value: JSONValue) -> TraitPairSpec:
    """Decode and validate one authored pair.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated pair.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing or
            mistyped.
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if a field is blank, or if
            the two continuations are identical -- a pair whose members do not
            differ measures nothing and scores exactly zero, which would enter
            the record as a tie rather than as the authoring mistake it is.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"trait pair must be a JSON object, got {type(value).__name__}")
    spec = TraitPairSpec(
        prompt=_require_nonempty(value, "prompt"),
        expressing=_require_nonempty(value, "expressing"),
        neutral=_require_nonempty(value, "neutral"),
    )
    if spec["expressing"] == spec["neutral"]:
        raise AppError(
            ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE,
            (
                f"the two continuations of the pair beginning {spec['prompt'][:40]!r} are "
                f"identical, so the difference of differences is exactly zero by "
                f"construction; it would be recorded as a tie, which is indistinguishable "
                f"from a trait the cartridge failed to express"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE),
        )
    return spec


def decode_trait_corpus(value: JSONValue) -> TraitCorpus:
    """Decode and validate one trait's whole pair set.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated corpus.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing or
            mistyped.
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if the trait is not one of
            :data:`CONSISTENTLY_EFFECTIVE_TRAITS`, if it holds no pairs, or if
            any pair is invalid.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"trait corpus must be a JSON object, got {type(value).__name__}")
    trait = require_str(value, "trait")
    if trait not in CONSISTENTLY_EFFECTIVE_TRAITS:
        raise AppError(
            ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE,
            (
                f"trait {trait!r} is not one of the consistently-effective traits "
                f"({', '.join(sorted(CONSISTENTLY_EFFECTIVE_TRAITS))}); the excluded ones "
                f"are positionally concentrated, so a composition null drawn from them "
                f"measures trait choice rather than composition"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE),
        )
    pairs = [decode_trait_pair_spec(entry) for entry in require_list(value, "pairs")]
    if not pairs:
        raise AppError(
            ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE,
            (
                f"trait {trait!r} carries no pairs, so every arm measured on it would "
                f"report zero items and a p-value of one; that is what 'nothing was "
                f"measured' looks like and it must not reach a record as a null result"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE),
        )
    return TraitCorpus(trait=trait, pairs=pairs)


def encode_trait_pair_spec(spec: TraitPairSpec) -> JSONObject:
    """Encode one authored pair.

    Args:
        spec: The pair to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "prompt": spec["prompt"],
        "expressing": spec["expressing"],
        "neutral": spec["neutral"],
    }


def encode_trait_corpus(corpus: TraitCorpus) -> JSONObject:
    """Encode one trait's pair set.

    Args:
        corpus: The corpus to encode.

    Returns:
        JSON-serialisable mapping carrying every field, pairs in order.
    """
    pairs: list[JSONValue] = [encode_trait_pair_spec(spec) for spec in corpus["pairs"]]
    return {"trait": corpus["trait"], "pairs": pairs}


def trait_corpus_digest(corpora: Sequence[TraitCorpus]) -> str:
    """Digest the exact text a trait measurement will train and score on.

    EVERY FIELD AND EVERY BOUNDARY, length-prefixed, for the reason
    :func:`~model_trainer.core.services.model.cartridge_plans.digest_parts`
    gives: concatenation alone would let two different splits of the same
    characters collide, and here a split IS the measurement -- the prompt
    boundary decides what both members share, and the member boundary decides
    what is differenced.

    Args:
        corpora: The traits, in the order the run consumes them. Order is
            significant: the first is the trait whose expression is the
            finding and the rest are compartments composed in front of it, so
            the same traits in another order are a different measurement.

    Returns:
        Hex digest over every trait name and every pair field.
    """
    accumulator = hashlib.sha256()
    for corpus in corpora:
        for part in (
            corpus["trait"],
            *(
                field
                for spec in corpus["pairs"]
                for field in (spec["prompt"], spec["expressing"], spec["neutral"])
            ),
        ):
            accumulator.update(str(len(part)).encode("utf-8"))
            accumulator.update(b"\x00")
            accumulator.update(part.encode("utf-8"))
    return accumulator.hexdigest()


__all__ = [
    "CONSISTENTLY_EFFECTIVE_TRAITS",
    "TRAIT_CORPUS_SUFFIX",
    "TraitCorpus",
    "TraitPairSpec",
    "decode_trait_corpus",
    "decode_trait_pair_spec",
    "encode_trait_corpus",
    "encode_trait_pair_spec",
    "trait_corpus_digest",
]
