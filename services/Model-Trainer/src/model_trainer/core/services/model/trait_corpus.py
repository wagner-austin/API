"""Turn authored trait pairs into the tensors an arm is actually scored on.

THE ORDER IS THE PAIRING, and that is why every step here is deterministic.
Two arms are compared item by item, so item ``n`` in one arm must be the same
text as item ``n`` in the other -- across arms, across seeds, across a resume
on another node, and across a rerun a month later. Sorting the trait files by
name and keeping each file's pairs in authored order is what makes that true
without anything having to remember it.

WHAT A TRAIT'S TRAINING TEXT IS. The expressing member of the pairs the
cartridge is NOT scored on. Nothing else would answer the question: training
on the neutral members would teach the absence of the trait, and training on
the held-out pairs would measure memorisation, which is the mistake
:mod:`~model_trainer.core.services.model.cartridge_scoring` was split out to
avoid on the corpus side.

WHY A MEMBER IS REFUSED RATHER THAN TRUNCATED when it does not fit the budget.
A trait is carried by particular tokens -- a bullet marker, a capitalised run,
a closing question mark -- and truncation removes them from the end first. A
truncated pair still scores, still produces a plausible number, and measures
the trait's absence; there is no reading of the record that recovers it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import torch
from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)
from platform_core.json_utils import load_json_str

from model_trainer.core.contracts.trait_corpus import (
    TRAIT_CORPUS_SUFFIX,
    TraitCorpus,
    decode_trait_corpus,
)
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_scoring import TraitPair

#: One pair, whatever stage it is at. The split rule reads no field of it, so
#: the same function serves the authored form and the tokenised one.
_PairT = TypeVar("_PairT")


def _refuse(message: str) -> AppError[ModelTrainerErrorCode]:
    """Build the error a trait corpus that cannot be measured raises.

    Args:
        message: What is wrong, phrased for the person who authored the pairs.

    Returns:
        The error to raise.
    """
    return AppError(
        ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE,
        message,
        model_trainer_status_for(ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE),
    )


def trait_corpus_paths(directory: Path, suffix: str) -> list[Path]:
    """Name every trait file in a staged directory, in a fixed order.

    Args:
        directory: The staged directory.
        suffix: Extension trait files carry.

    Returns:
        The paths, sorted by name so the trait order is a property of the
        corpus rather than of the filesystem that happened to list it.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if the directory holds none.
    """
    paths = sorted(directory.glob(f"*{suffix}"))
    if not paths:
        raise _refuse(
            f"{directory} holds no {suffix} trait file; a trait measurement needs at "
            f"least the trait whose expression is the finding, and a composed one needs "
            f"a file per compartment"
        )
    return paths


def require_named_trait(corpus: TraitCorpus, path: Path) -> TraitCorpus:
    """Refuse a trait file whose declared trait is not the one it is filed as.

    A FILENAME IS NOT CARRIED INTO A RECORD, and the declared trait is: it
    names every arm and appears in the label. So a file staged as one trait
    and declaring another would produce a complete record whose arms are
    labelled with a trait the numbers did not come from, and the two spellings
    are exactly the kind of thing a copy gets wrong.

    Args:
        corpus: The decoded corpus.
        path: The file it was read from.

    Returns:
        The corpus, unchanged.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if the two disagree.
    """
    if corpus["trait"] != path.stem:
        raise _refuse(
            f"{path} declares trait {corpus['trait']!r} but is filed as {path.stem!r}; "
            f"the declared trait names every arm in the record and the filename names "
            f"what a run selects, so a record written now would label its numbers with a "
            f"trait they did not come from"
        )
    return corpus


def load_trait_corpora(directory: Path, traits: Sequence[str]) -> list[TraitCorpus]:
    """Read the staged trait files a plan's roster names, in ROSTER order.

    THE ROSTER ORDER, NOT THE DIRECTORY'S. The first trait is the one whose
    expression is the finding and the rest are composed in front of it, so the
    order is part of the measurement and the plan owns it. Sorting the
    directory decides which files EXIST; the roster decides which are read and
    in what sequence, and the two are different questions.

    Args:
        directory: The staged directory.
        traits: The plan's roster, in order.

    Returns:
        One corpus per trait, in roster order.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if the directory holds no
            trait files, if a named trait has no file, or if any file fails
            its own validation.
        InvalidJsonError: If a trait file is not JSON at all.
    """
    available = {path.stem: path for path in trait_corpus_paths(directory, TRAIT_CORPUS_SUFFIX)}
    missing = [trait for trait in traits if trait not in available]
    if missing:
        raise _refuse(
            f"the roster names {len(missing)} trait(s) with no file in {directory} -- "
            f"{', '.join(missing)}; the directory holds {', '.join(sorted(available))}. A "
            f"run over the traits that happen to be present would compose a shorter "
            f"roster than its own label claims"
        )
    return [
        require_named_trait(
            decode_trait_corpus(load_json_str(available[trait].read_text(encoding="utf-8"))),
            available[trait],
        )
        for trait in traits
    ]


def _encode_member(
    tokenizer: HFTokenizerProto,
    text: str,
    member: str,
    *,
    trait: str,
    max_seq_len: int,
    device: str,
) -> torch.Tensor:
    """Encode one member of one pair into the tensor it is scored as.

    THE TEXT IS PASSED IN RATHER THAN SELECTED HERE by a member name. A
    ``TraitPairSpec`` is a TypedDict, so indexing it with a variable is not a
    typed read at all -- the checker cannot know which field it names, and the
    only ways to write it are a cast or a widening. Selecting at the two call
    sites keeps both reads literal and leaves this function with one job.

    Args:
        tokenizer: Tokenizer of the base being measured.
        text: The prompt and this member's continuation, already joined.
        member: Which member this is, for the refusal message only.
        trait: The trait, for the refusal message.
        max_seq_len: Token budget a member may not exceed.
        device: Torch device string to build the tensor on.

    Returns:
        Token ids shaped ``(1, tokens)``.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if the member encodes to
            nothing, or exceeds the budget.
    """
    ids = tokenizer.encode(text)
    if not ids:
        raise _refuse(
            f"the {member} member of a {trait!r} pair encodes to no tokens, so it has no "
            f"loss; the text is non-empty, which means this tokenizer drops it"
        )
    if len(ids) > max_seq_len:
        raise _refuse(
            f"the {member} member of a {trait!r} pair encodes to {len(ids)} tokens against "
            f"a budget of {max_seq_len}; it is refused rather than truncated, because a "
            f"trait is carried by particular tokens and truncation removes the end first, "
            f"producing a plausible number that measures the trait's absence"
        )
    # Allocated and filled rather than built from a list literal, for the
    # reason `build_windows` gives: `torch.tensor([...])` is typed as
    # returning Any, which would put an unchecked value into every caller.
    row = torch.empty((1, len(ids)), dtype=torch.long, device=device)
    for offset, token in enumerate(ids):
        row[0, offset] = int(token)
    return row


def tokenise_trait_pairs(
    corpus: TraitCorpus,
    tokenizer: HFTokenizerProto,
    *,
    max_seq_len: int,
    device: str,
) -> list[TraitPair]:
    """Encode every pair of one trait, both members, in authored order.

    Both members carry the SAME prompt, so nothing the prompt contributes can
    reach the difference of differences. It is prepended to each member rather
    than scored separately because a loss over a continuation alone would be a
    loss over text with no context, which is not what either arm sees.

    Args:
        corpus: The trait's authored pairs.
        tokenizer: Tokenizer of the base being measured.
        max_seq_len: Token budget a member may not exceed.
        device: Torch device string to build the tensors on.

    Returns:
        The pairs, in authored order.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if any member encodes to
            nothing or exceeds the budget.
    """
    return [
        TraitPair(
            expressing=_encode_member(
                tokenizer,
                spec["prompt"] + spec["expressing"],
                "expressing",
                trait=corpus["trait"],
                max_seq_len=max_seq_len,
                device=device,
            ),
            neutral=_encode_member(
                tokenizer,
                spec["prompt"] + spec["neutral"],
                "neutral",
                trait=corpus["trait"],
                max_seq_len=max_seq_len,
                device=device,
            ),
        )
        for spec in corpus["pairs"]
    ]


def split_trait_pairs(
    pairs: Sequence[_PairT], *, held_out_stride: int
) -> tuple[list[_PairT], list[_PairT]]:
    """Hold out every ``held_out_stride``th pair, train on the rest.

    GENERIC OVER WHAT A PAIR IS, because the rule is defined on POSITION and
    nothing here reads a field. The run splits tokenised pairs; the plan
    table's suite splits AUTHORED ones, to check that a committed corpus can
    still resolve the effect its plan declares without tokenising anything.
    Those are the same split, and a second copy taking the other type would be
    two definitions of which pairs are scored.

    THE SAME RULE AS
    :func:`~model_trainer.core.services.model.cartridge_corpus.split_by_stride`
    AND DELIBERATELY NOT THE SAME FUNCTION. That one splits windows of a text
    corpus and its refusals are about windowing -- a stride that holds out
    every window, a corpus too short to yield any. The refusal that matters
    here is a different and much sharper one: the held-out side is what the
    power gate reads, and a set too small to resolve the effect the plan
    declares is refused by
    :func:`~model_trainer.core.services.model.cartridge_qa_power.require_resolvable_pairs`
    with a count of how many pairs it would take. Folding this into the window
    splitter would mean either importing that reasoning into a module about
    text windows or losing it.

    Args:
        pairs: Every pair of one trait, in authored order.
        held_out_stride: Take one pair in this many for the held-out set.

    Returns:
        ``(train, held_out)``, each in authored order.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if the stride leaves either
            side empty. Both are checked because they read differently: no
            training pairs means the cartridge learned nothing, and no
            held-out pairs means it was scored on nothing.
    """
    if held_out_stride < 2:
        raise _refuse(
            f"a held-out stride of {held_out_stride} holds out every pair and leaves "
            f"nothing to train the trait cartridge on; the smallest stride that trains "
            f"anything is 2"
        )
    train = [pair for index, pair in enumerate(pairs) if index % held_out_stride != 0]
    held_out = [pair for index, pair in enumerate(pairs) if index % held_out_stride == 0]
    if not train:
        raise _refuse(
            f"{len(pairs)} pair(s) at a stride of {held_out_stride} leave no training "
            f"pairs, so there would be no trait cartridge to score"
        )
    return train, held_out


def training_items(pairs: Sequence[TraitPair]) -> list[torch.Tensor]:
    """Take the text a trait cartridge trains on.

    THE EXPRESSING MEMBER AND ONLY IT. Training on both members would teach
    the cartridge the pair rather than the trait, and training on the neutral
    member would teach its absence. This is the one place that choice is made,
    so no arm can make it differently.

    Args:
        pairs: The training pairs.

    Returns:
        One tensor per pair, in order.
    """
    return [pair["expressing"] for pair in pairs]


__all__ = [
    "load_trait_corpora",
    "require_named_trait",
    "split_trait_pairs",
    "tokenise_trait_pairs",
    "training_items",
    "trait_corpus_paths",
]
