"""Reading, tokenising and splitting a staged trait corpus.

WHAT THESE TESTS ARE ABOUT, and it is one thing: the ORDER and the SPLIT are
the instrument. Two arms are compared item by item, so a reordering makes two
runs' per-item comparisons describe different texts under the same indices,
and a split that leaked training pairs into the scored set would measure
memorisation while reporting trait expression. Neither failure raises.

The default reader is exercised against REAL FILES in a temporary directory
rather than behind a fake. A fake in front of a filesystem reader tests the
fake, and the things worth checking here -- that the roster orders the result,
that a mis-filed trait is caught -- are exactly the things a fake would be
written to satisfy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import dump_json_str

from model_trainer.core.contracts.trait_corpus import (
    TRAIT_CORPUS_SUFFIX,
    TraitCorpus,
    TraitPairSpec,
    encode_trait_corpus,
)
from model_trainer.core.services.model.cartridge_scoring import TraitPair
from model_trainer.core.services.model.trait_corpus import (
    load_trait_corpora,
    split_trait_pairs,
    tokenise_trait_pairs,
    training_items,
    trait_corpus_paths,
)


class _WordTokenizer:
    """A tokenizer that maps each whitespace token to a stable id.

    Real rather than a stand-in for one: it encodes, it decodes, and different
    text produces different ids, which is everything the code under test asks
    of a tokenizer. A fake returning a fixed list would make the length
    refusals below untestable, since they are about what the text encodes TO.
    """

    def __init__(self) -> None:
        """Start with an empty vocabulary, grown on demand."""
        self._ids: dict[str, int] = {}

    @property
    def eos_token_id(self) -> int | None:
        """This tokenizer has no end-of-sequence token."""
        return None

    @property
    def pad_token_id(self) -> int | None:
        """This tokenizer has no padding token."""
        return None

    def __len__(self) -> int:
        """Report the vocabulary size.

        Returns:
            How many distinct tokens have been seen.
        """
        return len(self._ids)

    def encode(self, text: str) -> list[int]:
        """Map each whitespace-separated token to a stable id.

        Args:
            text: The text to encode.

        Returns:
            One id per token, in order.
        """
        return [self._ids.setdefault(token, len(self._ids) + 1) for token in text.split()]

    def decode(self, ids: list[int]) -> str:
        """Map ids back to their tokens.

        Args:
            ids: Ids to decode.

        Returns:
            The tokens, space separated.
        """
        back = {value: key for key, value in self._ids.items()}
        return " ".join(back[identifier] for identifier in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        """Map one token to its id, assigning one if it is new.

        Args:
            token: The token.

        Returns:
            Its id.
        """
        return self._ids.setdefault(token, len(self._ids) + 1)


def _write(directory: Path, trait: str, pairs: int = 4) -> Path:
    """Write one trait file with a given number of pairs.

    Args:
        directory: Where to write it.
        trait: The trait to declare, which is also the filename.
        pairs: How many pairs to author.

    Returns:
        The path written.
    """
    corpus = TraitCorpus(
        trait=trait,
        pairs=[
            TraitPairSpec(
                prompt=f"prompt {index} ",
                expressing=f"expressing {trait} {index}",
                neutral=f"neutral {index}",
            )
            for index in range(pairs)
        ],
    )
    path = directory / f"{trait}{TRAIT_CORPUS_SUFFIX}"
    path.write_text(dump_json_str(encode_trait_corpus(corpus)), encoding="utf-8")
    return path


def _ids(row: torch.Tensor) -> list[int]:
    """Read one encoded member's token ids back out.

    Indexed by position rather than iterated, because iterating a tensor
    yields values this package's strictness will not accept unchecked.

    Args:
        row: Token ids shaped ``(1, tokens)``.

    Returns:
        The ids, in order.
    """
    return [int(row[0, index].item()) for index in range(int(row.shape[1]))]


def _pair(length: int) -> TraitPair:
    """Build one already-tokenised pair of a given length.

    Args:
        length: Tokens per member.

    Returns:
        The pair.
    """
    return TraitPair(
        expressing=torch.ones((1, length), dtype=torch.long),
        neutral=torch.zeros((1, length), dtype=torch.long),
    )


class TestTheRosterDecidesWhatIsRead:
    """Which files exist and which are read are different questions."""

    def test_corpora_come_back_in_roster_order_not_filename_order(self, tmp_path: Path) -> None:
        """The first trait is the finding, so the plan owns the order.

        Written in one order and requested in another, because sorting the
        directory is the failure this exists to rule out -- and alphabetical
        order would silently be right if the roster happened to be sorted.

        Args:
            tmp_path: Temporary directory.
        """
        _write(tmp_path, "bullets")
        _write(tmp_path, "formal-tone")
        corpora = load_trait_corpora(tmp_path, ["formal-tone", "bullets"])
        assert [corpus["trait"] for corpus in corpora] == ["formal-tone", "bullets"]

    def test_a_roster_naming_an_absent_trait_is_refused(self, tmp_path: Path) -> None:
        """Running the traits that happen to be present shortens the roster.

        The label would still claim the full one, so the record would name a
        composition that never happened.

        Args:
            tmp_path: Temporary directory.
        """
        _write(tmp_path, "bullets")
        with pytest.raises(AppError) as excinfo:
            load_trait_corpora(tmp_path, ["bullets", "formal-tone"])
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE
        assert "formal-tone" in excinfo.value.message

    def test_an_empty_directory_is_refused(self, tmp_path: Path) -> None:
        """Nothing to measure is a corpus failure, surfaced before any model.

        Args:
            tmp_path: Temporary directory.
        """
        with pytest.raises(AppError) as excinfo:
            trait_corpus_paths(tmp_path, TRAIT_CORPUS_SUFFIX)
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE

    def test_a_file_whose_declared_trait_is_not_its_name_is_refused(self, tmp_path: Path) -> None:
        """A filename is not carried into a record; the declared trait is.

        A file staged as one trait and declaring another produces a complete
        record whose arms are labelled with a trait the numbers did not come
        from.

        Args:
            tmp_path: Temporary directory.
        """
        path = _write(tmp_path, "bullets")
        path.rename(tmp_path / f"formal-tone{TRAIT_CORPUS_SUFFIX}")
        with pytest.raises(AppError) as excinfo:
            load_trait_corpora(tmp_path, ["formal-tone"])
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE
        assert "filed as" in excinfo.value.message


class TestTokenisationRefusesRatherThanTruncates:
    """A truncated pair still scores, and measures the trait's absence."""

    def test_both_members_carry_the_prompt(self) -> None:
        """Nothing the prompt contributes can reach the difference.

        Asserted by decoding: the prompt's tokens must be the leading tokens
        of both members, which is what makes the shared context shared.
        """
        tokenizer = _WordTokenizer()
        corpus = TraitCorpus(
            trait="bullets",
            pairs=[TraitPairSpec(prompt="shared start ", expressing="alpha", neutral="beta")],
        )
        pairs = tokenise_trait_pairs(corpus, tokenizer, max_seq_len=16, device="cpu")
        assert tokenizer.decode(_ids(pairs[0]["expressing"])) == "shared start alpha"
        assert tokenizer.decode(_ids(pairs[0]["neutral"])) == "shared start beta"

    def test_a_member_over_the_budget_is_refused(self) -> None:
        """Truncation removes the end first, where the trait usually lives."""
        tokenizer = _WordTokenizer()
        corpus = TraitCorpus(
            trait="bullets",
            pairs=[TraitPairSpec(prompt="a ", expressing="b c d e f", neutral="g")],
        )
        with pytest.raises(AppError) as excinfo:
            tokenise_trait_pairs(corpus, tokenizer, max_seq_len=3, device="cpu")
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE
        assert "refused rather than truncated" in excinfo.value.message

    def test_text_that_encodes_to_nothing_is_refused(self) -> None:
        """Non-empty text that this tokenizer drops has no loss to compare.

        The contract already refuses blank text, so reaching this arm means
        the TOKENIZER dropped something the author wrote -- a different fault
        with a different remedy, and one a record could not show.
        """
        tokenizer = _WordTokenizer()
        corpus = TraitCorpus(
            trait="bullets",
            pairs=[TraitPairSpec(prompt="\t", expressing="\t", neutral="\tx")],
        )
        with pytest.raises(AppError) as excinfo:
            tokenise_trait_pairs(corpus, tokenizer, max_seq_len=8, device="cpu")
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE
        assert "encodes to no tokens" in excinfo.value.message


class TestTheSplitKeepsTheScoredPairsOutOfTraining:
    """Training on the pairs an arm is scored on measures memorisation."""

    def test_every_pair_lands_on_exactly_one_side(self) -> None:
        """No pair may be both trained on and scored, and none may be lost."""
        pairs = [_pair(2) for _ in range(6)]
        train, held_out = split_trait_pairs(pairs, held_out_stride=2)
        assert len(train) + len(held_out) == len(pairs)
        assert all(any(item is pair for item in train + held_out) for pair in pairs)
        assert not [pair for pair in train if any(pair is item for item in held_out)]

    def test_a_stride_of_two_holds_out_half(self) -> None:
        """Half is the largest held-out set a stride can give.

        Which is also the lowest resolvable floor a corpus of this size can
        reach, so the plans declare their effect against it.
        """
        _train, held_out = split_trait_pairs([_pair(2) for _ in range(8)], held_out_stride=2)
        assert len(held_out) == 4

    def test_a_stride_of_one_is_refused(self) -> None:
        """Holding out every pair leaves no cartridge to score."""
        with pytest.raises(AppError) as excinfo:
            split_trait_pairs([_pair(2) for _ in range(4)], held_out_stride=1)
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE
        assert "smallest stride that trains anything is 2" in excinfo.value.message

    def test_too_few_pairs_to_train_on_is_refused(self) -> None:
        """One pair at stride two is held out and nothing is trained."""
        with pytest.raises(AppError) as excinfo:
            split_trait_pairs([_pair(2)], held_out_stride=2)
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE
        assert "no training" in excinfo.value.message

    def test_training_takes_the_expressing_member_only(self) -> None:
        """Training on the neutral member would teach the trait's ABSENCE.

        One place makes that choice, so no arm can make it differently -- and
        the assertion is on the tensors themselves rather than on a count,
        because a count cannot tell the two members apart.
        """
        pairs = [_pair(3) for _ in range(2)]
        items = training_items(pairs)
        assert all(bool(torch.all(item == 1)) for item in items)
