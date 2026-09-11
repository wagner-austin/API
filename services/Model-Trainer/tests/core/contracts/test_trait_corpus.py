"""What an authored trait corpus must be before anything measures with it.

EVERY REFUSAL HERE IS A NUMBER THAT WOULD OTHERWISE BE PRODUCED. That is the
property worth stating once: none of these failures raise on their own. A pair
whose members are identical scores exactly zero and enters a record as a tie; a
blank member scores a loss over no tokens; a trait outside the admissible set
produces a perfectly clean null that measures trait choice rather than
composition. All three arrive as complete, plausible tables, which is why the
check has to be at the decode and not in a reviewer's head.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from model_trainer.core.contracts.trait_corpus import (
    CONSISTENTLY_EFFECTIVE_TRAITS,
    TraitCorpus,
    TraitPairSpec,
    decode_trait_corpus,
    decode_trait_pair_spec,
    encode_trait_corpus,
    encode_trait_pair_spec,
    trait_corpus_digest,
)


def _pair(expressing: str = "- one\n- two", neutral: str = "one and two") -> JSONValue:
    """Build one authored pair as the JSON loader would hand it over.

    Args:
        expressing: The trait-expressing continuation.
        neutral: The continuation that does not express it.

    Returns:
        The pair, as a JSON object.
    """
    return {"prompt": "The steps are: ", "expressing": expressing, "neutral": neutral}


def _corpus(trait: str = "bullets") -> JSONValue:
    """Build one trait's whole authored corpus.

    Args:
        trait: The trait to declare.

    Returns:
        The corpus, as a JSON object.
    """
    return {"trait": trait, "pairs": [_pair()]}


class TestAPairMustBeAbleToMeasureSomething:
    """The pair is the instrument; a degenerate one scores zero and says so."""

    def test_a_valid_pair_decodes_to_its_three_fields(self) -> None:
        """The ordinary case, so the refusals below are not the only coverage."""
        spec = decode_trait_pair_spec(_pair())
        assert spec == TraitPairSpec(
            prompt="The steps are: ", expressing="- one\n- two", neutral="one and two"
        )

    def test_identical_continuations_are_refused(self) -> None:
        """A pair whose members match scores zero BY CONSTRUCTION.

        It would be recorded as a tie, which is indistinguishable from a trait
        the cartridge genuinely failed to express -- so nothing downstream can
        tell the authoring mistake from the result.
        """
        with pytest.raises(AppError) as excinfo:
            decode_trait_pair_spec(_pair(expressing="same text", neutral="same text"))
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE
        assert "identical" in excinfo.value.message

    @pytest.mark.parametrize("field", ["prompt", "expressing", "neutral"])
    def test_a_blank_field_is_refused(self, field: str) -> None:
        """Whitespace is not text: the loss would be over no tokens.

        Args:
            field: Which field to blank.
        """
        raw: JSONValue = {"prompt": "p", "expressing": "e", "neutral": "n", field: "   "}
        with pytest.raises(AppError) as excinfo:
            decode_trait_pair_spec(raw)
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE
        assert field in excinfo.value.message

    def test_a_non_object_pair_is_refused_by_type(self) -> None:
        """A list where an object belongs is a malformed file, not a bad pair."""
        with pytest.raises(JSONTypeError):
            decode_trait_pair_spec(["prompt", "expressing"])

    def test_encoding_a_pair_round_trips(self) -> None:
        """What is written must decode to what was held."""
        spec = decode_trait_pair_spec(_pair())
        assert decode_trait_pair_spec(encode_trait_pair_spec(spec)) == spec


class TestATraitMustBeOneTheResultCouldBeReadFrom:
    """The admissible set is a measured constraint, not a style preference."""

    def test_a_valid_corpus_decodes(self) -> None:
        """The ordinary case."""
        corpus = decode_trait_corpus(_corpus())
        assert corpus["trait"] == "bullets"
        assert len(corpus["pairs"]) == 1

    @pytest.mark.parametrize("trait", sorted(CONSISTENTLY_EFFECTIVE_TRAITS))
    def test_every_admissible_trait_is_accepted(self, trait: str) -> None:
        """The set is walked rather than sampled, so none rots unnoticed.

        Args:
            trait: One admissible trait.
        """
        assert decode_trait_corpus(_corpus(trait))["trait"] == trait

    def test_a_positionally_concentrated_trait_is_refused(self) -> None:
        """Rhyme matters at line ends, so a null from it measures the wrong thing.

        This is the refusal that keeps the admissible set from being an
        opinion: a composition null drawn from a trait that is hard to express
        at all says nothing about composition.
        """
        with pytest.raises(AppError) as excinfo:
            decode_trait_corpus(_corpus("rhyming-structure"))
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE
        assert "positionally concentrated" in excinfo.value.message

    def test_a_trait_with_no_pairs_is_refused(self) -> None:
        """Zero items reports a p-value of one, which reads as a null result."""
        with pytest.raises(AppError) as excinfo:
            decode_trait_corpus({"trait": "bullets", "pairs": []})
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE
        assert "no pairs" in excinfo.value.message

    def test_a_non_object_corpus_is_refused_by_type(self) -> None:
        """A bare list is a file that is not a trait corpus at all."""
        with pytest.raises(JSONTypeError):
            decode_trait_corpus([])

    def test_encoding_a_corpus_round_trips(self) -> None:
        """What is written must decode to what was held, pairs in order."""
        corpus = decode_trait_corpus(_corpus())
        assert decode_trait_corpus(encode_trait_corpus(corpus)) == corpus


class TestTheDigestSeparatesWhatIsActuallyDifferent:
    """Two corpora that measure different things must not share a digest."""

    def test_the_same_corpora_digest_identically(self) -> None:
        """Reproducibility: the digest is what pairs two runs."""
        first = decode_trait_corpus(_corpus())
        second = decode_trait_corpus(_corpus())
        assert trait_corpus_digest([first]) == trait_corpus_digest([second])

    def test_roster_order_changes_the_digest(self) -> None:
        """Order IS the measurement: the first trait is the finding.

        A rotated roster composes different cartridges in front of a different
        primary, so it must not be able to register under the other's name.
        """
        bullets = decode_trait_corpus(_corpus("bullets"))
        formal = decode_trait_corpus(_corpus("formal-tone"))
        assert trait_corpus_digest([bullets, formal]) != trait_corpus_digest([formal, bullets])

    def test_moving_a_boundary_changes_the_digest(self) -> None:
        """Concatenation alone would let two different splits collide.

        The prompt boundary decides what both members share and the member
        boundary decides what is differenced, so the same characters split
        elsewhere are a different instrument.
        """
        left = TraitCorpus(
            trait="bullets",
            pairs=[TraitPairSpec(prompt="ab", expressing="cd", neutral="ef")],
        )
        right = TraitCorpus(
            trait="bullets",
            pairs=[TraitPairSpec(prompt="a", expressing="bcd", neutral="ef")],
        )
        assert trait_corpus_digest([left]) != trait_corpus_digest([right])
