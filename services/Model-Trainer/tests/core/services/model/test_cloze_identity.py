"""The digest that tells two question sets apart.

NOTHING IS FAKED HERE. The subject is a pure function of literal items, so a
fake would only stand between the assertion and the arithmetic it is about.

Each test names the record pair or the run pair it would have caught, because
this module exists for a defect that was found in written records rather than
in a failing test.
"""

from __future__ import annotations

from model_trainer.core.contracts.cloze import ClozeItem
from model_trainer.core.services.model.cloze.identity import (
    QUESTION_SET_DIGEST_PREFIX,
    question_set_digest,
)


def _item(item_id: str, *, template: str, answer: str, distractors: list[str]) -> ClozeItem:
    """Build one item.

    Args:
        item_id: Its identifier.
        template: The sentence with the blank marker in it.
        answer: The true term.
        distractors: The wrong candidates offered beside it.

    Returns:
        The item.
    """
    return ClozeItem(item_id=item_id, template=template, answer=answer, distractors=distractors)


_BASE: tuple[ClozeItem, ...] = (
    _item(
        "d000-i000-NavProbe",
        template="The team measured ___ against the usual baseline.",
        answer="NavProbe",
        distractors=["ClearGBM", "CoverGate"],
    ),
    _item(
        "d001-i004-ClearGBM",
        template="A later pass moved ___ onto a faster route.",
        answer="ClearGBM",
        distractors=["NavProbe", "TankpitBot"],
    ),
)


class TestQuestionSetDigest:
    def test_the_same_items_digest_the_same_way(self) -> None:
        """Otherwise nothing downstream could ever conclude two runs agree."""
        assert question_set_digest(_BASE) == question_set_digest(_BASE)

    def test_it_names_its_algorithm(self) -> None:
        digest = question_set_digest(_BASE)

        assert digest.startswith(QUESTION_SET_DIGEST_PREFIX)
        assert len(digest) == len(QUESTION_SET_DIGEST_PREFIX) + 64

    def test_a_larger_item_set_digests_differently(self) -> None:
        """THE RECORD PAIR THIS EXISTS FOR, in miniature.

        `qa-record.json` asked 24 questions and `qa-svc-gpt2.json` asked 32
        from the same plan and the same corpus, and every identity field of
        the two records was equal.
        """
        grown = (
            *_BASE,
            _item(
                "d002-i009-CoverGate",
                template="Written notes about ___ explain the design.",
                answer="CoverGate",
                distractors=["NavProbe", "ClearGBM"],
            ),
        )

        assert question_set_digest(grown) != question_set_digest(_BASE)

    def test_changing_only_the_distractors_changes_the_digest(self) -> None:
        """THE FAILURE AN ID-ONLY DIGEST WOULD HAVE MISSED.

        Rotating distractors per item moved the base model from 0.2500 to
        0.5417 on the real corpus while every `item_id` stayed the same.
        """
        rotated = (
            _BASE[0],
            _item(
                _BASE[1]["item_id"],
                template=_BASE[1]["template"],
                answer=_BASE[1]["answer"],
                distractors=["CoverGate", "NavProbe"],
            ),
        )

        assert question_set_digest(rotated) != question_set_digest(_BASE)

    def test_changing_only_the_sentence_changes_the_digest(self) -> None:
        """A term can occur in several sentences, and which one was blanked
        decides how answerable the item is.
        """
        reworded = (
            _item(
                _BASE[0]["item_id"],
                template="Written notes about ___ explain the design.",
                answer=_BASE[0]["answer"],
                distractors=list(_BASE[0]["distractors"]),
            ),
            _BASE[1],
        )

        assert question_set_digest(reworded) != question_set_digest(_BASE)

    def test_reordering_the_same_items_changes_the_digest(self) -> None:
        """Order is content here: the items are built by one deterministic
        pass, so a different order means a different pass ran.
        """
        assert question_set_digest((_BASE[1], _BASE[0])) != question_set_digest(_BASE)

    def test_an_empty_set_digests_rather_than_returning_nothing(self) -> None:
        """A run that asked no questions is a fact, not a missing digest.

        Returning the empty string would collide with
        :data:`~platform_core.run_record.NO_PAYLOAD`, which means "this run
        emitted no payload worth hashing" -- the opposite claim.
        """
        empty = question_set_digest(())

        assert empty.startswith(QUESTION_SET_DIGEST_PREFIX)
        assert empty != question_set_digest(_BASE)
