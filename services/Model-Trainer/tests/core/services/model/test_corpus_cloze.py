"""Building questions out of a corpus without inventing anything.

THE TWO DEFECTS THIS FILE PINS were both found by running the generator and
reading what it produced, not by reasoning about it:

  - accepting any capitalised word made items whose answer was "Plain" against
    "Getting" and "Produced", which any English model answers without having
    read the corpus;
  - drawing distractors from a fixed position made a 24-item set share one
    distractor triple, and the whole measurement then turned on how those
    three words compared to each answer.

Both are quiet failures: the items look like questions either way, and only
the numbers downstream go wrong. So both are asserted here rather than left to
review.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.cloze import BLANK_MARKER, decode_cloze_item
from model_trainer.core.services.model.corpus_cloze import (
    MAX_SENTENCE_CHARS,
    MIN_SENTENCE_CHARS,
    build_items,
    sentences,
    terms_in,
)


class TestSentences:
    def test_it_splits_on_terminators(self) -> None:
        assert sentences("One thing. Two things! Three things?") == [
            "One thing.",
            "Two things!",
            "Three things?",
        ]

    def test_a_sentence_broken_across_lines_is_one_sentence(self) -> None:
        """The corpus is markdown, so a line break is layout rather than syntax.

        An item carrying the original break would be scored on text that
        appears nowhere.
        """
        assert sentences("A sentence split\nacross two lines.") == [
            "A sentence split across two lines."
        ]

    def test_code_fences_are_removed(self) -> None:
        """Blanking a token out of a code block asks the model to recall syntax."""
        assert sentences("Before it. ```python\nx = ClearGBM()\n``` After it.") == [
            "Before it.",
            "After it.",
        ]

    def test_urls_are_removed(self) -> None:
        """A URL is corpus-specific tokens no model predicts from meaning.

        An item blanking part of one measures memorisation of a path, which is
        the failure the whole held-out split exists to avoid.
        """
        assert sentences("See <https://example.com/a/b> for it.") == ["See for it."]

    def test_table_rows_are_removed(self) -> None:
        assert sentences("Real prose here.\n| a | b |\n| - | - |\nMore prose.") == [
            "Real prose here.",
            "More prose.",
        ]


class TestTermsIn:
    def test_camel_case_is_a_name(self) -> None:
        assert terms_in("We ship ClearGBM and TankpitBot.") == {"ClearGBM", "TankpitBot"}

    def test_an_internal_digit_is_a_name(self) -> None:
        assert terms_in("Runs on hpc3 against gpt2 weights.") == {"hpc3", "gpt2"}

    def test_a_hyphenated_capital_is_a_name(self) -> None:
        assert terms_in("The Model-Trainer service.") == {"Model-Trainer"}

    def test_all_capitals_is_a_name(self) -> None:
        assert terms_in("Measured by OCR tooling.") == {"OCR"}

    def test_a_merely_capitalised_word_is_not(self) -> None:
        """The defect that made the first item set worthless.

        Sentence-initial words are capitalised in English, so accepting them
        selects ordinary vocabulary and the resulting item measures fluency.
        """
        assert terms_in("Plain version. Getting there. Produced output.") == set()

    def test_format_words_are_excluded(self) -> None:
        """A corpus about software mentions JSON the way any software text does."""
        assert terms_in("Encoded as JSON over HTTP with a UUID.") == set()


def _corpus() -> tuple[list[str], str]:
    """A held-out document and the training text its terms occur in."""
    held = [
        "The ClearGBM engine rebuilt boosting from scratch in a Rust core here. "
        "A second page mentions TankpitBot doing something else entirely today."
    ]
    training = (
        "ClearGBM is the gradient boosting engine. TankpitBot plays the game. "
        "NavProbe measures things. HPC3 runs the jobs. LightGBM is the baseline."
    )
    return held, training


class TestBuildItems:
    def test_every_item_is_a_valid_cloze_item(self) -> None:
        """Validated through the contract's own decoder, not by inspection.

        `decode_cloze_item` enforces exactly one blank marker, a non-empty
        answer, and distractors distinct from it -- the rules the scorer
        relies on.
        """
        held, training = _corpus()

        items = build_items(held, training, distractor_count=2, max_items=10)

        for item in items:
            restored = decode_cloze_item(
                {
                    "item_id": item["item_id"],
                    "template": item["template"],
                    "answer": item["answer"],
                    "distractors": list(item["distractors"]),
                }
            )
            assert restored["answer"] == item["answer"]

    def test_the_answer_is_removed_from_the_template(self) -> None:
        held, training = _corpus()

        items = build_items(held, training, distractor_count=2, max_items=10)

        for item in items:
            assert item["answer"] not in item["template"]
            assert item["template"].count(BLANK_MARKER) == 1

    def test_every_answer_occurs_in_the_training_text(self) -> None:
        """What makes an item answerable from the corpus at all.

        A term appearing only in held-out text could not have been learned by
        the cartridge, so the item would measure the base model's guessing.
        """
        held, training = _corpus()

        items = build_items(held, training, distractor_count=2, max_items=10)

        assert items != []
        for item in items:
            assert item["answer"] in training

    def test_distractors_are_never_terms_from_the_item_s_own_document(self) -> None:
        """Otherwise a distractor is a plausible answer rather than a wrong one."""
        held, training = _corpus()

        items = build_items(held, training, distractor_count=2, max_items=10)

        own_terms = terms_in(held[0])
        for item in items:
            assert set(item["distractors"]).isdisjoint(own_terms)

    def test_distractors_vary_across_items(self) -> None:
        """The defect that inverted a measurement.

        With one distractor triple shared by every item, the base model scored
        exactly chance and the cartridge looked significant at p = 0.006;
        rotating them moved the base to 0.5417 and the effect vanished. A set
        whose items all offer the same wrong answers is measuring those three
        words.
        """
        held = [
            "The ClearGBM engine rebuilt boosting from scratch in a Rust core here. "
            "The TankpitBot client plays the game over a decoded wire protocol now."
        ]
        training = (
            "ClearGBM is the engine. TankpitBot is the client. NavProbe measures. "
            "HPC3 schedules. LightGBM compares. CoverageGate checks. MyPy types."
        )

        items = build_items(held, training, distractor_count=2, max_items=10)

        rendered = {tuple(item["distractors"]) for item in items}
        assert len(rendered) > 1

    def test_every_item_gets_the_number_of_distractors_asked_for(self) -> None:
        """A short item would have a different chance baseline from its neighbours."""
        held, training = _corpus()

        items = build_items(held, training, distractor_count=3, max_items=10)

        for item in items:
            assert len(item["distractors"]) == 3
            assert len(set(item["distractors"])) == 3

    def test_max_items_bounds_the_set(self) -> None:
        held, training = _corpus()

        assert len(build_items(held, training, distractor_count=2, max_items=1)) == 1

    def test_a_sentence_below_the_floor_is_skipped(self) -> None:
        """Too little context, and the item becomes about English word frequency."""
        short = "ClearGBM ran."
        assert len(short) < MIN_SENTENCE_CHARS

        with pytest.raises(AppError) as excinfo:
            build_items(
                [short], "ClearGBM TankpitBot NavProbe HPC3", distractor_count=2, max_items=10
            )

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE

    def test_a_sentence_above_the_ceiling_is_skipped(self) -> None:
        """A very long one crowds the retrieval arm's context window."""
        long_sentence = "ClearGBM " + ("padding words here " * 40) + "end."
        assert len(long_sentence) > MAX_SENTENCE_CHARS

        with pytest.raises(AppError):
            build_items(
                [long_sentence],
                "ClearGBM TankpitBot NavProbe HPC3",
                distractor_count=2,
                max_items=10,
            )

    def test_a_term_appearing_twice_in_one_sentence_is_skipped(self) -> None:
        """Blanking one leaves the other visible beside it, answering the item."""
        held = [
            "The ClearGBM engine and the ClearGBM core were both rebuilt here today "
            "by the same author over one long weekend of sustained work."
        ]
        training = "ClearGBM TankpitBot NavProbe HPC3 LightGBM"

        with pytest.raises(AppError):
            build_items(held, training, distractor_count=2, max_items=10)

    def test_a_sentence_whose_page_owns_every_other_term_is_skipped(self) -> None:
        """Distractors come from OTHER pages, and a page may leave too few.

        The corpus has three terms in total and all of them appear on the one
        held-out page, so an item from it can draw no distractor that is not
        one of its own page's subjects. Skipping is right: a distractor the
        page is also about is a plausible answer rather than a wrong one, and
        an item with fewer candidates has a different chance baseline from its
        neighbours.
        """
        held = [
            "The ClearGBM engine and the TankpitBot client and the NavProbe rig "
            "were all rebuilt during one long and rather sustained working week."
        ]
        training = "ClearGBM TankpitBot NavProbe"

        with pytest.raises(AppError) as excinfo:
            build_items(held, training, distractor_count=2, max_items=10)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE

    def test_too_few_terms_to_draw_distractors_is_refused(self) -> None:
        """Wrong candidates that are not corpus terms give the item away."""
        with pytest.raises(AppError) as excinfo:
            build_items(["ClearGBM is here."], "ClearGBM", distractor_count=5, max_items=10)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE
        assert "distractor" in excinfo.value.message

    def test_a_document_with_no_learnable_term_is_refused(self) -> None:
        held = [
            "The NavProbe measurement rig recorded nothing at all that afternoon, "
            "which was itself the finding worth writing down for later."
        ]
        training = "ClearGBM TankpitBot HPC3 LightGBM CoverageGate"

        with pytest.raises(AppError) as excinfo:
            build_items(held, training, distractor_count=2, max_items=10)

        assert "unanswerable from the corpus" in excinfo.value.message

    def test_item_ids_are_unique_even_when_a_term_recurs(self) -> None:
        """A document usually names its own subject in several sentences.

        Arms are paired BY ID, so two items sharing one would collapse in the
        lookup and pair an arm's outcome against a different question's --
        silently, and in a direction nobody could predict. The id therefore
        carries the item's position as well as its document and term.
        """
        held = [
            "The ClearGBM engine rebuilt boosting from scratch inside a Rust core. "
            "A later pass moved ClearGBM onto a histogram path for speed today."
        ]
        training = "ClearGBM TankpitBot NavProbe HPC3 LightGBM CoverageGate"

        items = build_items(held, training, distractor_count=2, max_items=10)

        answers = [item["answer"] for item in items]
        identifiers = [item["item_id"] for item in items]
        assert answers.count("ClearGBM") > 1
        assert len(identifiers) == len(set(identifiers))

    def test_items_are_deterministic(self) -> None:
        """No sampling anywhere: the set is a function of the corpus."""
        held, training = _corpus()

        first = build_items(held, training, distractor_count=2, max_items=10)
        second = build_items(held, training, distractor_count=2, max_items=10)

        assert first == second
