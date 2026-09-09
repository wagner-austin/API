"""The reshaping rule, and the gate that proves it only deletes.

NOTHING IS FAKED. Both the rule and the gate are pure functions of literal
strings over the production sentence splitter and term extractor, and using
the real ones is the point: the whole claim of this reshaping is that it
produces exactly the prose the ITEM BUILDER already reads, so a test against a
stand-in splitter would be a test of a different transformation.

THE CENTRAL PROPERTY, and every clause exists to protect it: the rule can only
REMOVE. That is what separates this from the automatic triple extraction AKEW
measured in single digits, which had to create structure the prose did not
have. A test suite for a delete-only rule is a suite about what survived.
"""

from __future__ import annotations

from model_trainer.core.services.model.corpus_cloze import sentences
from model_trainer.core.services.model.corpus_reshape import (
    RESHAPE_CLAUSES,
    gate_reshape,
    reshape_corpus,
    reshape_document,
)

#: A document with one of each thing the splitter strips, plus real prose.
_MESSY = """## Twelve backends behind one interface

Seven classifiers sit behind one interface, XGBoost and ClearGBM among them.

| backend | language |
|---|---|
| ClearGBM | Rust |
| TankpitBot | Python |

The team measured NavProbe against the usual baseline over many weeks.

```python
DTypeLike = str
```

See <https://example.com/CoverGate/v2> for the rest.
"""


class TestReshapeDocument:
    def test_the_prose_survives(self) -> None:
        reshaped = reshape_document(_MESSY)

        assert "Seven classifiers sit behind one interface" in reshaped
        assert "The team measured NavProbe against the usual baseline" in reshaped

    def test_the_table_rows_do_not(self) -> None:
        """THE GHOST-TERM HABITAT. A term that lives only in a table row is a
        term the corpus cannot teach in a sentence, and the item builder
        already refuses to ask about it.
        """
        reshaped = reshape_document(_MESSY)

        assert "| Rust |" not in reshaped
        assert "TankpitBot" not in reshaped

    def test_the_fenced_code_does_not(self) -> None:
        assert "DTypeLike" not in reshape_document(_MESSY)

    def test_the_url_does_not(self) -> None:
        """An item blanking a token out of a URL measures memorisation of a
        path, which is why the splitter removes them and why a cartridge
        should not spend prefix capacity on them either.
        """
        assert "example.com" not in reshape_document(_MESSY)

    def test_a_document_of_pure_scaffolding_reshapes_to_nothing(self) -> None:
        """Reported by the gate rather than raised here: an empty result is a
        fact about the document.
        """
        assert reshape_document("| a | b |\n|---|---|\n| c | d |\n") == ""

    def test_reshaping_is_idempotent(self) -> None:
        """A second pass must be a no-op, or the rule is not a projection and
        'the prose the item builder reads' is not well defined.
        """
        once = reshape_document(_MESSY)

        assert reshape_document(once) == once


class TestGateReshape:
    def test_a_faithful_reshape_passes_every_clause(self) -> None:
        verdict = gate_reshape(_MESSY, reshape_document(_MESSY))

        assert verdict["failed_clauses"] == ()

    def test_it_reports_the_terms_the_reshape_cost(self) -> None:
        """The price of the manipulation, per document. A reader who cannot
        see which facts the reshaped corpus can no longer teach cannot judge
        a difference measured on it.
        """
        verdict = gate_reshape(_MESSY, reshape_document(_MESSY))

        assert "DTypeLike" in verdict["terms_lost"]
        assert "TankpitBot" in verdict["terms_lost"]
        assert "NavProbe" not in verdict["terms_lost"]

    def test_an_invented_sentence_is_caught(self) -> None:
        """THE CLAUSE THAT MAKES THIS NOT AN EXTRACTION. A rule that could add
        a sentence could add a fact, and the whole argument for trusting this
        manipulation is that it cannot.

        ONLY THE SENTENCE CLAUSE FIRES HERE, and the reason is worth stating:
        `CoverGate` and `Rust` both occur in the SOURCE -- one inside the
        stripped URL, one inside the stripped table -- so the term clause is
        satisfied by a forgery built from scaffolding vocabulary. The sentence
        clause is the one with teeth; the term clause catches the narrower
        case of vocabulary from nowhere.
        """
        forged = f"{reshape_document(_MESSY)} CoverGate was written in Rust."

        assert gate_reshape(_MESSY, forged)["failed_clauses"] == (
            "every_sentence_came_from_the_source",
        )

    def test_an_invented_term_inside_a_real_sentence_is_caught(self) -> None:
        """A subtler forgery than a whole sentence: the sentence is not from
        the source either, so both clauses fire, and that is correct -- the
        term clause is the one that would still catch a rule which spliced
        vocabulary into text it was entitled to keep.
        """
        verdict = gate_reshape(_MESSY, "The team measured PyTorch against the usual baseline.")

        assert "no_term_was_invented" in verdict["failed_clauses"]

    def test_an_empty_reshape_is_reported(self) -> None:
        assert gate_reshape("| a | b |\n", "")["failed_clauses"] == ("document_is_not_empty",)

    def test_dropping_a_sentence_is_allowed(self) -> None:
        """The rule deletes, so a reshape that keeps LESS than it could is
        still faithful. Only additions are forgeries.

        The kept sentence is taken from the splitter rather than retyped: a
        hand-copied one differs from the source by whatever the splitter's
        whitespace collapsing did, and would fail for a reason the test is
        not about.
        """
        kept = sentences(_MESSY)[0]

        assert gate_reshape(_MESSY, kept)["failed_clauses"] == ()


class TestReshapeCorpus:
    def test_it_preserves_document_order(self) -> None:
        """Order decides which windows the stride holds out, so a reshaped
        corpus that reordered its pages would be a different experiment.
        """
        report = reshape_corpus([_MESSY, "A wholly different page about ClearGBM and Rust."])

        assert report["documents"][0]["source"] == _MESSY
        assert "wholly different" in report["documents"][1]["reshaped"]

    def test_it_counts_what_passed(self) -> None:
        report = reshape_corpus([_MESSY, "| only | a | table |\n"])

        assert report["accepted"] == 1
        assert report["rejected"] == 1

    def test_corpus_terms_lost_are_not_the_sum_of_document_losses(self) -> None:
        """A term dropped from one page's table can survive in another page's
        prose, and the question set is built against the CORPUS. Summing the
        per-document losses would overcount exactly those terms.
        """
        elsewhere = "The registry lists TankpitBot beside the other clients."
        report = reshape_corpus([_MESSY, elsewhere])

        assert "TankpitBot" in report["documents"][0]["terms_lost"]
        assert "TankpitBot" not in report["terms_lost"]
        assert "DTypeLike" in report["terms_lost"]

    def test_an_empty_corpus_reports_zeroes(self) -> None:
        report = reshape_corpus([])

        assert report["documents"] == ()
        assert report["accepted"] == 0
        assert report["rejected"] == 0
        assert report["terms_lost"] == ()


class TestTheClauseList:
    def test_every_clause_the_gate_can_emit_is_named(self) -> None:
        """A clause the gate emits but the list does not name would be a
        rejection reason no report could count.
        """
        emitted = set(gate_reshape(_MESSY, "Invented sentence about PyTorch.")["failed_clauses"])
        emitted |= set(gate_reshape("| a |\n", "")["failed_clauses"])

        assert emitted <= set(RESHAPE_CLAUSES)

    def test_the_rule_only_deletes_so_there_are_exactly_three(self) -> None:
        """Stated as a test because the count is an argument: a rule that
        rewrote text would need clauses about meaning, and this one does not.
        """
        assert len(RESHAPE_CLAUSES) == 3
