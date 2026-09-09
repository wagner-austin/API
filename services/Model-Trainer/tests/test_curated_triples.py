"""The shipped curation, checked against what its own docstring claims.

WHAT THIS CAN AND CANNOT CHECK, stated first because the boundary is the
point. Six of the seven gate clauses depend only on a row and its own source
sentence, so they are checkable here with no corpus at all. The seventh --
whether that sentence is really in the training half -- depends on the twelve
me-wiki pages, which are not in this repository, so it is checked by the arm
at run time against the corpus digest the rows are pinned to.

That split is why every row carries its source sentence verbatim rather than a
pointer to one: a citation nothing can open is a citation nothing checks.
"""

from __future__ import annotations

from model_trainer.core.services.model.editing.curated_triples import (
    CORPUS_DIGEST,
    ME_WIKI_PUBLIC_TRIPLES,
    UNGROUNDABLE_ANSWERS,
)
from model_trainer.core.services.model.editing.grounding import gate_triple

#: Every row judged against its OWN sentence, which turns off the one clause
#: that needs the corpus and leaves the other six fully exercised.
_VERDICTS = tuple(
    gate_triple(candidate, training_sentences=[candidate["source_sentence"]])
    for candidate in ME_WIKI_PUBLIC_TRIPLES
)


class TestTheRowsThemselves:
    def test_every_item_id_is_distinct(self) -> None:
        """Ids pair a triple with its edit record and its verdict."""
        ids = [candidate["item_id"] for candidate in ME_WIKI_PUBLIC_TRIPLES]

        assert len(ids) == len(set(ids))

    def test_no_field_is_blank(self) -> None:
        for candidate in ME_WIKI_PUBLIC_TRIPLES:
            assert candidate["subject"].strip(), candidate["item_id"]
            assert candidate["relation"].strip(), candidate["item_id"]
            assert candidate["object"].strip(), candidate["item_id"]
            assert candidate["source_document"].endswith(".md"), candidate["item_id"]
            assert candidate["source_sentence"].strip(), candidate["item_id"]

    def test_the_corpus_digest_is_a_full_sha256(self) -> None:
        """A truncated digest would match corpora it was not curated against."""
        assert len(CORPUS_DIGEST) == 64
        assert set(CORPUS_DIGEST) <= set("0123456789abcdef")


class TestTheDocumentedRejections:
    def test_exactly_the_ungroundable_answers_fail(self) -> None:
        """THE CLAIM THE MODULE MAKES ABOUT ITSELF, checked rather than trusted.

        `UNGROUNDABLE_ANSWERS` names seven of the twenty distinct answers as
        having no training sentence with a corpus term ahead of them. If a
        later curator rewrites a row and rescues one, this fails and the
        constant has to be corrected -- which is the whole point of writing
        the reject rate down as data instead of prose.
        """
        rejected = {
            candidate["object"]
            for candidate, verdict in zip(ME_WIKI_PUBLIC_TRIPLES, _VERDICTS, strict=True)
            if not verdict["accepted"]
        }

        assert rejected == set(UNGROUNDABLE_ANSWERS)

    def test_thirteen_of_twenty_rows_clear_the_bar(self) -> None:
        """The cost line's headline number, pinned so a silent drift is loud."""
        assert len(ME_WIKI_PUBLIC_TRIPLES) == 20
        assert sum(1 for verdict in _VERDICTS if verdict["accepted"]) == 13

    def test_every_rejection_names_a_reason(self) -> None:
        """A rejection with no clause would be a verdict nobody can act on."""
        for verdict in _VERDICTS:
            assert verdict["accepted"] != bool(verdict["failed_clauses"]), verdict["item_id"]

    def test_no_row_fails_on_a_clause_the_curator_controls(self) -> None:
        """THE CLAUSES THAT WOULD MEAN SLOPPY CURATION RATHER THAN HARD PROSE.

        A subject or object missing from its own sentence, a prompt leaking
        its answer, or a subject containing its object are all curation
        mistakes. None of them appears: every rejection is
        `subject_is_a_corpus_term` or `object_after_subject`, which are facts
        about the sentence, not about the person who read it.
        """
        controllable = {
            "subject_in_sentence",
            "object_in_sentence",
            "prompt_does_not_give_away_object",
            "subject_and_object_are_distinct",
        }
        for verdict in _VERDICTS:
            assert not controllable & set(verdict["failed_clauses"]), verdict["item_id"]
