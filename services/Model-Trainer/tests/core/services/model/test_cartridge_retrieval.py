"""BM25 retrieval, tested on what it must get RIGHT and what it must not know.

The arm exists to be beatable, so the tests that matter are the ones proving
it is a real retriever rather than the oracle wearing a different name: it
sees only the question, it ranks by term evidence, and it returns its best
guesses even when they are wrong.
"""

from __future__ import annotations

from model_trainer.core.services.model import cartridge_retrieval as retrieval

#: Two documents whose sentences share almost no vocabulary, so a correct
#: ranking is unambiguous and a broken one cannot pass by luck.
_DOCUMENTS: tuple[str, ...] = (
    "The gradient booster splits histograms across feature bins. "
    "Quantized training reduces the width of each accumulator. "
    "The learner prunes leaves that carry too little gain.",
    "The submarine navigates by inertial dead reckoning. "
    "Sonar returns are filtered before the pilot ever sees them. "
    "Ballast tanks flood to trim the vessel at depth.",
)


class TestTerms:
    def test_it_lowercases_and_keeps_repeats(self) -> None:
        """Repeats are kept because term FREQUENCY is what BM25 weighs."""
        assert retrieval.terms("Sonar sonar SONAR depth") == [
            "sonar",
            "sonar",
            "sonar",
            "depth",
        ]

    def test_punctuation_is_not_a_term(self) -> None:
        assert retrieval.terms("depth, trim; ballast.") == ["depth", "trim", "ballast"]


class TestBuildIndex:
    def test_it_indexes_every_sentence_as_its_own_chunk(self) -> None:
        index = retrieval.build_index(_DOCUMENTS)

        assert len(index["chunks"]) == 6
        assert len(index["term_frequencies"]) == 6
        assert len(index["chunk_lengths"]) == 6

    def test_document_frequency_counts_chunks_not_occurrences(self) -> None:
        """A term repeated inside one chunk still occurs in ONE chunk.

        Conflating the two would deflate IDF for exactly the terms that
        distinguish a chunk, which is the ranking this arm depends on.
        """
        index = retrieval.build_index(("Sonar and sonar and sonar. Ballast alone.",))

        assert index["document_frequency"]["sonar"] == 1
        assert index["term_frequencies"][0]["sonar"] == 3

    def test_an_empty_corpus_indexes_to_nothing_without_dividing(self) -> None:
        """Zero chunks must not raise on the average-length division."""
        index = retrieval.build_index(())

        assert index["chunks"] == ()
        assert index["average_length"] == 0.0


class TestScoreChunk:
    def test_a_chunk_sharing_no_term_scores_zero(self) -> None:
        index = retrieval.build_index(_DOCUMENTS)

        assert retrieval.score_chunk(index, retrieval.terms("xylophone"), 0) == 0.0

    def test_a_rare_term_outscores_a_common_one(self) -> None:
        """IDF is the half of BM25 that does the discriminating.

        "the" opens most chunks here; "sonar" appears in one. A ranking that
        ignored IDF would rate them equally and retrieve on stopwords.
        """
        index = retrieval.build_index(_DOCUMENTS)
        sonar = next(i for i, c in enumerate(index["chunks"]) if "Sonar" in c)

        rare = retrieval.score_chunk(index, retrieval.terms("sonar"), sonar)
        common = retrieval.score_chunk(index, retrieval.terms("the"), sonar)

        assert rare > common

    def test_idf_never_goes_negative_for_a_very_common_term(self) -> None:
        """The textbook IDF goes negative past 50% document frequency.

        That would make a chunk score WORSE for containing a query word than
        for omitting it, so a query of common words would rank the least
        relevant chunks first. Lucene's non-negative form is used instead.
        """
        index = retrieval.build_index(("shared alpha.", "shared beta.", "shared gamma."))

        assert retrieval.score_chunk(index, retrieval.terms("shared"), 0) > 0.0


class TestRetrieve:
    def test_it_finds_the_chunk_the_question_is_about(self) -> None:
        index = retrieval.build_index(_DOCUMENTS)

        found = retrieval.retrieve(index, "How are sonar returns handled?", limit=1)

        assert "Sonar returns are filtered" in found

    def test_it_cannot_see_an_answer_it_was_not_given(self) -> None:
        """THE DIFFERENCE FROM THE ORACLE, asserted rather than described.

        The oracle is handed the answer term and finds the sentence
        containing it every time. Here the question deliberately shares
        vocabulary with the WRONG document, and the retriever follows the
        words it was given -- which is a retrieval failure, and exactly the
        failure mode the oracle arm can never exhibit.
        """
        index = retrieval.build_index(_DOCUMENTS)

        found = retrieval.retrieve(index, "Which ballast trims the histogram?", limit=1)

        assert "Ballast tanks flood" in found
        assert "histograms across feature bins" not in found

    def test_it_returns_its_best_guesses_even_when_nothing_matches(self) -> None:
        """A real retriever answers every query, right or wrong.

        Returning nothing would let unanswerable questions drop out of the
        arm, and the accuracy reported would then be an average over only
        the items retrieval happened to serve.
        """
        index = retrieval.build_index(_DOCUMENTS)

        found = retrieval.retrieve(index, "xylophone concerto", limit=2)

        assert found != ""
        assert len(found.split(". ")) >= 2

    def test_results_come_back_in_corpus_order(self) -> None:
        """Ranking picks the chunks; corpus order presents them.

        Evidence read back in relevance order would put a later sentence
        before an earlier one it depends on, which changes what the prose
        says without changing which sentences were chosen.
        """
        index = retrieval.build_index(_DOCUMENTS)

        found = retrieval.retrieve(index, "ballast sonar", limit=2)

        assert found.index("Sonar returns") < found.index("Ballast tanks")

    def test_the_limit_bounds_what_comes_back(self) -> None:
        index = retrieval.build_index(_DOCUMENTS)

        assert len(retrieval.retrieve(index, "the", limit=1).split(". ")) == 1
        assert len(retrieval.retrieve(index, "the", limit=3).split(". ")) == 3

    def test_an_empty_index_retrieves_nothing(self) -> None:
        assert retrieval.retrieve(retrieval.build_index(()), "anything") == ""
