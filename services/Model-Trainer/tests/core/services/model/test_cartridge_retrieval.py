"""BM25 retrieval, tested on what it must get RIGHT and what it must not know.

The arm exists to be beatable, so the tests that matter are the ones proving
it is a real retriever rather than the oracle wearing a different name: it
sees only the question, it ranks by term evidence, and it returns its best
guesses even when they are wrong.
"""

from __future__ import annotations

from tests._retrieval_support import standard_index

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
        index = standard_index(_DOCUMENTS)

        assert len(index["chunks"]) == 6
        assert len(index["term_frequencies"]) == 6
        assert len(index["chunk_lengths"]) == 6

    def test_document_frequency_counts_chunks_not_occurrences(self) -> None:
        """A term repeated inside one chunk still occurs in ONE chunk.

        Conflating the two would deflate IDF for exactly the terms that
        distinguish a chunk, which is the ranking this arm depends on.
        """
        index = standard_index(("Sonar and sonar and sonar. Ballast alone.",))

        assert index["document_frequency"]["sonar"] == 1
        assert index["term_frequencies"][0]["sonar"] == 3

    def test_an_empty_corpus_indexes_to_nothing_without_dividing(self) -> None:
        """Zero chunks must not raise on the average-length division."""
        index = standard_index(())

        assert index["chunks"] == ()
        assert index["average_length"] == 0.0


class TestScoreChunk:
    def test_a_chunk_sharing_no_term_scores_zero(self) -> None:
        index = standard_index(_DOCUMENTS)

        assert retrieval.score_chunk(index, retrieval.terms("xylophone"), 0) == 0.0

    def test_a_rare_term_outscores_a_common_one(self) -> None:
        """IDF is the half of BM25 that does the discriminating.

        "the" opens most chunks here; "sonar" appears in one. A ranking that
        ignored IDF would rate them equally and retrieve on stopwords.
        """
        index = standard_index(_DOCUMENTS)
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
        index = standard_index(("shared alpha.", "shared beta.", "shared gamma."))

        assert retrieval.score_chunk(index, retrieval.terms("shared"), 0) > 0.0


class TestTheScoringParametersAreActuallyParameters:
    """Declared fields that nothing can move are constants with extra steps.

    ``k1``, ``b`` and ``retrieved_chunks`` were module constants until
    2026-09-09, which is how "the cartridge beats BM25" came to be reported
    of one arbitrary point in BM25's parameter space. Moving them onto the
    plan is only worth anything if they demonstrably change the arm, so each
    is measured here in the direction its definition predicts.
    """

    #: One chunk repeating a term, one longer chunk holding it once. Written
    #: to make saturation and length normalisation separately visible.
    _MIXED = (
        "Sonar and sonar and sonar returns. Ballast tanks flood slowly and deliberately here now.",
    )

    def test_raising_saturation_raises_the_score_of_a_repeated_term(self) -> None:
        """``k1`` is how slowly repetition stops counting.

        At a low ``k1`` the second and third occurrence add almost nothing;
        raising it lets them count. Measured 0.9050 -> 1.1980 -> 1.4648 for
        the three-occurrence chunk.
        """
        scores = [
            retrieval.score_chunk(
                retrieval.build_index(self._MIXED, k1=k1, b=0.75, retrieved_chunks=5),
                retrieval.terms("sonar"),
                0,
            )
            for k1 in (0.5, 1.5, 3.0)
        ]

        assert scores[0] < scores[1] < scores[2]

    def test_raising_length_normalisation_penalises_the_longer_chunk(self) -> None:
        """``b`` is how much a long chunk is discounted for its length.

        At ``b = 0`` length is ignored entirely; at 1 it is charged in full.
        The chunk measured here is longer than the corpus average, so its
        score must fall as ``b`` rises: 0.6931 -> 0.6513 -> 0.6384.
        """
        scores = [
            retrieval.score_chunk(
                retrieval.build_index(self._MIXED, k1=1.5, b=b, retrieved_chunks=5),
                retrieval.terms("ballast"),
                1,
            )
            for b in (0.0, 0.75, 1.0)
        ]

        assert scores[0] > scores[1] > scores[2]

    def test_the_index_carries_the_configuration_it_was_built_under(self) -> None:
        """A record has to be able to say what it measured.

        The values live on the index rather than in a module so that nothing
        can score one index under two configurations by accident, and so a
        run can report the retriever it actually ran.
        """
        index = retrieval.build_index(self._MIXED, k1=2.25, b=0.4, retrieved_chunks=3)

        assert index["k1"] == 2.25
        assert index["b"] == 0.4
        assert index["retrieved_chunks"] == 3


class TestExpandQuery:
    """Searching twice, and the two ways that can go."""

    #: A corpus where the question's words and the answer's words differ,
    #: which is the failure ranking alone cannot fix.
    _VOCAB = (
        "The vessel submerges using ballast. "
        "Ballast tanks flood to trim the craft at depth. "
        "Depth control depends on trim and on flooding rate.",
    )

    def test_it_adds_terms_the_question_did_not_contain(self) -> None:
        """The whole mechanism: reach words the asker did not use.

        WHAT IT DOES NOT ADD IS THE INTERESTING HALF, and this test was
        written asserting the wrong thing first. "ballast" is the obvious
        expansion of a question about submerging, and it is NOT chosen: it
        occurs in two of the three chunks, so its inverse document frequency
        is lower than that of terms occurring once. The arm adds what
        DISCRIMINATES between chunks, not what a reader would free-associate,
        and asserting the latter would have pinned an intuition rather than
        the algorithm.
        """
        question = "How does the vessel submerge?"
        index = standard_index(self._VOCAB)

        expanded = retrieval.expand_query(index, question, feedback_chunks=2, expansion_terms=3)

        assert expanded.startswith(question)
        added = set(retrieval.terms(expanded)) - set(retrieval.terms(question))
        assert len(added) == 3
        # Every added term came from the corpus rather than from nowhere.
        assert added <= set(retrieval.terms(self._VOCAB[0]))
        # And the rarer terms outrank the twice-occurring one.
        assert retrieval.inverse_document_frequency(
            index, next(iter(added))
        ) >= retrieval.inverse_document_frequency(index, "ballast")

    def test_it_never_re_adds_a_term_the_query_already_has(self) -> None:
        """Repeating a query term would reweight the original, not expand it."""
        index = standard_index(self._VOCAB)
        query = "ballast trim depth"

        expanded = retrieval.expand_query(index, query, feedback_chunks=3, expansion_terms=4)

        original = retrieval.terms(query)
        added = [term for term in retrieval.terms(expanded) if term not in original]
        assert len(set(added)) == len(added)
        assert not set(added) & set(original)

    def test_asking_for_no_terms_returns_the_query_unchanged(self) -> None:
        """The identity case has to be the identity, not a near miss."""
        index = standard_index(self._VOCAB)

        assert (
            retrieval.expand_query(index, "ballast", feedback_chunks=2, expansion_terms=0)
            == "ballast"
        )

    def test_the_expansion_is_a_function_of_the_corpus_and_the_query_alone(self) -> None:
        """Two runs of one plan must agree, so ties break on the term."""
        index = standard_index(self._VOCAB)

        first = retrieval.expand_query(index, "trim", feedback_chunks=3, expansion_terms=6)
        second = retrieval.expand_query(index, "trim", feedback_chunks=3, expansion_terms=6)

        assert first == second

    def test_a_wider_feedback_set_can_change_what_is_added(self) -> None:
        """Which chunks are assumed relevant is the arm's central assumption.

        If this were inert the parameter would be decoration, and the arm
        would be reporting a setting it does not actually respond to.
        """
        index = standard_index(self._VOCAB)

        narrow = retrieval.expand_query(index, "submerges", feedback_chunks=1, expansion_terms=2)
        wide = retrieval.expand_query(index, "submerges", feedback_chunks=3, expansion_terms=2)

        assert narrow != wide


class TestInverseDocumentFrequency:
    """The weight expansion and scoring must agree on."""

    def test_a_term_in_every_chunk_still_weighs_above_zero(self) -> None:
        """Lucene's non-negative form, asserted where it is now shared."""
        index = standard_index(("shared alpha.", "shared beta.", "shared gamma."))

        assert retrieval.inverse_document_frequency(index, "shared") > 0.0

    def test_a_rarer_term_weighs_more(self) -> None:
        index = standard_index(("shared alpha.", "shared beta.", "shared gamma."))

        rare = retrieval.inverse_document_frequency(index, "alpha")
        common = retrieval.inverse_document_frequency(index, "shared")

        assert rare > common


class TestRetrieve:
    def test_it_finds_the_chunk_the_question_is_about(self) -> None:
        found = retrieval.retrieve(
            standard_index(_DOCUMENTS, retrieved_chunks=1), "How are sonar returns handled?"
        )

        assert "Sonar returns are filtered" in found

    def test_it_cannot_see_an_answer_it_was_not_given(self) -> None:
        """THE DIFFERENCE FROM THE ORACLE, asserted rather than described.

        The oracle is handed the answer term and finds the sentence
        containing it every time. Here the question deliberately shares
        vocabulary with the WRONG document, and the retriever follows the
        words it was given -- which is a retrieval failure, and exactly the
        failure mode the oracle arm can never exhibit.
        """
        found = retrieval.retrieve(
            standard_index(_DOCUMENTS, retrieved_chunks=1), "Which ballast trims the histogram?"
        )

        assert "Ballast tanks flood" in found
        assert "histograms across feature bins" not in found

    def test_it_returns_its_best_guesses_even_when_nothing_matches(self) -> None:
        """A real retriever answers every query, right or wrong.

        Returning nothing would let unanswerable questions drop out of the
        arm, and the accuracy reported would then be an average over only
        the items retrieval happened to serve.
        """
        found = retrieval.retrieve(
            standard_index(_DOCUMENTS, retrieved_chunks=2), "xylophone concerto"
        )

        assert found != ""
        assert len(found.split(". ")) >= 2

    def test_results_come_back_in_corpus_order(self) -> None:
        """Ranking picks the chunks; corpus order presents them.

        Evidence read back in relevance order would put a later sentence
        before an earlier one it depends on, which changes what the prose
        says without changing which sentences were chosen.
        """
        found = retrieval.retrieve(standard_index(_DOCUMENTS, retrieved_chunks=2), "ballast sonar")

        assert found.index("Sonar returns") < found.index("Ballast tanks")

    def test_the_limit_bounds_what_comes_back(self) -> None:
        returned = retrieval.retrieve(standard_index(_DOCUMENTS, retrieved_chunks=1), "the")
        assert len(returned.split(". ")) == 1
        returned = retrieval.retrieve(standard_index(_DOCUMENTS, retrieved_chunks=3), "the")
        assert len(returned.split(". ")) == 3

    def test_an_empty_index_retrieves_nothing(self) -> None:
        assert retrieval.retrieve(standard_index(()), "anything") == ""


class TestReciprocalRankFusion:
    """Ported from wiki-search's fusion.ts, so the tests pin what must agree.

    The constant and the tie-break are the two things that would silently
    give the two repos different retrievers under one name.
    """

    def test_the_damping_constant_matches_the_typescript_it_was_ported_from(self) -> None:
        """Sixty, from Cormack et al. (2009), same as fusion.ts."""
        assert retrieval.RRF_K == 60

    def test_agreeing_arms_beat_one_arm_s_first_place(self) -> None:
        """THE PROPERTY THE WHOLE FUSION EXISTS FOR.

        Chunk 7 places second in both arms; chunk 1 places first in one and
        nowhere in the other. Any scheme that merely took the best single
        rank would put chunk 1 on top, and the hybrid would then be an
        expensive way to run whichever arm was loudest.
        """
        dense = [1, 7, 2]
        lexical = [3, 7, 4]

        fused = retrieval.fuse_by_reciprocal_rank(dense, lexical, limit=1)

        assert fused == (7,)

    def test_a_chunk_only_one_arm_found_still_places(self) -> None:
        """A strong single-arm hit must survive, or the fusion is an AND.

        The lexical arm is what finds proper nouns the embedder misses; if
        appearing in one arm disqualified a chunk, the hybrid would be worse
        than either half.
        """
        fused = retrieval.fuse_by_reciprocal_rank([5], [9], limit=2)

        assert set(fused) == {5, 9}

    def test_a_repeated_id_keeps_its_best_position(self) -> None:
        """A later duplicate does not make a chunk less relevant."""
        first = retrieval.fuse_by_reciprocal_rank([4, 8, 4], [8], limit=2)
        without = retrieval.fuse_by_reciprocal_rank([4, 8], [8], limit=2)

        assert first == without

    def test_ties_break_by_chunk_index_so_the_order_is_total(self) -> None:
        """Two chunks in identical positions in both arms score identically.

        Left to sort stability that would depend on set iteration order,
        which is not a reproducible measurement.
        """
        fused = retrieval.fuse_by_reciprocal_rank([2, 6], [2, 6], limit=2)

        assert fused == (2, 6)

    def test_the_limit_bounds_the_fused_result(self) -> None:
        assert len(retrieval.fuse_by_reciprocal_rank([1, 2, 3], [3, 2, 1], limit=2)) == 2

    def test_fusing_nothing_returns_nothing(self) -> None:
        assert retrieval.fuse_by_reciprocal_rank([], [], limit=5) == ()


class TestRankAndJoin:
    def test_ranking_returns_every_chunk_best_first(self) -> None:
        index = standard_index(_DOCUMENTS)

        ranked = retrieval.rank_chunks(index, "How are sonar returns handled?")

        assert len(ranked) == len(index["chunks"])
        assert "Sonar returns are filtered" in index["chunks"][ranked[0]]

    def test_joining_reads_back_in_corpus_order_not_rank_order(self) -> None:
        """Evidence in relevance order changes what the prose says."""
        index = standard_index(_DOCUMENTS)

        joined = retrieval.join_chunks(index, [4, 3])

        assert joined == f"{index['chunks'][3]} {index['chunks'][4]}"
