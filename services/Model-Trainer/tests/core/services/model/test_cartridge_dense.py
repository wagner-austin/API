"""The dense arm's logic, tested without loading a hundred megabytes.

WHAT IS FAKED AND WHY. The embedder is, because loading real gte weights
would make every one of these tests need a model cache to run at all. What
is NOT faked is the arithmetic under test -- pooling, normalisation-aware
ranking, tie-breaking -- because that is where this arm can be wrong in the
way that matters: silently returning plausible vectors that rank badly, so
the arm reads as evidence about dense retrieval when it is evidence about a
pooling bug.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from model_trainer.core.services.model import cartridge_dense as dense
from model_trainer.core.services.model.cartridge_retrieval import build_index

_DOCUMENTS: tuple[str, ...] = (
    "The submarine navigates by inertial dead reckoning under the ice. "
    "Sonar returns are filtered before the pilot ever sees any of them.",
)


def _planted(mapping: dict[str, tuple[float, ...]]) -> dense.EmbedderProto:
    """Build an embedder returning planted, already-normalised vectors.

    Planted rather than random: the ranking under test is a function of the
    vectors, so a test that cannot state which vector each text gets cannot
    state what the right answer is either.

    Args:
        mapping: Text to its vector. Any text absent gets a zero vector,
            which scores zero against everything and so sorts last on merit
            rather than by accident.

    Returns:
        The embedder.
    """

    def embed(texts: Sequence[str], /) -> torch.Tensor:
        width = len(next(iter(mapping.values())))
        rows: list[list[float]] = [list(mapping.get(text, (0.0,) * width)) for text in texts]
        return torch.tensor(rows, dtype=torch.float32)

    return embed


class TestMaskedMean:
    def test_padding_is_excluded_from_the_average(self) -> None:
        """THE BUG THIS PREVENTS DOES NOT RAISE, it just retrieves worse.

        Averaging over padded positions drags every short text toward
        whatever padding embeds to, shortening the distance between
        unrelated short chunks.
        """
        rows: list[list[list[float]]] = [[[2.0, 4.0], [4.0, 8.0], [100.0, 100.0]]]
        flags: list[list[int]] = [[1, 1, 0]]
        hidden = torch.tensor(rows, dtype=torch.float32)
        mask = torch.tensor(flags, dtype=torch.long)

        pooled = dense.masked_mean(hidden, mask)

        # Compared elementwise as floats rather than through `tolist`, which
        # is typed Any and would smuggle one into an assertion.
        assert float(pooled[0][0]) == 3.0
        assert float(pooled[0][1]) == 6.0

    def test_an_all_padding_row_does_not_divide_by_zero(self) -> None:
        """A degenerate row must return a vector, not a NaN.

        A NaN propagates into every similarity score and turns the whole
        ranking into sort order, which looks like a result.
        """
        pooled = dense.masked_mean(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long))

        assert torch.isfinite(pooled).all()


class TestRankBySimilarity:
    def test_the_nearest_vector_ranks_first(self) -> None:
        query_row: list[float] = [1.0, 0.0]
        chunk_rows: list[list[float]] = [[0.0, 1.0], [1.0, 0.0], [0.7071, 0.7071]]
        query = torch.tensor(query_row, dtype=torch.float32)
        chunks = torch.tensor(chunk_rows, dtype=torch.float32)

        ranked = dense.rank_by_similarity(query, chunks, tie_break=range(3))

        assert ranked == (1, 2, 0)

    def test_equal_scores_break_by_corpus_position(self) -> None:
        """Otherwise the order depends on sort stability over float equality,
        which is not a reproducible measurement.
        """
        query_row: list[float] = [1.0, 0.0]
        chunk_rows: list[list[float]] = [[1.0, 0.0], [1.0, 0.0]]
        query = torch.tensor(query_row, dtype=torch.float32)
        chunks = torch.tensor(chunk_rows, dtype=torch.float32)

        assert dense.rank_by_similarity(query, chunks, tie_break=range(2)) == (0, 1)


class TestEmbedChunks:
    def test_it_embeds_the_corpus_once_and_returns_a_row_per_chunk(self) -> None:
        index = build_index(_DOCUMENTS)
        embed = _planted(dict.fromkeys(index["chunks"], (1.0, 0.0)))

        vectors = dense.embed_chunks(index, embed)

        assert vectors.shape == (len(index["chunks"]), 2)

    def test_an_empty_index_embeds_nothing_at_all(self) -> None:
        """Returning early matters: embedding an empty batch is a torch error
        rather than an empty result.
        """

        def refuse(texts: Sequence[str], /) -> torch.Tensor:
            raise AssertionError("the embedder ran on an empty index")

        assert dense.embed_chunks(build_index(()), refuse).shape == (0, 0)


class TestDenseRanking:
    def test_it_ranks_the_chunk_whose_meaning_matches(self) -> None:
        index = build_index(_DOCUMENTS)
        sonar = next(i for i, c in enumerate(index["chunks"]) if "Sonar" in c)
        other = 1 - sonar
        embed = _planted(
            {
                index["chunks"][sonar]: (1.0, 0.0),
                index["chunks"][other]: (0.0, 1.0),
                "how is sonar handled": (1.0, 0.0),
            }
        )
        vectors = dense.embed_chunks(index, embed)

        ranked = dense.dense_ranking(vectors, "how is sonar handled", embed)

        assert ranked[0] == sonar

    def test_it_embeds_the_query_and_only_the_query(self) -> None:
        """THE DEFECT THIS SIGNATURE EXISTS TO PREVENT, asserted directly.

        The first version took the index and re-embedded every chunk on
        every call. Timed that way the arm recorded 17452 ms/item against
        BM25's 72 -- a 240x gap that measured a design nobody deploys. A
        per-request path may embed exactly one text: the question.
        """
        index = build_index(_DOCUMENTS)
        embed = _planted(dict.fromkeys(index["chunks"], (1.0, 0.0)))
        vectors = dense.embed_chunks(index, embed)
        seen: list[int] = []

        def counting(texts: Sequence[str], /) -> torch.Tensor:
            seen.append(len(texts))
            return embed(texts)

        dense.dense_ranking(vectors, "how is sonar handled", counting)

        assert seen == [1], "the query path embedded something other than the query"

    def test_it_ranks_every_chunk_so_fusion_has_a_full_ordering(self) -> None:
        """Fusion combines POSITIONS, so a truncated ranking would give the
        dense arm no opinion about chunks it merely rated lower.
        """
        index = build_index(_DOCUMENTS)
        embed = _planted({index["chunks"][0]: (1.0, 0.0)})
        vectors = dense.embed_chunks(index, embed)

        assert len(dense.dense_ranking(vectors, "anything", embed)) == len(index["chunks"])

    def test_no_chunks_ranks_nothing_without_embedding(self) -> None:
        def refuse(texts: Sequence[str], /) -> torch.Tensor:
            raise AssertionError("the embedder ran with no chunks to rank")

        assert dense.dense_ranking(torch.zeros((0, 0)), "anything", refuse) == ()


class TestModelChoice:
    def test_the_model_is_the_family_wiki_search_deploys(self) -> None:
        """Matched by FAMILY, not by leaderboard rank.

        `packages/wiki-search` runs Xenova/gte-small over the civic wiki, so
        measuring against a gte model keeps this answerable about the
        retriever the workspace actually serves.
        """
        assert dense.DENSE_MODEL_ID.startswith("thenlper/gte")


class TestProductionEmbedder:
    """The real weights, loaded once, because the plumbing can be wrong quietly.

    Follows the pattern `_default_load_hf_model` uses -- that one loads a
    real `sshleifer/tiny-gpt2` rather than faking transformers away. gte-base
    is in the local cache, so this reads from disk and downloads nothing.
    """

    def test_it_returns_unit_vectors_of_the_model_s_width(self) -> None:
        """Normalisation is asserted because `rank_by_similarity` ASSUMES it.

        That function takes a dot product as cosine. An unnormalised
        embedder makes long chunks win on magnitude rather than meaning, and
        nothing raises.
        """
        vectors = dense._default_embedder(["one short sentence", "another short sentence"])

        assert vectors.shape == (2, 768)
        # A unit vector dotted with itself is exactly 1.0, which asserts the
        # normalisation without reaching for `Tensor.norm` -- typed Any, and
        # an Any in an assertion is what this repo refuses.
        for row in range(2):
            assert abs(float(vectors[row] @ vectors[row]) - 1.0) < 1e-5

    def test_it_places_a_related_sentence_nearer_than_an_unrelated_one(self) -> None:
        """The check that the pooling is right rather than merely finite.

        A wrong-but-plausible pooling returns unit vectors that rank badly,
        and the arm would then read as evidence about dense retrieval when
        it is evidence about this function.
        """
        vectors = dense._default_embedder(
            [
                "Sonar returns are filtered before the pilot ever sees them.",
                "Ballast tanks flood to trim the vessel at depth.",
                "How are sonar returns handled?",
            ]
        )

        related = float(vectors[0] @ vectors[2])
        unrelated = float(vectors[1] @ vectors[2])

        assert related > unrelated
