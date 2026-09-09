"""Retrieve evidence the way a real system must: from the question alone.

WHY THIS EXISTS BESIDE THE ORACLE. ``cartridge_qa.evidence_for`` selects each
item's evidence by searching for its own ANSWER, which is deliberate and is
documented there: it bounds what any retriever could achieve. But it bounds
the measurement in BOTH directions at once, and that turned out to matter.
On accuracy it is an upper bound -- it answered every item in the 2026-09-07
run, 1.0000 against a 0.5417 base. On latency it is a LOWER bound, because it
embeds nothing, ranks nothing and searches no index; it is handed the answer.
So a cartridge that loses to it has been told nothing about a real pipeline,
which is precisely the position that run ended in.

This module is the arm that can be lost to informatively.

LEXICAL BM25, NOT A DENSE RETRIEVER, and the reason is honesty about what is
installed rather than a claim that lexical is better. A dense arm needs an
embedding model this repo does not carry, and adding one would edit
``pyproject.toml`` for a measurement. BM25 is the standard lexical baseline,
it is deterministic, it needs no weights, and it is a real retriever in the
only sense that matters here: IT CANNOT SEE THE ANSWER. Where it retrieves
the wrong sentences, that is a retrieval failure the oracle arm cannot have,
and reporting it is the entire point.

The chunk is a SENTENCE, matching what the oracle selects, so the two arms
differ in their selection and in nothing else.

INDEXING IS SEPARATED FROM QUERYING because a real system pays them at
different times. Building the index is offline work done once; scoring a
query against it is per-request. Folding the two together would charge
retrieval a cost no deployment pays per query, which is the mirror image of
the error the oracle arm invites.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from typing_extensions import TypedDict

from model_trainer.core.services.model.corpus_cloze import sentences

# THESE WERE MODULE CONSTANTS -- K1 = 1.5, B = 0.75, RETRIEVED_CHUNKS = 5 --
# AND THAT IS WHY THE RETRIEVAL SIDE WAS THREE FIXED POINTS. A cartridge is
# reported as beating or losing to "BM25", and BM25 is a family: its
# saturation, its length normalisation and how many chunks it returns all
# move the arm it names. Frozen at their standard values, in a module nothing
# had to declare, they made the retriever look like a constant of nature
# rather than a configuration somebody chose. Registered as this arm's
# frozen-by-copy knobs in the 2026-09-09 audit and moved onto the plan, where
# they can be swept and where a record carries the values it was measured
# under.
#
# They live on the INDEX rather than being threaded through every scoring
# call because the index is what a measurement builds once and reads many
# times, and binding them there makes it impossible to score one index under
# two configurations by accident.

#: Words, lowercased. Deliberately crude: BM25's strength is that it needs no
#: model, and a clever tokenizer here would be a second, untested one beside
#: the model's own.
_WORD = re.compile(r"[a-z0-9]+")

#: Reciprocal-rank-fusion damping, from Cormack et al. (2009).
#:
#: Sixty, matching `packages/wiki-search/src/fusion.ts` in the MCPs repo
#: exactly. It sets how sharply rank 1 outweighs rank 2: at this value the
#: top few ranks sit close together, so AGREEMENT BETWEEN ARMS matters more
#: than either arm's confidence in its own ordering. Changing it here
#: without changing it there would give the two wikis different retrievers
#: under one name.
RRF_K = 60


class Bm25Index(TypedDict):
    """A searchable index over one corpus's sentences.

    Held in memory and never serialised, which is why it carries no encode or
    decode pair: it is built at the start of a measurement and read by that
    same process. A codec nothing calls is the husk this repo removes.

    Attributes:
        chunks: The sentences, in corpus order. A retrieval result is
            reported as text from here rather than as indices, so nothing
            downstream has to hold this object to read an answer.
        term_frequencies: Per chunk, how often each of its terms occurs.
        chunk_lengths: Per chunk, its term count. Kept beside the frequencies
            rather than summed on demand, because BM25 reads it once per
            (query term, chunk) pair.
        document_frequency: Per term, how many chunks contain it at all.
        average_length: Mean chunk length, the normaliser BM25 divides by.
        k1: Term-frequency saturation. A term occurring ten times in a chunk
            is not ten times as much evidence as one occurrence, and how
            quickly that flattens is this number.
        b: Length normalisation, in ``[0, 1]``. At 0 a long chunk is not
            penalised for its length at all; at 1 it is penalised in full.
        retrieved_chunks: How many chunks one query returns. Part of the
            index because it is part of the ARM: an oracle that concatenates
            every sentence containing the answer is not comparable to a
            single-chunk retriever, and the difference would read as
            selection quality rather than as evidence volume.
    """

    chunks: tuple[str, ...]
    term_frequencies: tuple[Mapping[str, int], ...]
    chunk_lengths: tuple[int, ...]
    document_frequency: Mapping[str, int]
    average_length: float
    k1: float
    b: float
    retrieved_chunks: int


def terms(text: str) -> list[str]:
    """Split text into the terms BM25 matches on.

    Args:
        text: Any text -- a chunk when indexing, a question when querying.

    Returns:
        Lowercased alphanumeric runs, in order, with duplicates kept because
        term frequency is what BM25 weighs.
    """
    # `finditer` rather than `findall`: the latter is typed `list[Any]`
    # because a pattern MAY carry capture groups, and this one does not.
    return [match.group(0) for match in _WORD.finditer(text.lower())]


def build_index(
    documents: Sequence[str], *, k1: float, b: float, retrieved_chunks: int
) -> Bm25Index:
    """Index a corpus's sentences for retrieval, under a stated configuration.

    OFFLINE WORK, and timed as such by callers. A deployment builds this once
    when its corpus changes, not once per question, so charging it to
    per-query latency would overstate retrieval by however long indexing
    happens to take.

    THE THREE PARAMETERS ARE KEYWORD-ONLY AND HAVE NO DEFAULTS. They used to
    be module constants, which is how "the cartridge beats BM25" came to be
    said of one arbitrary point in BM25's parameter space. A caller that has
    not decided what saturation it is measuring under has not decided what it
    is measuring.

    Args:
        documents: Document bodies. Split into sentences by the same splitter
            the item builder and the oracle arm use, so all three agree on
            what a sentence is.
        k1: Term-frequency saturation.
        b: Length normalisation.
        retrieved_chunks: How many chunks a query returns.

    Returns:
        The index. Empty of chunks when the corpus yields no sentences, which
        is a real state a caller must handle rather than an error here.
    """
    chunks = tuple(sentence for document in documents for sentence in sentences(document))
    frequencies: list[Mapping[str, int]] = []
    lengths: list[int] = []
    document_frequency: dict[str, int] = {}
    for chunk in chunks:
        counted: dict[str, int] = {}
        for term in terms(chunk):
            counted[term] = counted.get(term, 0) + 1
        frequencies.append(counted)
        lengths.append(sum(counted.values()))
        for term in counted:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    total = sum(lengths)
    return Bm25Index(
        chunks=chunks,
        term_frequencies=tuple(frequencies),
        chunk_lengths=tuple(lengths),
        document_frequency=document_frequency,
        # Zero for an empty corpus rather than a division: `score_chunk`
        # never runs without chunks, so the value is unused, and guarding it
        # here keeps the arithmetic below unconditional.
        average_length=total / float(len(chunks)) if chunks else 0.0,
        k1=k1,
        b=b,
        retrieved_chunks=retrieved_chunks,
    )


def score_chunk(index: Bm25Index, query_terms: Sequence[str], chunk: int) -> float:
    """Score one chunk against one query, by BM25.

    Args:
        index: The index the chunk belongs to.
        query_terms: The question's terms, from :func:`terms`.
        chunk: Which chunk to score.

    Returns:
        The score. Zero when the chunk shares no term with the query, which
        is the common case and not a failure.
    """
    frequencies = index["term_frequencies"][chunk]
    length = index["chunk_lengths"][chunk]
    total_chunks = len(index["chunks"])
    score = 0.0
    for term in query_terms:
        occurrences = frequencies.get(term, 0)
        if occurrences == 0:
            continue
        containing = index["document_frequency"][term]
        # Lucene's non-negative IDF. The textbook form goes negative for a
        # term in more than half the corpus, which would make a chunk score
        # WORSE for containing a common query word than for omitting it.
        idf = math.log(1.0 + (total_chunks - containing + 0.5) / (containing + 0.5))
        normalised = length / index["average_length"]
        k1 = index["k1"]
        b = index["b"]
        score += idf * (occurrences * (k1 + 1.0)) / (occurrences + k1 * (1.0 - b + b * normalised))
    return score


def retrieve(index: Bm25Index, query: str) -> str:
    """Retrieve the best chunks for one question.

    RETURNS ITS TOP CHOICES EVEN WHEN NOTHING MATCHES, which is deliberate
    and is the behaviour that makes this arm informative. A real retriever
    handed a question it cannot serve returns its top k anyway, and the
    reader is misled by irrelevant evidence rather than told there is none.
    Refusing instead would quietly convert retrieval failures into skipped
    items and report an accuracy averaged over only the questions retrieval
    happened to get right.

    Ties break by corpus order, so the result is a function of the corpus and
    the query and nothing else.

    Args:
        index: The index to search.
        query: The question, as the asker wrote it. The ANSWER is not an
            argument here, and that is the whole difference from the oracle.

    Returns:
        The chosen chunks joined by spaces, in corpus order. Empty when the
        index holds no chunks.
    """
    return join_chunks(index, rank_chunks(index, query)[: index["retrieved_chunks"]])


def rank_chunks(index: Bm25Index, query: str) -> tuple[int, ...]:
    """Rank every chunk against one query, best first.

    Separated from :func:`retrieve` because fusion consumes RANKS rather
    than text: reciprocal-rank fusion combines the positions two arms
    assigned, so an arm that only returned its joined evidence could not
    participate.

    Args:
        index: The index to search.
        query: The question. The ANSWER is not an argument, which is the
            whole difference from the oracle.

    Returns:
        Every chunk index, best first. Ties break by corpus position, so the
        order is a total function of the corpus and the query.
    """
    query_terms = terms(query)

    def rank(chunk: int) -> tuple[float, int]:
        """Order by score descending, then by corpus position.

        A named function rather than a lambda because a lambda's parameter
        is untyped, and an untyped parameter is an ``Any`` this repo refuses.

        Args:
            chunk: Which chunk to place.

        Returns:
            The sort key. The chunk index is the tie-break, so equal scores
            resolve by corpus order rather than by whatever order `sorted`
            happened to see them in.
        """
        return (-score_chunk(index, query_terms, chunk), chunk)

    return tuple(sorted(range(len(index["chunks"])), key=rank))


def join_chunks(index: Bm25Index, chosen: Sequence[int]) -> str:
    """Render chosen chunks as evidence text.

    Args:
        index: The index the chunks belong to.
        chosen: Chunk indices, in any order.

    Returns:
        The chunks joined by spaces, in CORPUS order rather than rank order.
        Evidence read back in relevance order would put a later sentence
        before an earlier one it depends on, changing what the prose says
        without changing which sentences were chosen.
    """
    return " ".join(index["chunks"][chunk] for chunk in sorted(chosen))


def fuse_by_reciprocal_rank(
    dense_ranked: Sequence[int], lexical_ranked: Sequence[int], *, limit: int
) -> tuple[int, ...]:
    """Fuse two rankings into one by reciprocal rank.

    PORTED, NOT SHARED, and the reason is a language boundary rather than a
    preference. ``packages/wiki-search/src/fusion.ts`` in the MCPs repo
    already does this for the civic wiki and its docstring carries the
    argument for it; that file is TypeScript and this is Python, so there is
    no lift available -- only a reimplementation. The constant and the
    tie-break are kept identical to it so the two agree, and the algorithm
    is Cormack et al. (2009) rather than either codebase's invention.

    WHY RANKS RATHER THAN SCORES, from that same file's reasoning: the two
    arms produce scores on incomparable scales. Cosine similarity lands in a
    compressed band while BM25 is unbounded above and depends on corpus
    statistics, so normalising either onto the other is a tuning exercise
    that silently re-tunes itself every time the corpus grows. Combining
    positions sidesteps the scales entirely -- placing second in both arms
    beats placing first in one and nowhere in the other, which is exactly
    the behaviour a hybrid retriever exists to get.

    Args:
        dense_ranked: Chunk indices from the embedding arm, best first.
        lexical_ranked: Chunk indices from BM25, best first.
        limit: How many fused chunks to return.

    Returns:
        The fused top chunks, best first. Ties break by chunk index
        ascending so the ordering is total and reproducible.
    """
    dense_at = _first_positions(dense_ranked)
    lexical_at = _first_positions(lexical_ranked)
    scored: list[tuple[float, int]] = []
    for chunk in sorted(set(dense_ranked) | set(lexical_ranked)):
        score = 0.0
        for positions in (dense_at, lexical_at):
            rank = positions.get(chunk)
            if rank is not None:
                score += 1.0 / float(RRF_K + rank)
        scored.append((-score, chunk))
    return tuple(chunk for _score, chunk in sorted(scored)[:limit])


def _first_positions(ranked: Sequence[int]) -> dict[int, int]:
    """Map each chunk to its 1-based rank, keeping the first occurrence.

    Args:
        ranked: Chunk indices, best first.

    Returns:
        Chunk to rank. A repeated id keeps its BEST position rather than its
        last, because a duplicate later in a list does not make a document
        less relevant than the first mention said it was.
    """
    positions: dict[int, int] = {}
    for index, chunk in enumerate(ranked):
        if chunk not in positions:
            positions[chunk] = index + 1
    return positions


__all__ = [
    "RRF_K",
    "Bm25Index",
    "build_index",
    "fuse_by_reciprocal_rank",
    "join_chunks",
    "rank_chunks",
    "retrieve",
    "score_chunk",
    "terms",
]
