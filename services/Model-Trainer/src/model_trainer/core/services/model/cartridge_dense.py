"""Rank corpus chunks by embedding similarity, the way a real system would.

THE ARM BM25 IS NOT. ``cartridge_retrieval`` matches terms; this matches
MEANING, which is what a deployed retriever actually does and what the
2026-09-08 ladder left untested. The cartridge overtook BM25 on accuracy at
774M and 1.5B, and BM25's per-query search costs 1.84 ms/item -- nearly
free. A dense arm is the honest next opponent on both counts: it retrieves
better, and it costs real milliseconds, which is the regime where the
cartridge's latency advantage was measured but never contested.

WHY gte RATHER THAN A BETTER MODEL. ``packages/wiki-search`` in the MCPs
repo already runs a dense arm over the civic wiki with ``Xenova/gte-small``,
an ONNX port of ``thenlper/gte-small``. Using the same family keeps this
measurement answerable against the retriever the workspace actually
deploys, rather than against whichever embedder happens to score best on a
leaderboard. The size differs -- base rather than small -- because base is
what is in the local cache and nothing here should download weights.

MEAN POOLING WITH THE ATTENTION MASK, THEN L2 NORMALISATION, which is what
the gte model card specifies. Getting this wrong does not raise: it returns
plausible vectors that rank badly, and the arm would then look like evidence
that dense retrieval is weak on this corpus when it is evidence that the
pooling was wrong. The masked mean is asserted directly in the tests for
that reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from model_trainer.core.services.model.cartridge_retrieval import Bm25Index

#: The embedding model this arm measures with.
#:
#: Matched by FAMILY to what `packages/wiki-search` deploys, not by size.
DENSE_MODEL_ID = "thenlper/gte-base"


class EmbedderProto(Protocol):
    """Protocol for the text embedder the dense arm ranks with.

    Behind a hook because it loads real weights. A test that reached the
    real one would need a model cache to run at all, and the arm's LOGIC --
    pooling, normalisation, ranking, fusion -- is what the tests are for.
    """

    def __call__(self, texts: Sequence[str], /) -> torch.Tensor:
        """Embed each text, returning one L2-normalised row per input."""
        ...


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average token vectors, counting only real tokens.

    THE MASK IS THE WHOLE POINT. A padded batch carries positions that mean
    nothing, and averaging over them drags every short text toward whatever
    the padding embeds to -- which shortens the distance between unrelated
    short chunks and makes the ranking quietly worse. It does not fail; it
    just retrieves the wrong thing.

    Args:
        hidden: Token vectors, shaped (batch, tokens, features).
        mask: Attention mask, shaped (batch, tokens), 1 for real tokens.

    Returns:
        One pooled vector per row, shaped (batch, features).
    """
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)


def rank_by_similarity(
    query: torch.Tensor, chunks: torch.Tensor, *, tie_break: Sequence[int]
) -> tuple[int, ...]:
    """Rank chunks against one query by cosine similarity, best first.

    Cosine reduces to a dot product because both sides arrive L2-normalised,
    so this does not renormalise -- doing it twice would hide an unnormalised
    embedder rather than let its ranking look wrong.

    Args:
        query: One query vector, shaped (features,).
        chunks: Chunk vectors, shaped (chunks, features).
        tie_break: Corpus position per chunk, used to break equal scores so
            the ordering is total and reproducible rather than dependent on
            sort stability over float equality.

    Returns:
        Chunk indices, best first.
    """
    # Read out one element at a time rather than through `Tensor.tolist`,
    # which is typed Any. A named key function rather than a lambda for the
    # same reason a lambda's parameter is untyped.
    products = chunks @ query
    scores: list[float] = [float(products[chunk]) for chunk in range(len(tie_break))]

    def rank(chunk: int) -> tuple[float, int]:
        """Order by similarity descending, then by corpus position.

        Args:
            chunk: Which chunk to place.

        Returns:
            The sort key, with the corpus position as the tie-break.
        """
        return (-scores[chunk], tie_break[chunk])

    return tuple(sorted(range(len(scores)), key=rank))


def dense_ranking(index: Bm25Index, query: str, embed: EmbedderProto) -> tuple[int, ...]:
    """Rank an index's chunks against one question by meaning.

    Embeds the chunks on every call rather than caching them, which is
    deliberate for a MEASUREMENT: the caller times indexing and querying
    separately, and a hidden cache would move indexing cost into whichever
    query happened to run first. A deployment would cache; this is not one.

    Args:
        index: The index whose chunks to rank. Its BM25 statistics are
            unused here -- the chunk list is the shared corpus view, so both
            arms rank exactly the same units.
        query: The question. The answer is not an argument.
        embed: The embedder.

    Returns:
        Chunk indices, best first. Empty when the index holds no chunks.
    """
    chunks = index["chunks"]
    if not chunks:
        return ()
    vectors = embed(list(chunks))
    query_vector = embed([query])[0]
    return rank_by_similarity(query_vector, vectors, tie_break=range(len(chunks)))


class _BatchProto(Protocol):
    """The tokenizer's output, read by key rather than unpacked.

    Narrowed here because transformers ships no types this repo can trust.
    Reading ``input_ids`` and ``attention_mask`` explicitly rather than
    splatting the batch into the model is what makes that possible -- and it
    also names, at the call site, the two tensors the pooling depends on.
    """

    def __getitem__(self, key: str) -> torch.Tensor:
        """Return one named tensor from the batch."""
        ...


class _TokenizerProto(Protocol):
    """A loaded tokenizer, called on a list of texts."""

    def __call__(
        self, texts: list[str], *, padding: bool, truncation: bool, return_tensors: str
    ) -> _BatchProto:
        """Encode a batch of texts."""
        ...


class _TokenizerClassProto(Protocol):
    """The ``AutoTokenizer`` class object."""

    def from_pretrained(self, model_id: str) -> _TokenizerProto:
        """Load the tokenizer for a model id."""
        ...


class _EncoderOutputProto(Protocol):
    """What an encoder forward pass returns."""

    @property
    def last_hidden_state(self) -> torch.Tensor:
        """Token vectors, shaped (batch, tokens, features)."""
        ...


class _EncoderProto(Protocol):
    """A loaded encoder model."""

    def __call__(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> _EncoderOutputProto:
        """Run the encoder over one batch."""
        ...

    def eval(self) -> _EncoderProto:
        """Put the model in evaluation mode, returning itself."""
        ...


class _EncoderClassProto(Protocol):
    """The ``AutoModel`` class object."""

    def from_pretrained(self, model_id: str) -> _EncoderProto:
        """Load the encoder for a model id."""
        ...


def _default_embedder(texts: Sequence[str], /) -> torch.Tensor:
    """Production embedder - used as default hook.

    Imported inside the function for the reason the hub loaders are: parsing
    a command line must not pull transformers into the process. The
    ``__import__`` plus protocol-annotated assignment is the same shape
    ``hf_lm._test_hooks`` uses to narrow that untyped boundary.

    EVAL MODE MATTERS and is not decoration: gte carries dropout, and a
    train-mode forward pass would embed the same text differently on every
    call, which makes the ranking a function of nothing.

    Args:
        texts: Texts to embed.

    Returns:
        One L2-normalised row per input.
    """
    transformers = __import__("transformers", fromlist=["AutoModel", "AutoTokenizer"])
    tokenizer_cls: _TokenizerClassProto = transformers.AutoTokenizer
    model_cls: _EncoderClassProto = transformers.AutoModel

    tokenizer = tokenizer_cls.from_pretrained(DENSE_MODEL_ID)
    model = model_cls.from_pretrained(DENSE_MODEL_ID).eval()
    batch = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    mask = batch["attention_mask"]
    with torch.no_grad():
        hidden = model(input_ids=batch["input_ids"], attention_mask=mask).last_hidden_state
    return torch.nn.functional.normalize(masked_mean(hidden, mask), p=2.0, dim=1)


__all__ = [
    "DENSE_MODEL_ID",
    "EmbedderProto",
    "_default_embedder",
    "dense_ranking",
    "masked_mean",
    "rank_by_similarity",
]
