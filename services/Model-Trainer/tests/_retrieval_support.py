"""The BM25 configuration these tests hold fixed, in one place.

``build_index`` takes its saturation, length normalisation and chunk cutoff as
required keyword arguments and has no defaults, deliberately: they used to be
module constants, and a retrieval arm reported without them is a comparison
against one arbitrary point in BM25's parameter space.

That is right for production and noisy for a test suite where nineteen call
sites want the same standard configuration and only a handful vary it. So the
standard values live here, named, and a test that VARIES one calls
``build_index`` directly with all three spelled out -- which is what makes the
variation visible in the test that performs it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from model_trainer.core.services.model.cartridge_retrieval import Bm25Index, build_index

#: BM25 term-frequency saturation, the standard value.
STANDARD_K1: Final[float] = 1.5

#: BM25 length normalisation, the standard value.
STANDARD_B: Final[float] = 0.75

#: Chunks returned per query. Five rather than one because the oracle arm
#: concatenates EVERY sentence containing the answer, so a single-chunk
#: retriever would lose on evidence volume rather than on selection.
STANDARD_RETRIEVED_CHUNKS: Final[int] = 5


def standard_index(
    documents: Sequence[str], *, retrieved_chunks: int = STANDARD_RETRIEVED_CHUNKS
) -> Bm25Index:
    """Index a corpus under the configuration these tests hold fixed.

    ONE FUNCTION RATHER THAN TWO. A separate ``index_returning`` that differed
    from this only in passing a cutoff through would be a wrapper with no
    content of its own, and the keyword here says the same thing where the
    caller can see it. The default is a named constant rather than a literal,
    so the standard configuration is stated once.

    The cutoff is a property of the INDEX because it is a property of the ARM
    being measured rather than of one lookup, which is also why a plan
    sweeping ``retrieved_chunks`` builds a different index per cell.

    Args:
        documents: Document bodies to index.
        retrieved_chunks: How many chunks a query should return.

    Returns:
        The index, built with the standard saturation and normalisation.
    """
    return build_index(documents, k1=STANDARD_K1, b=STANDARD_B, retrieved_chunks=retrieved_chunks)
