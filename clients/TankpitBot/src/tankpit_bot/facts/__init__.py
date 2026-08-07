"""Facts vocabulary: every belief carries source, time, confidence, provenance.

Phase 1 of the self-observing bot architecture. ``Fact[T]`` is
introduced alongside the raw world-state types (Phase 1a); the
retrofits of ``ContainerStateDict`` / ``TankStateDict`` / the rest are
Phases 1b-1d.

This package is a LEAF: it defines the vocabulary and imports no other
``tankpit_bot`` package. The Fact[T] projections that read world state
back live in :mod:`tankpit_bot.state.projections`, above ``state``,
because ``state/types`` imports this vocabulary.
"""

from tankpit_bot.facts.confidence import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    combine_independent,
    combine_weighted,
    decay_by_age,
    require_confidence_in_bounds,
)
from tankpit_bot.facts.fact import Fact, decode_fact, encode_fact, make_fact
from tankpit_bot.facts.provenance import (
    ProvenanceChainDict,
    SourceRefDict,
    decode_provenance,
    decode_source_ref,
    encode_provenance,
    encode_source_ref,
    make_provenance,
    make_source_ref,
)
from tankpit_bot.facts.source import (
    FACT_SOURCES,
    INFERENCE_SOURCE,
    FactSource,
    is_observation_source,
    require_fact_source,
)

__all__ = [
    "CONFIDENCE_MAX",
    "CONFIDENCE_MIN",
    "FACT_SOURCES",
    "INFERENCE_SOURCE",
    "Fact",
    "FactSource",
    "ProvenanceChainDict",
    "SourceRefDict",
    "combine_independent",
    "combine_weighted",
    "decay_by_age",
    "decode_fact",
    "decode_provenance",
    "decode_source_ref",
    "encode_fact",
    "encode_provenance",
    "encode_source_ref",
    "is_observation_source",
    "make_fact",
    "make_provenance",
    "make_source_ref",
    "require_confidence_in_bounds",
    "require_fact_source",
]
