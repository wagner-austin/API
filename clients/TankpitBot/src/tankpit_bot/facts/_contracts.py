"""Fact-construction contracts.

The Phase 1 spec names three contracts on fact construction:

* ``NoUnsourcedFactContract`` -- every fact carries source,
  observed_ms, confidence, and provenance. Presence is structural
  (``make_fact`` takes keyword-only typed arguments); what remains a
  runtime question is the validity of ``observed_ms``.
* ``ConfidenceInBoundsContract`` -- confidence in [0.0, 1.0].
* ``ProvenanceRootednessContract`` -- observations may have an empty
  derivation list; inferences must cite prior sources.

The first two are composed into :class:`FactConstructionContract`;
each raise path carries its own error class and contract name, so the
named contracts stay individually identifiable in failures.
Rootedness needs the typed provenance chain and is enforced inside
:func:`tankpit_bot.facts.fact.make_fact` via ``require`` (the spec's
second enforcement mechanism).
"""

from __future__ import annotations

from tankpit_bot.contracts.base import NoUnsourcedFactError
from tankpit_bot.contracts.enforcement import require
from tankpit_bot.facts.confidence import require_confidence_in_bounds
from tankpit_bot.facts.source import FactSource


class FactConstructionContract:
    """Scalar-metadata contract for fact constructors."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "fact_construction"

    def check(self, *, source: FactSource, observed_ms: int, confidence: float) -> None:
        """Validate a fact's scalar metadata.

        Args:
            source: Channel the belief came from.
            observed_ms: When the belief was observed (or inferred).
            confidence: Trust in the belief.

        Raises:
            NoUnsourcedFactError: If observed_ms is negative.
            ConfidenceOutOfBoundsError: If confidence is out of range.
        """
        require(
            observed_ms >= 0,
            NoUnsourcedFactError,
            observed_ms=repr(observed_ms),
            source=source,
        )
        require_confidence_in_bounds(confidence)


__all__ = [
    "FactConstructionContract",
]
