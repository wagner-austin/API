"""Fact[T] projections over the raw world-state TypedDicts.

These read models sit ABOVE ``state/types`` and below nothing: they
turn a ``ContainerStateDict`` / ``TankStateDict`` / ``MineStateDict``
and friends into the ``Fact[T]`` view that carries source, timestamp,
confidence, and provenance.

They live here rather than under ``facts/`` because ``facts/`` is the
leaf vocabulary -- ``Fact``, ``FactSource``, provenance, confidence --
which every ``state/types`` module imports. Projections import
``state.types`` back, so keeping them in ``facts/`` made the two
packages mutually dependent. Splitting the vocabulary (leaf) from the
projections (read models) removes the cycle without changing either.

Submodules are imported directly; this file stays a docstring so the
package cannot become a second import surface.
"""

from __future__ import annotations
