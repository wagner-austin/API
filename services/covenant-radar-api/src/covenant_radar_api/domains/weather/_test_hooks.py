"""Test hooks for weather domain.

The WeatherFeatureExtractor is pure computation with no external
dependencies. State is injected via constructor, so tests construct
extractors directly with fake TemporalFeatureState data.

No hookable dependencies exist in this module.
"""

from __future__ import annotations

__all__: list[str] = []
