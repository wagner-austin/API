"""Strategic-map (0x4C) physics: the fuel-dot exposure law."""

from __future__ import annotations

MAP_DOT_MIN_VOLUME = 500
"""Minimum fuel volume for an exposed container to join the map atlas.

Wiki: [[game-economy]] claim ``map-dot-min-volume`` and
[[map-data-decode]] — a fuel container becomes a yellow map dot when
it is EXPOSED (radar/viewport reveal) while holding at least this
volume; the dot then persists as exposure memory while the container
drains. Measured 2026-07-25 over 223 archive sessions: 0 of 163
sub-500 reveals ever joined the atlas, the 500-509 band joins, and
all 605 within-session dot appearances were exposure-preceded.
"""

__all__ = ["MAP_DOT_MIN_VOLUME"]
