"""Tests for the driven adapters.

These are the only tests that import a vendor. They exist in two layers: a drift
layer that re-reads the installed API on every run and fails when it stops
matching the Protocols declared for it, and a behaviour layer that drives the
real simulator through the probe's port.
"""

from __future__ import annotations
