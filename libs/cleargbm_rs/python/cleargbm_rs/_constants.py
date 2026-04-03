"""Shared constants for cleargbm_rs Python stubs.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

NOT_BUILT_MSG: str = "Rust extension not built. Run: maturin develop --features extension-module"

__all__ = [
    "NOT_BUILT_MSG",
]
