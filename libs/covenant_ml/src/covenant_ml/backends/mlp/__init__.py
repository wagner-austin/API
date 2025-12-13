"""Namespace for MLP backend.

Exports factory symbol to be used by the registry. Implementation will be
provided by backend.py in this package.
"""

from __future__ import annotations

from .backend import MLP_CAPABILITIES, create_mlp_backend

__all__ = ["MLP_CAPABILITIES", "create_mlp_backend"]
