"""ClearGBM Rust core — Python entry point for the compiled extension.

The compiled extension module (``cleargbm_rs.cleargbm_rs``) carries every real
implementation. Attribute access on this package is forwarded to it on demand
via :pep:`562`, so the package exposes exactly what the extension exports with
no hand-maintained signature list to drift out of sync with the Rust source.

Deliberately untyped here. Consumers bind the symbols they use to
Protocol-typed names at their own boundary -- see ``cleargbm._rust``, which
resolves each function through ``__import__`` and annotates it at the
assignment site. That keeps the type contract in one place, next to the code
that depends on it, rather than duplicated into stub modules that have to be
edited in lockstep with the Rust signatures.
"""

from __future__ import annotations

import types

#: Dotted path of the compiled extension module built by maturin.
_EXTENSION_MODULE = "cleargbm_rs.cleargbm_rs"


def _extension() -> types.ModuleType:
    """Import the compiled extension module.

    Returns:
        The loaded extension module. The import is cached by ``sys.modules``,
        so repeated calls cost a dictionary lookup.

    Raises:
        ImportError: If the extension has not been built. Build it with
            ``maturin develop --release`` from ``libs/cleargbm_rs``.
    """
    return __import__(_EXTENSION_MODULE, fromlist=["*"])


def __getattr__(name: str) -> types.FunctionType | type:
    """Resolve an attribute from the compiled extension module.

    Args:
        name: Attribute name, such as ``train_gradient_boosting_rs``.

    Returns:
        The function or class the extension exports under ``name``.

    Raises:
        AttributeError: If the extension exports no such attribute.
    """
    resolved: types.FunctionType | type = getattr(_extension(), name)
    return resolved


__all__: list[str] = []
