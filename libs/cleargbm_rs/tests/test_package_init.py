"""Tests for the ``cleargbm_rs`` package entry point.

Exercises the :pep:`562` forwarding layer against the real compiled
extension — no stubs and no doubles, so a drift between this package and the
Rust module surfaces here.
"""

from __future__ import annotations

import cleargbm_rs
import pytest


def test_extension_module_loads() -> None:
    """The compiled extension imports under its own dotted name."""
    extension = cleargbm_rs._extension()

    assert extension.__name__ == "cleargbm_rs.cleargbm_rs"


def test_extension_import_is_cached() -> None:
    """Repeated resolution returns the same module object."""
    assert cleargbm_rs._extension() is cleargbm_rs._extension()


def test_forwards_a_training_function() -> None:
    """A known Rust function resolves through package attribute access."""
    resolved = cleargbm_rs.train_gradient_boosting_rs

    assert callable(resolved)


def test_forwards_a_model_class() -> None:
    """A known Rust class resolves through package attribute access."""
    resolved = cleargbm_rs.PyGbmModel

    assert resolved.__name__ == "PyGbmModel"


def test_forwarding_returns_a_stable_object() -> None:
    """Repeated access yields the same object, not a fresh wrapper each time.

    Consumers bind these once at import (see ``cleargbm._rust``), so identity
    across accesses is what makes that binding safe.
    """
    first = cleargbm_rs.predict_proba_model_rs
    second = cleargbm_rs.predict_proba_model_rs

    assert first is second


def test_unknown_attribute_raises() -> None:
    """An attribute the extension does not export propagates AttributeError."""
    with pytest.raises(AttributeError):
        _ = cleargbm_rs.definitely_not_a_rust_function


def test_package_exports_nothing_implicitly() -> None:
    """``__all__`` is empty: consumers name what they import.

    The package deliberately publishes no star-import surface, so adding a
    Rust function cannot silently change what ``from cleargbm_rs import *``
    pulls in.
    """
    assert cleargbm_rs.__all__ == []
