"""Production implementations for Kaggle API access."""

from __future__ import annotations

from .types import KaggleApiProtocol, KaggleClientProtocol


def _get_kaggle_api() -> KaggleApiProtocol:
    """Return the pre-authenticated kaggle.api singleton.

    The kaggle module authenticates at import time using KAGGLE_API_TOKEN.
    """
    import kaggle

    api: KaggleApiProtocol = kaggle.api
    return api


def default_kaggle_api_factory() -> KaggleApiProtocol:
    """Factory that uses hooks.kaggle_api_factory.

    In production, returns kaggle.api. Tests can override via hooks.
    """
    from .testing import hooks

    return hooks.kaggle_api_factory()


def make_kaggle_client() -> KaggleClientProtocol:
    """Production factory for KaggleClient."""
    from .client import KaggleClient

    return KaggleClient()
