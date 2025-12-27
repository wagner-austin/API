"""Production implementations for Kaggle API access.

This module contains production implementations that use the hooks system
for testability. Tests can override hooks.kaggle_module to provide fakes.
"""

from __future__ import annotations

from .types import KaggleApiProtocol, KaggleClientProtocol


def create_kaggle_api() -> KaggleApiProtocol:
    """Create and authenticate real Kaggle API client.

    This function:
    1. Gets the kaggle module via hooks.kaggle_module
    2. Creates a KaggleApi instance
    3. Calls authenticate() which reads ~/.kaggle/kaggle.json

    Returns:
        Authenticated KaggleApi instance.

    Raises:
        SystemExit: If kaggle.json credentials file is not found.
    """
    from .testing import hooks

    kaggle_mod = hooks.kaggle_module()
    api: KaggleApiProtocol = kaggle_mod.KaggleApi()
    api.authenticate()
    return api


def default_kaggle_api_factory() -> KaggleApiProtocol:
    """Production factory for Kaggle API - calls create_kaggle_api."""
    return create_kaggle_api()


def make_kaggle_client() -> KaggleClientProtocol:
    """Production factory for KaggleClient."""
    from .client import KaggleClient

    return KaggleClient()
