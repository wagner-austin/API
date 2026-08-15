"""testing: _FAR_FUTURE_DEADLINE and related definitions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from platform_kaggle.types import (
    CodebaseProfile,
    KaggleApiFactoryProtocol,
    KaggleClientProtocol,
    KagglePageFetcherProtocol,
)

_FAR_FUTURE_DEADLINE = "2999-12-31"
_FAR_FUTURE_DEADLINE_DT = datetime(2999, 12, 31, 23, 59, 59)


# -----------------------------------------------------------------------------
# Hook Types
# -----------------------------------------------------------------------------

KaggleClientHook = Callable[[], KaggleClientProtocol]
PageFetcherHook = Callable[[], KagglePageFetcherProtocol]
ProfileScannerHook = Callable[[Path], CodebaseProfile]


# -----------------------------------------------------------------------------
# Hooks Container
# -----------------------------------------------------------------------------


class HooksContainer:
    """Container for dependency injection hooks.

    Attributes:
        kaggle_api_factory: Factory for Kaggle API (returns pre-authenticated api).
        kaggle_client: Factory for Kaggle client.
        page_fetcher: Factory for page fetcher.
        profile_scanner: Factory for codebase profile scanner.
    """

    kaggle_api_factory: KaggleApiFactoryProtocol
    kaggle_client: KaggleClientHook
    page_fetcher: PageFetcherHook
    profile_scanner: ProfileScannerHook


hooks = HooksContainer()


def _init_hooks() -> None:
    """Initialize hooks with production implementations."""
    from platform_kaggle._production import _get_kaggle_api, make_kaggle_client
    from platform_kaggle.capabilities import scan_codebase
    from platform_kaggle.internal_api import create_page_fetcher

    hooks.kaggle_api_factory = _get_kaggle_api
    hooks.kaggle_client = make_kaggle_client
    hooks.page_fetcher = create_page_fetcher
    hooks.profile_scanner = scan_codebase


def reset_hooks() -> None:
    """Reset hooks to production implementations (for test teardown)."""
    _init_hooks()


# Initialize hooks on module load
_init_hooks()
