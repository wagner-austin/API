"""Playwright factory loader hook.

``sync_playwright`` and ``get_sync_playwright`` are the only entry
points where ``playwright.sync_api`` is imported. Splitting them off
keeps every other module free of a direct Playwright dependency, so
tests can run without Playwright installed at all.
"""

from __future__ import annotations

from collections.abc import Callable

from tankpit_bot._test_hooks.browser import SyncPlaywrightFactoryProtocol


def _real_get_sync_playwright() -> SyncPlaywrightFactoryProtocol:
    """Real implementation - imports playwright.

    Returns:
        The sync_playwright factory from playwright.sync_api.
    """
    pw_module = __import__("playwright.sync_api", fromlist=["sync_playwright"])
    real_sync_playwright: SyncPlaywrightFactoryProtocol = pw_module.sync_playwright
    return real_sync_playwright


sync_playwright: SyncPlaywrightFactoryProtocol | None = None
"""Optional override that short-circuits the loader. Tests inject a fake
factory here; production leaves it ``None`` so ``get_sync_playwright``
performs the live import on first use."""

get_sync_playwright: Callable[[], SyncPlaywrightFactoryProtocol] = _real_get_sync_playwright


__all__ = [
    "get_sync_playwright",
    "sync_playwright",
]
