"""Fake Playwright classes for testing.

Provides fake implementations of Playwright protocols that don't require
real browser installation. All fakes match the protocol signatures in
tankpit_bot._test_hooks.

This package is split into modules:
- payloads: canned wire payloads and page-runtime results
- terrain: the in-memory terrain map
- cdp: fake CDP sessions
- page: fake page, keyboard, and response
- base: the Playwright browser stack and sync-context factories
- probe: Probe-specific fakes
- bot: Bot-specific fakes
"""

from tests.fakes.base import (
    FakeBrowser,
    FakeBrowserContext,
    FakeBrowserType,
    FakePlaywright,
    FakeSyncPlaywrightContextManager,
    fake_sync_playwright,
    fake_sync_playwright_login_fails,
    fake_sync_playwright_no_messages,
    fake_sync_playwright_rate_limited,
    fake_sync_playwright_with_magic,
    fake_sync_playwright_with_mixed_scripts,
    fake_sync_playwright_with_scripts,
)
from tests.fakes.bot import (
    FakeBrowserBot,
    FakeBrowserContextBot,
    FakeBrowserTypeBot,
    FakeCDPSessionBot,
    FakePageBot,
    FakePageInterrupting,
    FakePlaywrightBot,
    FakeSyncPlaywrightContextManagerBot,
    fake_sync_playwright_bot,
)
from tests.fakes.cdp import (
    FakeCDPSession,
    FakeCDPSessionRateLimited,
)
from tests.fakes.page import (
    FakeKeyboard,
    FakePage,
    FakePageGrowingMessages,
    FakePageNoMessages,
    FakeResponse,
)
from tests.fakes.probe import (
    FakeBrowserContextProbe,
    FakeBrowserProbe,
    FakeBrowserTypeProbe,
    FakePlaywrightProbe,
    FakeSyncPlaywrightContextManagerProbe,
    fake_sync_playwright_probe,
    fake_sync_playwright_probe_before_playing,
    fake_sync_playwright_probe_both_emit,
    fake_sync_playwright_probe_delayed_messages,
    fake_sync_playwright_probe_invalid_viewport,
    fake_sync_playwright_probe_mouse_emits,
    fake_sync_playwright_probe_no_key_emits,
    fake_sync_playwright_probe_no_messages,
    fake_sync_playwright_probe_non_dict_viewport,
)
from tests.fakes.probe_cdp import FakeCDPSessionProbe
from tests.fakes.probe_page import (
    FakePageProbe,
    FakePageProbeNoMessages,
)
from tests.fakes.terrain import (
    InMemoryTerrainMap,
)

__all__ = [
    "FakeBrowser",
    "FakeBrowserBot",
    "FakeBrowserContext",
    "FakeBrowserContextBot",
    "FakeBrowserContextProbe",
    "FakeBrowserProbe",
    "FakeBrowserType",
    "FakeBrowserTypeBot",
    "FakeBrowserTypeProbe",
    "FakeCDPSession",
    "FakeCDPSessionBot",
    "FakeCDPSessionProbe",
    "FakeCDPSessionRateLimited",
    "FakeKeyboard",
    "FakePage",
    "FakePageBot",
    "FakePageGrowingMessages",
    "FakePageInterrupting",
    "FakePageNoMessages",
    "FakePageProbe",
    "FakePageProbeNoMessages",
    "FakePlaywright",
    "FakePlaywrightBot",
    "FakePlaywrightProbe",
    "FakeResponse",
    "FakeSyncPlaywrightContextManager",
    "FakeSyncPlaywrightContextManagerBot",
    "FakeSyncPlaywrightContextManagerProbe",
    "InMemoryTerrainMap",
    "fake_sync_playwright",
    "fake_sync_playwright_bot",
    "fake_sync_playwright_login_fails",
    "fake_sync_playwright_no_messages",
    "fake_sync_playwright_probe",
    "fake_sync_playwright_probe_before_playing",
    "fake_sync_playwright_probe_both_emit",
    "fake_sync_playwright_probe_delayed_messages",
    "fake_sync_playwright_probe_invalid_viewport",
    "fake_sync_playwright_probe_mouse_emits",
    "fake_sync_playwright_probe_no_key_emits",
    "fake_sync_playwright_probe_no_messages",
    "fake_sync_playwright_probe_non_dict_viewport",
    "fake_sync_playwright_rate_limited",
    "fake_sync_playwright_with_magic",
    "fake_sync_playwright_with_mixed_scripts",
    "fake_sync_playwright_with_scripts",
]
