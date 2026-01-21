"""Fake Playwright classes for testing.

Provides fake implementations of Playwright protocols that don't require
real browser installation. All fakes match the protocol signatures in
tankpit_bot._test_hooks.

This package is split into modules:
- base: Core fakes for general testing
- probe: Probe-specific fakes
- bot: Bot-specific fakes

All classes and functions are re-exported here for backward compatibility.
"""

from tests.fakes.base import (
    FakeBrowser,
    FakeBrowserContext,
    FakeBrowserType,
    FakeCDPSession,
    FakeCDPSessionRateLimited,
    FakeKeyboard,
    FakePage,
    FakePageNoMessages,
    FakePlaywright,
    FakeResponse,
    FakeSyncPlaywrightContextManager,
    FakeTerrainMap,
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
from tests.fakes.probe import (
    FakeBrowserContextProbe,
    FakeBrowserProbe,
    FakeBrowserTypeProbe,
    FakeCDPSessionProbe,
    FakePageProbe,
    FakePageProbeNoMessages,
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
    "FakeTerrainMap",
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
