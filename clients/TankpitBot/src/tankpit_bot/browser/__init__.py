"""Browser session management module.

This module provides browser-based WebSocket capture capabilities,
organized into submodules:

- types: Constants, error types
- key_discovery: Static XOR key extraction
- session: BrowserSession base class
- login: Guest and account login logic
- dom_scraper: DOM scraping for game log capture
- fuel_probe: Fuel/HP value probing via DOM
"""

from __future__ import annotations

from tankpit_bot.browser.accounts import (
    Account,
    AccountNotFoundError,
    decode_account,
    decode_account_list,
    encode_account,
    encode_account_list,
    load_accounts,
    resolve_account,
    select_account,
)
from tankpit_bot.browser.cdp_utils import (
    cdp_timestamp_to_ms,
    get_current_time_ms,
    reset_cdp_time_offset,
)
from tankpit_bot.browser.dom_scraper import (
    GameLogEntry,
    GameLogScraper,
    LogCategory,
    decode_game_log_entry,
    encode_game_log_entry,
)
from tankpit_bot.browser.fuel_probe import (
    DOMBarElement,
    FuelProber,
    FuelProbeResult,
    JSVariable,
    probe_all,
)
from tankpit_bot.browser.key_discovery import (
    extract_xor_first_bytes,
    find_best_static_byte,
    load_static_key,
    save_static_key,
)
from tankpit_bot.browser.login import (
    AccountLoginResult,
    GuestLoginResult,
    handle_account_login,
    handle_guest_login,
    handle_login_flow,
)
from tankpit_bot.browser.session import BrowserSession
from tankpit_bot.browser.types import (
    KNOWN_PROTOCOL_SIGNATURES,
    STATIC_KEY_LENGTH,
    STATIC_KEY_PATH,
    TEXT_MESSAGE_TYPES,
    BrowserError,
    GameNotJoinedError,
    PlaywrightNotInstalledError,
)

__all__ = [
    "KNOWN_PROTOCOL_SIGNATURES",
    "STATIC_KEY_LENGTH",
    "STATIC_KEY_PATH",
    "TEXT_MESSAGE_TYPES",
    "Account",
    "AccountLoginResult",
    "AccountNotFoundError",
    "BrowserError",
    "BrowserSession",
    "DOMBarElement",
    "FuelProbeResult",
    "FuelProber",
    "GameLogEntry",
    "GameLogScraper",
    "GameNotJoinedError",
    "GuestLoginResult",
    "JSVariable",
    "LogCategory",
    "PlaywrightNotInstalledError",
    "cdp_timestamp_to_ms",
    "decode_account",
    "decode_account_list",
    "decode_game_log_entry",
    "encode_account",
    "encode_account_list",
    "encode_game_log_entry",
    "extract_xor_first_bytes",
    "find_best_static_byte",
    "get_current_time_ms",
    "handle_account_login",
    "handle_guest_login",
    "handle_login_flow",
    "load_accounts",
    "load_static_key",
    "probe_all",
    "reset_cdp_time_offset",
    "resolve_account",
    "save_static_key",
    "select_account",
]
