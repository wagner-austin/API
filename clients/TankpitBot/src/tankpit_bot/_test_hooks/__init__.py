"""Test-hook surface for tankpit_bot.

Production code uses real implementations; tests override these
module-level symbols to inject fakes without conditionals in core
logic.

This package replaces the previous monolithic ``_test_hooks.py``.
Every public name remains importable from ``tankpit_bot._test_hooks``
exactly as before; the submodules group hooks by domain:

* :mod:`tankpit_bot._test_hooks.env` -- environment-variable resolution.
* :mod:`tankpit_bot._test_hooks.fs` -- filesystem operations.
* :mod:`tankpit_bot._test_hooks.cdp` -- Playwright Page / CDP / Keyboard
  / Response protocols.
* :mod:`tankpit_bot._test_hooks.browser` -- Playwright Browser /
  BrowserContext / BrowserType / Playwright protocols.
* :mod:`tankpit_bot._test_hooks.playwright_loader` --
  ``sync_playwright`` factory hook.
* :mod:`tankpit_bot._test_hooks.terrain` -- TerrainMap interface +
  loader hook.
* :mod:`tankpit_bot._test_hooks.bot` -- ``BotProtocol`` command-dispatch
  surface.
* :mod:`tankpit_bot._test_hooks.runtime` -- argv, static-byte discovery,
  replay dispatch.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from tankpit_bot._test_hooks.bot import BotProtocol, BufferedMessageSourceProtocol
from tankpit_bot._test_hooks.browser import (
    AutoscrollEnforcerProtocol,
    AutoscrollKeyProtocol,
    AutoscrollPageProtocol,
    BrowserContextProtocol,
    BrowserProtocol,
    BrowserTypeLaunchProtocol,
    BrowserTypeProtocol,
    PlaywrightProtocol,
    SyncPlaywrightContextManagerProtocol,
    SyncPlaywrightFactoryProtocol,
    _real_ensure_autoscroll_off,
    ensure_autoscroll_off,
)
from tankpit_bot._test_hooks.cdp import (
    CDPSessionProtocol,
    KeyboardProtocol,
    PageProtocol,
    ResponseProtocol,
)
from tankpit_bot._test_hooks.env import _default_get_env, get_env
from tankpit_bot._test_hooks.fs import (
    AppendTextProtocol,
    GlobPathsProtocol,
    PathExistsProtocol,
    ReadTextProtocol,
    RemoveFileProtocol,
    WriteBytesProtocol,
    WriteTextProtocol,
    _real_append_text,
    _real_glob_paths,
    _real_path_exists,
    _real_read_text,
    _real_remove_file,
    _real_write_bytes,
    _real_write_text,
    append_text,
    glob_paths,
    path_exists,
    read_text,
    remove_file,
    write_bytes,
    write_text,
)
from tankpit_bot._test_hooks.playwright_loader import (
    _real_get_sync_playwright,
    get_sync_playwright,
    sync_playwright,
)
from tankpit_bot._test_hooks.runtime import (
    FindBestStaticByteProtocol,
    InstallSignalHandlersProtocol,
    ProcessReceivedMessageProtocol,
    StartWatchdogProtocol,
    _real_get_argv,
    _real_get_current_time_ms,
    _real_install_signal_handlers,
    _real_process_received_message,
    _real_start_watchdog,
    find_best_static_byte,
    force_exit,
    get_argv,
    get_current_time_ms,
    install_signal_handlers,
    process_received_message_hook,
    start_watchdog,
)
from tankpit_bot._test_hooks.terrain import (
    LoadTerrainMapProtocol,
    TerrainMapProtocol,
    _real_load_terrain_map,
    load_terrain_map,
)

__all__ = [
    "AppendTextProtocol",
    "AutoscrollEnforcerProtocol",
    "AutoscrollKeyProtocol",
    "AutoscrollPageProtocol",
    "BotProtocol",
    "BrowserContextProtocol",
    "BrowserProtocol",
    "BrowserTypeLaunchProtocol",
    "BrowserTypeProtocol",
    "BufferedMessageSourceProtocol",
    "CDPSessionProtocol",
    "FindBestStaticByteProtocol",
    "GlobPathsProtocol",
    "InstallSignalHandlersProtocol",
    "KeyboardProtocol",
    "LoadTerrainMapProtocol",
    "PageProtocol",
    "PathExistsProtocol",
    "PlaywrightProtocol",
    "ProcessReceivedMessageProtocol",
    "ReadTextProtocol",
    "RemoveFileProtocol",
    "ResponseProtocol",
    "StartWatchdogProtocol",
    "SyncPlaywrightContextManagerProtocol",
    "SyncPlaywrightFactoryProtocol",
    "TerrainMapProtocol",
    "WriteBytesProtocol",
    "WriteTextProtocol",
    "_default_get_env",
    "_real_append_text",
    "_real_ensure_autoscroll_off",
    "_real_get_argv",
    "_real_get_current_time_ms",
    "_real_get_sync_playwright",
    "_real_glob_paths",
    "_real_install_signal_handlers",
    "_real_load_terrain_map",
    "_real_path_exists",
    "_real_process_received_message",
    "_real_read_text",
    "_real_remove_file",
    "_real_start_watchdog",
    "_real_write_bytes",
    "_real_write_text",
    "append_text",
    "ensure_autoscroll_off",
    "find_best_static_byte",
    "force_exit",
    "get_argv",
    "get_current_time_ms",
    "get_env",
    "get_sync_playwright",
    "glob_paths",
    "install_signal_handlers",
    "load_terrain_map",
    "path_exists",
    "process_received_message_hook",
    "read_text",
    "remove_file",
    "start_watchdog",
    "sync_playwright",
    "write_bytes",
    "write_text",
]
