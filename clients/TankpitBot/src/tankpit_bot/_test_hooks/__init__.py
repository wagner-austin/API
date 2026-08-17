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
* :mod:`tankpit_bot._test_hooks.bot` -- the buffered-message-source
  surface.
* :mod:`tankpit_bot._test_hooks.runtime` -- argv, static-byte discovery,
  watchdog, signals. Replay decode moved to
  :mod:`tankpit_bot.replay._test_hooks` (step 8).

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from tankpit_bot._test_hooks.bot import BufferedMessageSourceProtocol
from tankpit_bot._test_hooks.browser import (
    BrowserContextProtocol,
    BrowserProtocol,
    BrowserTypeLaunchProtocol,
    BrowserTypeProtocol,
    GamePageProtocol,
    PageKeyboardProtocol,
    PageWaitProtocol,
    PlaywrightProtocol,
    RoomJoinPageProtocol,
    SyncPlaywrightContextManagerProtocol,
    SyncPlaywrightFactoryProtocol,
)
from tankpit_bot._test_hooks.cdp import (
    CDPSessionProtocol,
    KeyboardProtocol,
    PageProtocol,
    ResponseProtocol,
)
from tankpit_bot._test_hooks.env import (
    LoadDotenvProtocol,
    _default_get_env,
    _real_load_dotenv,
    get_env,
    load_dotenv,
)
from tankpit_bot._test_hooks.fs import (
    AppendTextProtocol,
    GlobPathsProtocol,
    PathExistsProtocol,
    ReadTextProtocol,
    RemoveFileProtocol,
    ReplaceTextProtocol,
    WriteBytesProtocol,
    WriteTextProtocol,
    _real_append_text,
    _real_glob_paths,
    _real_path_exists,
    _real_read_text,
    _real_remove_file,
    _real_replace_text,
    _real_write_bytes,
    _real_write_text,
    append_text,
    glob_paths,
    path_exists,
    read_text,
    remove_file,
    replace_text,
    write_bytes,
    write_text,
)
from tankpit_bot._test_hooks.playwright_loader import (
    _real_get_sync_playwright,
    get_sync_playwright,
    sync_playwright,
)
from tankpit_bot._test_hooks.runtime import (
    InstallSignalHandlersProtocol,
    StartWatchdogProtocol,
    _real_get_argv,
    _real_get_current_time_ms,
    _real_install_signal_handlers,
    _real_start_watchdog,
    force_exit,
    get_argv,
    get_current_time_ms,
    install_signal_handlers,
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
    "BrowserContextProtocol",
    "BrowserProtocol",
    "BrowserTypeLaunchProtocol",
    "BrowserTypeProtocol",
    "BufferedMessageSourceProtocol",
    "CDPSessionProtocol",
    "GamePageProtocol",
    "GlobPathsProtocol",
    "InstallSignalHandlersProtocol",
    "KeyboardProtocol",
    "LoadDotenvProtocol",
    "LoadTerrainMapProtocol",
    "PageKeyboardProtocol",
    "PageProtocol",
    "PageWaitProtocol",
    "PathExistsProtocol",
    "PlaywrightProtocol",
    "ReadTextProtocol",
    "RemoveFileProtocol",
    "ReplaceTextProtocol",
    "ResponseProtocol",
    "RoomJoinPageProtocol",
    "StartWatchdogProtocol",
    "SyncPlaywrightContextManagerProtocol",
    "SyncPlaywrightFactoryProtocol",
    "TerrainMapProtocol",
    "WriteBytesProtocol",
    "WriteTextProtocol",
    "_default_get_env",
    "_real_append_text",
    "_real_get_argv",
    "_real_get_current_time_ms",
    "_real_get_sync_playwright",
    "_real_glob_paths",
    "_real_install_signal_handlers",
    "_real_load_dotenv",
    "_real_load_terrain_map",
    "_real_path_exists",
    "_real_read_text",
    "_real_remove_file",
    "_real_replace_text",
    "_real_start_watchdog",
    "_real_write_bytes",
    "_real_write_text",
    "append_text",
    "force_exit",
    "get_argv",
    "get_current_time_ms",
    "get_env",
    "get_sync_playwright",
    "glob_paths",
    "install_signal_handlers",
    "load_dotenv",
    "load_terrain_map",
    "path_exists",
    "read_text",
    "remove_file",
    "replace_text",
    "start_watchdog",
    "sync_playwright",
    "write_bytes",
    "write_text",
]
