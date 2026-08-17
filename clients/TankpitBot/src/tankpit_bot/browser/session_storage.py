"""Persistent Playwright storage-state for the bot's Chromium session.

The bot's first launch runs the full tankpit login flow (guest join or
account credentials). Storing Playwright's ``storage_state``
(cookies + localStorage) between launches lets the second-and-beyond
launch re-use the auth Chromium already established: the ``navigate_and_login``
call sees a signed-in state and short-circuits the credential entry.

* **Load** returns a filesystem path :meth:`BrowserProtocol.new_context`
  can pass to Playwright as its ``storage_state`` argument, or ``None``
  when no cached state is available. The bot flow keeps the file if
  present + parseable, and hard-fails on the two "cache is corrupt"
  edge cases — empty file and malformed JSON — so a broken cache
  produces a loud error instead of a silent auth-flow loop.
* **Save** captures the current context's storage state via
  :meth:`BrowserContextProtocol.storage_state`, serialises it with
  :func:`platform_core.json_utils.dump_json_str`, and writes it
  through :mod:`tankpit_bot._test_hooks.fs.write_text` so tests can
  inject a fake file system.

Callers pass in the path explicitly rather than reading a module-level
constant so tests + multiple bot flavours can point at different
cache files without global mutation.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import dump_json_str, load_json_str

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import BrowserContextProtocol
from tankpit_bot.browser.accounts import resolve_account

_STORAGE_STATE_DIR: Path = Path("runs/state")
"""Directory holding the per-identity storage-state caches.

Created on demand by :func:`save_storage_state`."""


def resolve_storage_state_path(prefer_account: bool) -> Path:
    """Return the storage-state cache path for the session's login identity.

    The cache MUST be keyed by who the cookies belong to. A single
    shared file made every instance resume whatever account logged in
    first: on 2026-08-13 a fleet child spawned with
    ``TANKPIT_ACCOUNT=Arterial`` found the shared jar holding Artax's
    session, skipped the credential flow entirely (its log shows a
    straight ``Navigated to /play`` with no login), joined as a second
    Artax, and the server cut its socket two seconds later. Keying the
    path by the RESOLVED account makes the cache self-consistent: a
    child selecting a different account finds no cache and runs the
    real login.

    Args:
        prefer_account: The session's account-vs-guest preference
            (``resolve_prefer_account()``); guest sessions share the
            ``guest`` identity.

    Returns:
        ``runs/state/tankpit.<identity>.storage.json`` where identity
        is the resolved account username (lowered, sanitized to the
        instance-name grammar) or ``guest``.

    Raises:
        AccountNotFoundError: If ``TANKPIT_ACCOUNT`` names an account
            that does not exist — the same error the login flow would
            raise later, surfaced before a wrong jar can be read.
    """
    identity = "guest"
    if prefer_account:
        account = resolve_account()
        if account is not None:
            cleaned = "".join(
                ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "-"
                for ch in account["username"].lower()
            )
            identity = cleaned or "guest"
    return _STORAGE_STATE_DIR / f"tankpit.{identity}.storage.json"


class StorageStateCacheError(ValueError):
    """Raised when the cached storage-state file exists but is unusable.

    Distinguished from a missing file — which just means "no cache
    yet, first launch" — so callers can surface the corruption
    without masking a legitimate cold start.
    """


def load_storage_state(path: Path) -> str | None:
    """Validate the cached storage-state and return its path for Playwright.

    Args:
        path: Cache-file location on disk.

    Returns:
        ``str(path)`` when the file exists and parses as JSON — the
        bot passes this to :meth:`BrowserProtocol.new_context`'s
        ``storage_state`` kwarg. ``None`` when the file is absent —
        the bot launches fresh and the next :func:`save_storage_state`
        creates the cache.

    Raises:
        StorageStateCacheError: The file exists but is empty or the
            content does not parse as JSON. This is a hard failure so
            a broken cache does not silently downgrade to a login
            flow on every launch.
    """
    if not _test_hooks.path_exists(path):
        return None
    content = _test_hooks.read_text(path)
    if not content.strip():
        raise StorageStateCacheError(f"storage state file is empty: {path}")
    _validate_storage_state_json(content, path)
    return str(path)


def _validate_storage_state_json(content: str, path: Path) -> None:
    """Confirm ``content`` parses as JSON, re-raising with the cache path.

    Args:
        content: File body read from ``path``.
        path: Cache-file location, embedded in the raised message so
            operators know which file to nuke.

    Raises:
        StorageStateCacheError: When ``content`` is not valid JSON.
            The underlying ``InvalidJsonError`` is chained via ``from``
            so the trace stays readable.
    """
    from platform_core.json_utils import InvalidJsonError

    try:
        load_json_str(content)
    except InvalidJsonError as exc:
        raise StorageStateCacheError(f"storage state file is not valid JSON: {path}") from exc


def save_storage_state(context: BrowserContextProtocol, path: Path) -> None:
    """Snapshot the browser context's storage state to ``path``.

    Called from :meth:`Bot.run` immediately after
    :func:`wait_for_game_ready` succeeds so the freshly-issued auth
    cookies land on disk before the game loop can crash. The write
    goes through :func:`tankpit_bot._test_hooks.fs.write_text` so a
    :class:`~tests.conftest.FakeFileSystem` captures it without
    hitting the real disk.

    Args:
        context: Live Playwright context to snapshot.
        path: Destination file. Parent-directory creation is handled
            by :func:`_real_write_text`, so the first-ever save on a
            fresh checkout succeeds without a separate ``mkdir`` call.
    """
    payload = dump_json_str(context.storage_state(), compact=False, indent=2)
    _test_hooks.write_text(path, payload)


__all__ = [
    "StorageStateCacheError",
    "load_storage_state",
    "resolve_storage_state_path",
    "save_storage_state",
]
