"""Test hooks for scripts.

Production code uses real implementations; tests override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from tankpit_bot.decoder import DecodedCommand, DecodedLobbyMessage

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class PathExistsProtocol(Protocol):
    """Protocol for checking if a path exists."""

    def __call__(self, path: Path) -> bool:
        """Check if path exists.

        Args:
            path: Path to check.

        Returns:
            True if path exists, False otherwise.
        """
        ...


def _real_path_exists(path: Path) -> bool:
    """Real implementation using Path.exists().

    Args:
        path: Path to check.

    Returns:
        True if path exists, False otherwise.
    """
    return path.exists()


path_exists: PathExistsProtocol = _real_path_exists


class ReadTextProtocol(Protocol):
    """Protocol for reading text from a file."""

    def __call__(self, path: Path) -> str:
        """Read text from a file.

        Args:
            path: Path to the file.

        Returns:
            File contents as string.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        ...


def _real_read_text(path: Path) -> str:
    """Real implementation using Path.read_text().

    Args:
        path: Path to the file.

    Returns:
        File contents as string.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    return path.read_text(encoding="utf-8")


read_text: ReadTextProtocol = _real_read_text


class SessionDecoderProtocol(Protocol):
    """Protocol for session decoder interface."""

    @property
    def commands(self) -> list[DecodedCommand]:
        """Get decoded commands."""
        ...

    @property
    def lobby_messages(self) -> list[DecodedLobbyMessage]:
        """Get decoded lobby messages."""
        ...


# Type alias for load_and_decode_session hook
LoadAndDecodeSessionFunc = Callable[[Path], SessionDecoderProtocol]


def _real_load_and_decode_session(session_path: Path) -> SessionDecoderProtocol:
    """Real implementation using decoder module.

    Args:
        session_path: Path to session JSON file.

    Returns:
        SessionDecoder with decoded messages.

    Raises:
        MissingMagicError: If session has no magic key.
        FileNotFoundError: If file doesn't exist.
    """
    from tankpit_bot.decoder import load_and_decode_session as real_load

    return real_load(session_path)


load_and_decode_session: LoadAndDecodeSessionFunc = _real_load_and_decode_session


class SetupRichLoggingProtocol(Protocol):
    """Protocol for setting up rich logging."""

    def __call__(self, level: LogLevel) -> None:
        """Set up rich logging.

        Args:
            level: Log level string.
        """
        ...


def _real_setup_rich_logging(level: LogLevel) -> None:
    """Real implementation using platform_core.

    Args:
        level: Log level string.
    """
    from platform_core.logging import setup_rich_logging as real_setup

    real_setup(level=level)


setup_rich_logging: SetupRichLoggingProtocol = _real_setup_rich_logging


class HttpGetResponseProtocol(Protocol):
    """Protocol for an HTTP GET response."""

    @property
    def status_code(self) -> int:
        """HTTP status code."""
        ...

    @property
    def content(self) -> bytes:
        """Response body bytes."""
        ...


class HttpGetProtocol(Protocol):
    """Protocol for performing HTTP GET requests."""

    def __call__(self, url: str) -> HttpGetResponseProtocol:
        """Perform an HTTP GET request.

        Args:
            url: URL to fetch.

        Returns:
            Response with status_code and content.
        """
        ...


def _real_http_get(url: str) -> HttpGetResponseProtocol:
    """Real implementation using httpx.

    Args:
        url: URL to fetch.

    Returns:
        httpx Response.
    """
    httpx_mod = __import__("httpx")
    response: HttpGetResponseProtocol = httpx_mod.get(url, timeout=30.0, follow_redirects=True)
    return response


http_get: HttpGetProtocol = _real_http_get


class ResolveTreeHashProtocol(Protocol):
    """Protocol for resolving a repo path to its committed object id."""

    def __call__(self, project_root: Path, repo_path: str) -> str | None:
        """Resolve one path to the object id HEAD records for it.

        Args:
            project_root: Repository directory to resolve within.
            repo_path: Path relative to ``project_root``.

        Returns:
            The 40-hex tree or blob id, or None when the path is not in
            HEAD (untracked, deleted, or outside a git work tree).
        """
        ...


def _real_resolve_tree_hash(project_root: Path, repo_path: str) -> str | None:
    """Real implementation shelling out to ``git rev-parse``.

    Args:
        project_root: Repository directory to resolve within.
        repo_path: Path relative to ``project_root``.

    Returns:
        The 40-hex tree or blob id, or None when git cannot resolve it.
    """
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", f"HEAD:./{repo_path}"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


resolve_tree_hash: ResolveTreeHashProtocol = _real_resolve_tree_hash


class RunForProjectProtocol(Protocol):
    """Protocol for the orchestrator's ``run_for_project`` function."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guards for a project.

        Args:
            monorepo_root: Root of the monorepo.
            project_root: Root of the project to check.

        Returns:
            Exit code, 0 when no violations were found.
        """
        ...


class IsDirProtocol(Protocol):
    """Protocol for checking whether a path is a directory."""

    def __call__(self, path: Path) -> bool:
        """Check whether a path is a directory.

        Args:
            path: Path to check.

        Returns:
            True when the path is a directory, False otherwise.
        """
        ...


class GetScriptPathProtocol(Protocol):
    """Protocol for resolving the guard script's own path."""

    def __call__(self) -> Path:
        """Resolve the guard script path.

        Returns:
            Absolute path to the guard script.

        Raises:
            RuntimeError: When the script path has not been set.
        """
        ...


class LoadOrchestratorProtocol(Protocol):
    """Protocol for loading the monorepo guard orchestrator."""

    def __call__(self, monorepo_root: Path) -> RunForProjectProtocol:
        """Load the orchestrator.

        Args:
            monorepo_root: Root of the monorepo.

        Returns:
            The orchestrator's ``run_for_project`` function.
        """
        ...


def _real_is_dir(path: Path) -> bool:
    """Real implementation using Path.is_dir().

    Args:
        path: Path to check.

    Returns:
        True when the path is a directory, False otherwise.
    """
    return path.is_dir()


_SCRIPT_PATH: Path | None = None


def set_script_path(path: Path) -> None:
    """Record the guard script's path for production use.

    Args:
        path: Absolute path to the guard script.
    """
    global _SCRIPT_PATH
    _SCRIPT_PATH = path


def _real_get_script_path() -> Path:
    """Real implementation returning the recorded guard script path.

    Returns:
        Absolute path to the guard script.

    Raises:
        RuntimeError: When the script path has not been set.
    """
    if _SCRIPT_PATH is None:
        raise RuntimeError("Script path not set - call set_script_path first")
    return _SCRIPT_PATH


def _real_load_orchestrator(monorepo_root: Path) -> RunForProjectProtocol:
    """Real implementation importing the orchestrator from the monorepo.

    Args:
        monorepo_root: Root of the monorepo.

    Returns:
        The orchestrator's ``run_for_project`` function.
    """
    libs_path = monorepo_root / "libs"
    guards_src = libs_path / "monorepo_guards" / "src"
    sys.path.insert(0, str(guards_src))
    sys.path.insert(0, str(libs_path))
    mod = __import__("monorepo_guards.orchestrator", fromlist=["run_for_project"])
    run_for_project: RunForProjectProtocol = mod.run_for_project
    return run_for_project


is_dir: IsDirProtocol = _real_is_dir
get_script_path: GetScriptPathProtocol = _real_get_script_path
load_orchestrator: LoadOrchestratorProtocol = _real_load_orchestrator


__all__ = [
    "GetScriptPathProtocol",
    "HttpGetProtocol",
    "HttpGetResponseProtocol",
    "IsDirProtocol",
    "LoadAndDecodeSessionFunc",
    "LoadOrchestratorProtocol",
    "LogLevel",
    "PathExistsProtocol",
    "ReadTextProtocol",
    "ResolveTreeHashProtocol",
    "RunForProjectProtocol",
    "SessionDecoderProtocol",
    "SetupRichLoggingProtocol",
    "get_script_path",
    "http_get",
    "is_dir",
    "load_and_decode_session",
    "load_orchestrator",
    "path_exists",
    "read_text",
    "resolve_tree_hash",
    "set_script_path",
    "setup_rich_logging",
]
