"""Test hooks for dependency injection.

Every hook is bound to its real implementation here, so production code calls
it directly with no conditional. Tests rebind a hook to a fake and restore it
afterwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from platform_codebase.github_scanner import GitHubClient


class ContainerFindMonorepoRootProto(Protocol):
    """Protocol for container's find_monorepo_root hook."""

    def __call__(self) -> Path:
        """Find monorepo root."""
        ...


class GitHubClientFactoryProto(Protocol):
    """Protocol for GitHub client factory hook."""

    def __call__(self, token: str) -> GitHubClientProtocol:
        """Create a GitHub client."""
        ...


class GitHubClientProtocol(Protocol):
    """Protocol for GitHub API client (minimal for type hints)."""

    def list_directory(self, owner: str, repo: str, path: str) -> list[str]:
        """List directory contents."""
        ...

    def get_file_content(self, owner: str, repo: str, path: str) -> str | None:
        """Get file content."""
        ...

    def check_path_exists(self, owner: str, repo: str, path: str, pattern: str) -> bool:
        """Check if path matching pattern exists."""
        ...


def _real_find_monorepo_root() -> Path:
    """Find the monorepo root by looking for a libs directory.

    Returns:
        Path to the monorepo root.

    Raises:
        RuntimeError: If no ancestor directory contains 'libs'.
    """
    current = Path(__file__).resolve()
    while True:
        if (current / "libs").is_dir():
            return current
        if current.parent == current:
            raise RuntimeError("Could not find monorepo root with 'libs' directory")
        current = current.parent


# Container hooks, bound to their real implementations so callers invoke
# them directly. Tests rebind them to fakes and restore them afterwards.
container_find_monorepo_root: ContainerFindMonorepoRootProto = _real_find_monorepo_root
container_github_client_factory: GitHubClientFactoryProto = GitHubClient
