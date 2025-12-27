"""Test hooks for dependency injection in tests.

This module provides hook points that tests can override to inject fake
implementations. Production code checks if hooks are None and uses
production implementations if so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class FindMonorepoRootProto(Protocol):
    """Protocol for find_monorepo_root hook."""

    def __call__(self, start: Path) -> Path:
        """Find monorepo root from start path."""
        ...


class RunForProjectProto(Protocol):
    """Protocol for run_for_project function from orchestrator."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guard checks for a project."""
        ...


class LoadOrchestratorProto(Protocol):
    """Protocol for load_orchestrator hook."""

    def __call__(self, monorepo_root: Path) -> RunForProjectProto:
        """Load the orchestrator module."""
        ...


class IsDirProto(Protocol):
    """Protocol for is_dir check hook."""

    def __call__(self, path: Path) -> bool:
        """Check if path is a directory."""
        ...


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


# Guard hooks - None means use default behavior (production implementation)
guard_find_monorepo_root: FindMonorepoRootProto | None = None
guard_load_orchestrator: LoadOrchestratorProto | None = None
guard_is_dir: IsDirProto | None = None

# Container hooks - None means use default behavior (production implementation)
container_find_monorepo_root: ContainerFindMonorepoRootProto | None = None
container_github_client_factory: GitHubClientFactoryProto | None = None
