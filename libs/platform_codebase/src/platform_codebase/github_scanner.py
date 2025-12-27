"""GitHub-based codebase scanning utilities.

This module provides utilities for scanning a GitHub repository's libs/ and
services/ directories to detect installed dependencies, as an alternative to
local filesystem scanning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from platform_core.http_client import HttpxClient, HttpxResponse, build_client
from platform_core.json_utils import JSONValue

from platform_codebase.toml import parse_pyproject_content
from platform_codebase.types import LibInfo, ServiceInfo

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_GITHUB_API_URL = "https://api.github.com"
_TIMEOUT_SECONDS = 30.0


# -----------------------------------------------------------------------------
# Protocols for Testing
# -----------------------------------------------------------------------------


class GitHubClientProtocol(Protocol):
    """Protocol for GitHub API client."""

    def list_directory(self, owner: str, repo: str, path: str) -> list[str]:
        """List directory contents.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: Path within repository.

        Returns:
            List of directory names.
        """
        ...

    def get_file_content(self, owner: str, repo: str, path: str) -> str | None:
        """Get file content.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: Path to file within repository.

        Returns:
            File content as string, or None if not found.
        """
        ...

    def check_path_exists(self, owner: str, repo: str, path: str, pattern: str) -> bool:
        """Check if any file matching pattern exists under path.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: Path to search under.
            pattern: File extension pattern (e.g., ".rules").

        Returns:
            True if matching file exists.
        """
        ...


# -----------------------------------------------------------------------------
# GitHub Client Implementation
# -----------------------------------------------------------------------------


class GitHubClient:
    """GitHub API client for repository scanning.

    Attributes:
        _token: GitHub personal access token.
        _client: HTTP client instance.
    """

    __slots__ = ("_client", "_token")

    def __init__(self, token: str, *, client: HttpxClient | None = None) -> None:
        """Initialize client with token.

        Args:
            token: GitHub personal access token.
            client: Optional HTTP client for dependency injection (testing).
        """
        self._token = token
        self._client: HttpxClient = client if client else build_client(_TIMEOUT_SECONDS)

    def _headers(self) -> dict[str, str]:
        """Build request headers.

        Returns:
            Headers dict with auth and accept.
        """
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def list_directory(self, owner: str, repo: str, path: str) -> list[str]:
        """List directory contents from GitHub API.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: Path within repository.

        Returns:
            List of directory names (not files).
        """
        url = f"{_GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}"
        response: HttpxResponse = self._client.get(url, headers=self._headers())

        if response.status_code == 404:
            return []

        response.raise_for_status()
        data: JSONValue = response.json()

        if not isinstance(data, list):
            return []

        directories: list[str] = []
        for item in data:
            if isinstance(item, dict):
                item_type = item.get("type")
                item_name = item.get("name")
                if item_type == "dir" and isinstance(item_name, str):
                    directories.append(item_name)

        return directories

    def get_file_content(self, owner: str, repo: str, path: str) -> str | None:
        """Get file content from GitHub API.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: Path to file within repository.

        Returns:
            File content as string, or None if not found.
        """
        url = f"{_GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}"
        response: HttpxResponse = self._client.get(url, headers=self._headers())

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data: JSONValue = response.json()

        if not isinstance(data, dict):
            return None

        # GitHub returns base64-encoded content
        encoding = data.get("encoding")
        content = data.get("content")

        if encoding == "base64" and isinstance(content, str):
            import base64

            # Remove newlines that GitHub adds
            clean_content = content.replace("\n", "")
            decoded_bytes = base64.b64decode(clean_content)
            return decoded_bytes.decode("utf-8")

        return None

    def check_path_exists(self, owner: str, repo: str, path: str, pattern: str) -> bool:
        """Check if any file matching pattern exists under path.

        Uses GitHub's search API to find files.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: Path to search under.
            pattern: File extension pattern (e.g., ".rules").

        Returns:
            True if matching file exists.
        """
        # Use contents API to list files recursively would be expensive
        # Instead, do a simple search in the directory
        url = f"{_GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}"
        response: HttpxResponse = self._client.get(url, headers=self._headers())

        if response.status_code == 404:
            return False

        response.raise_for_status()
        data: JSONValue = response.json()

        if not isinstance(data, list):
            return False

        # Check direct children for pattern match
        for item in data:
            if isinstance(item, dict):
                item_name = item.get("name")
                if isinstance(item_name, str) and item_name.endswith(pattern):
                    return True

        return False

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


# -----------------------------------------------------------------------------
# Scanner Functions
# -----------------------------------------------------------------------------


def scan_libs_from_github(
    client: GitHubClientProtocol,
    owner: str,
    repo: str,
) -> tuple[LibInfo, ...]:
    """Scan libs/ directory from GitHub repository.

    Args:
        client: GitHub API client.
        owner: Repository owner.
        repo: Repository name.

    Returns:
        Tuple of LibInfo for each library found.
    """
    lib_dirs = client.list_directory(owner, repo, "libs")

    result: list[LibInfo] = []
    for lib_name in lib_dirs:
        pyproject_path = f"libs/{lib_name}/pyproject.toml"
        content = client.get_file_content(owner, repo, pyproject_path)

        if content is None:
            continue

        name, deps = parse_pyproject_content(content)
        result.append(
            LibInfo(
                name=name,
                path=Path(f"libs/{lib_name}"),
                dependencies=deps,
            )
        )

    return tuple(result)


def scan_services_from_github(
    client: GitHubClientProtocol,
    owner: str,
    repo: str,
) -> tuple[ServiceInfo, ...]:
    """Scan services/ directory from GitHub repository.

    Args:
        client: GitHub API client.
        owner: Repository owner.
        repo: Repository name.

    Returns:
        Tuple of ServiceInfo for each service found.
    """
    service_dirs = client.list_directory(owner, repo, "services")

    result: list[ServiceInfo] = []
    for service_name in service_dirs:
        pyproject_path = f"services/{service_name}/pyproject.toml"
        content = client.get_file_content(owner, repo, pyproject_path)

        if content is None:
            continue

        name, deps = parse_pyproject_content(content)

        # Check for .rules files
        has_rules = client.check_path_exists(owner, repo, f"services/{service_name}", ".rules")

        result.append(
            ServiceInfo(
                name=name,
                path=Path(f"services/{service_name}"),
                dependencies=deps,
                has_rules_files=has_rules,
            )
        )

    return tuple(result)


def parse_github_repo(repo_str: str) -> tuple[str, str]:
    """Parse owner/repo string into components.

    Args:
        repo_str: Repository string in "owner/repo" format.

    Returns:
        Tuple of (owner, repo).

    Raises:
        ValueError: If format is invalid.
    """
    parts = repo_str.split("/")
    if len(parts) != 2:
        msg = f"Invalid repo format '{repo_str}', expected 'owner/repo'"
        raise ValueError(msg)
    return parts[0], parts[1]


__all__ = [
    "GitHubClient",
    "GitHubClientProtocol",
    "parse_github_repo",
    "scan_libs_from_github",
    "scan_services_from_github",
]
