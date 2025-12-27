"""Public test utilities for platform_codebase consumers.

This module provides fakes, factories, and hooks for testing code that
depends on platform_codebase.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from platform_core.http_client import HttpxClient, HttpxResponse
from platform_core.json_utils import JSONValue

from platform_codebase.github_scanner import GitHubClientProtocol
from platform_codebase.types import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    LibInfo,
    ServiceInfo,
)

# -----------------------------------------------------------------------------
# Fake GitHub Client
# -----------------------------------------------------------------------------


class FakeGitHubClient:
    """Fake GitHub client for testing.

    This fake implements GitHubClientProtocol and allows tests to configure
    what directories and files exist in the fake repository.

    Attributes:
        directories: Dict mapping path -> list of directory names.
        files: Dict mapping path -> file content.
        path_patterns: Dict mapping (path, pattern) -> bool for pattern checks.
    """

    __slots__ = ("_directories", "_files", "_path_patterns")

    def __init__(
        self,
        *,
        directories: dict[str, list[str]] | None = None,
        files: dict[str, str] | None = None,
        path_patterns: dict[tuple[str, str], bool] | None = None,
    ) -> None:
        """Initialize fake client with configured responses.

        Args:
            directories: Mapping of path -> list of directory names.
            files: Mapping of path -> file content.
            path_patterns: Mapping of (path, pattern) -> exists result.
        """
        self._directories: dict[str, list[str]] = directories if directories else {}
        self._files: dict[str, str] = files if files else {}
        self._path_patterns: dict[tuple[str, str], bool] = path_patterns if path_patterns else {}

    def list_directory(self, owner: str, repo: str, path: str) -> list[str]:
        """List directory contents.

        Args:
            owner: Repository owner (ignored in fake).
            repo: Repository name (ignored in fake).
            path: Path within repository.

        Returns:
            List of directory names from configured directories.
        """
        # Ignore owner/repo in fake - just look up path
        _ = owner, repo
        return self._directories.get(path, [])

    def get_file_content(self, owner: str, repo: str, path: str) -> str | None:
        """Get file content.

        Args:
            owner: Repository owner (ignored in fake).
            repo: Repository name (ignored in fake).
            path: Path to file within repository.

        Returns:
            File content from configured files, or None if not found.
        """
        _ = owner, repo
        return self._files.get(path)

    def check_path_exists(self, owner: str, repo: str, path: str, pattern: str) -> bool:
        """Check if any file matching pattern exists under path.

        Args:
            owner: Repository owner (ignored in fake).
            repo: Repository name (ignored in fake).
            path: Path to search under.
            pattern: File extension pattern.

        Returns:
            Configured result for (path, pattern), or False if not configured.
        """
        _ = owner, repo
        return self._path_patterns.get((path, pattern), False)


# Type assertion to verify FakeGitHubClient implements protocol
_fake_client_check: GitHubClientProtocol = FakeGitHubClient()


# -----------------------------------------------------------------------------
# Fake HTTP Client for GitHubClient Testing
# -----------------------------------------------------------------------------


class FakeHttpxResponse:
    """Fake HTTP response for testing.

    Implements HttpxResponse protocol with configurable status, data, and headers.
    Uses public attributes as required by the Protocol (not properties).
    """

    __slots__ = ("_data", "content", "headers", "status_code", "text")

    def __init__(
        self,
        *,
        status_code: int = 200,
        data: JSONValue = None,
        text: str = "",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize fake response.

        Args:
            status_code: HTTP status code.
            data: JSON data to return from json().
            text: Response text.
            headers: Response headers.
        """
        self.status_code: int = status_code
        self._data: JSONValue = data
        self.text: str = text
        self.headers: Mapping[str, str] = headers if headers else {}
        self.content: bytes | bytearray = text.encode("utf-8")

    def json(self) -> JSONValue:
        """Return the JSON data."""
        return self._data

    def raise_for_status(self) -> None:
        """Raise HTTPStatusError if status >= 400.

        Raises:
            Exception: If status code indicates an error.
        """
        if self.status_code >= 400:
            msg = f"HTTP {self.status_code}"
            raise Exception(msg)


# Type assertion to verify FakeHttpxResponse implements protocol
_fake_response_check: HttpxResponse = FakeHttpxResponse()


class FakeHttpxClient:
    """Fake HTTP client for testing GitHubClient.

    Allows configuring responses for specific URL patterns.
    """

    __slots__ = ("_responses",)

    def __init__(
        self,
        *,
        responses: dict[str, FakeHttpxResponse] | None = None,
    ) -> None:
        """Initialize fake client with configured responses.

        Args:
            responses: Mapping of URL substring -> response to return.
        """
        self._responses: dict[str, FakeHttpxResponse] = responses if responses else {}

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str | int] | None = None,
    ) -> HttpxResponse:
        """Simulate GET request.

        Args:
            url: Request URL.
            headers: Request headers (ignored).
            params: Query parameters (ignored).

        Returns:
            Configured response for matching URL, or 404 response.
        """
        _ = headers, params
        for url_pattern, response in self._responses.items():
            if url_pattern in url:
                return response
        # Default to 404 if no match
        return FakeHttpxResponse(status_code=404)

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        """Simulate POST request (not used by GitHubClient).

        Args:
            url: Request URL.
            headers: Request headers.
            json: JSON body.
            files: File uploads.

        Returns:
            404 response (POST not used by GitHub scanner).
        """
        _ = url, headers, json, files
        return FakeHttpxResponse(status_code=404)

    def close(self) -> None:
        """Close the client (no-op for fake)."""


# Type assertion to verify FakeHttpxClient implements protocol
_fake_http_client_check: HttpxClient = FakeHttpxClient()


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def make_fake_capability(
    *,
    name: str = "test_capability",
    strength: CapabilityStrength = "moderate",
    tags: tuple[str, ...] = ("test",),
    description: str = "Test capability",
) -> CodebaseCapability:
    """Create a fake CodebaseCapability for testing.

    Args:
        name: Capability identifier.
        strength: Capability strength level.
        tags: Tuple of tags.
        description: Human-readable description.

    Returns:
        CodebaseCapability instance.
    """
    return CodebaseCapability(
        name=name,
        strength=strength,
        tags=tags,
        description=description,
    )


def make_fake_profile(
    *,
    capabilities: tuple[CodebaseCapability, ...] = (),
    technologies: tuple[str, ...] = (),
    frameworks: tuple[str, ...] = (),
    ml_backends: tuple[str, ...] = (),
    data_formats: tuple[str, ...] = (),
    task_types: tuple[str, ...] = (),
) -> CodebaseProfile:
    """Create a fake CodebaseProfile for testing.

    Args:
        capabilities: Tuple of detected capabilities.
        technologies: Tuple of technology names.
        frameworks: Tuple of framework names.
        ml_backends: Tuple of ML backend names.
        data_formats: Tuple of supported data formats.
        task_types: Tuple of supported task types.

    Returns:
        CodebaseProfile instance.
    """
    return CodebaseProfile(
        capabilities=capabilities,
        technologies=technologies,
        frameworks=frameworks,
        ml_backends=ml_backends,
        data_formats=data_formats,
        task_types=task_types,
    )


def make_fake_lib_info(
    *,
    name: str = "test-lib",
    path: Path | None = None,
    dependencies: tuple[str, ...] = (),
) -> LibInfo:
    """Create a fake LibInfo for testing.

    Args:
        name: Library name.
        path: Path to library directory.
        dependencies: Tuple of dependency names.

    Returns:
        LibInfo instance.
    """
    return LibInfo(
        name=name,
        path=path if path is not None else Path("libs/test-lib"),
        dependencies=dependencies,
    )


def make_fake_service_info(
    *,
    name: str = "test-service",
    path: Path | None = None,
    dependencies: tuple[str, ...] = (),
    has_rules_files: bool = False,
) -> ServiceInfo:
    """Create a fake ServiceInfo for testing.

    Args:
        name: Service name.
        path: Path to service directory.
        dependencies: Tuple of dependency names.
        has_rules_files: Whether service has .rules files.

    Returns:
        ServiceInfo instance.
    """
    return ServiceInfo(
        name=name,
        path=path if path is not None else Path("services/test-service"),
        dependencies=dependencies,
        has_rules_files=has_rules_files,
    )
