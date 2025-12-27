"""Service container for dependency injection."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from platform_codebase import (
    CodebaseProfile,
    GitHubClient,
    GitHubClientProtocol,
    LibInfo,
    ServiceInfo,
    parse_github_repo,
    scan_libs_from_github,
    scan_services_from_github,
)
from platform_codebase import (
    scan_libs as codebase_scan_libs,
)
from platform_codebase import (
    scan_services as codebase_scan_services,
)
from platform_devpost import DevpostClient, DevpostClientProtocol
from platform_kaggle import KaggleClient, KaggleClientProtocol, build_profile

from opportunity_radar_api.config import OpportunityRadarSettings


class ServiceContainer:
    """Container for service dependencies.

    Provides dependency injection for API routes. Production implementations
    are wired by default, but can be replaced for testing.

    Attributes:
        monorepo_root: Path to the monorepo root directory.
    """

    __slots__ = (
        "_codebase_profile_factory",
        "_devpost_client_factory",
        "_kaggle_client_factory",
        "_libs_scanner",
        "_services_scanner",
        "monorepo_root",
    )

    def __init__(
        self,
        *,
        monorepo_root: Path,
        kaggle_client_factory: Callable[[], KaggleClientProtocol],
        devpost_client_factory: Callable[[], DevpostClientProtocol],
        codebase_profile_factory: Callable[[Path], CodebaseProfile],
        libs_scanner: Callable[[Path], tuple[LibInfo, ...]],
        services_scanner: Callable[[Path], tuple[ServiceInfo, ...]],
    ) -> None:
        """Initialize container.

        Args:
            monorepo_root: Path to monorepo root.
            kaggle_client_factory: Factory for Kaggle client.
            devpost_client_factory: Factory for Devpost client.
            codebase_profile_factory: Factory for codebase profile.
            libs_scanner: Function to scan libs directory.
            services_scanner: Function to scan services directory.
        """
        self.monorepo_root = monorepo_root
        self._kaggle_client_factory = kaggle_client_factory
        self._devpost_client_factory = devpost_client_factory
        self._codebase_profile_factory = codebase_profile_factory
        self._libs_scanner = libs_scanner
        self._services_scanner = services_scanner

    def get_kaggle_client(self) -> KaggleClientProtocol:
        """Get Kaggle API client.

        Returns:
            Configured Kaggle client.
        """
        return self._kaggle_client_factory()

    def get_devpost_client(self) -> DevpostClientProtocol:
        """Get Devpost API client.

        Returns:
            Configured Devpost client.
        """
        return self._devpost_client_factory()

    def get_codebase_profile(self) -> CodebaseProfile:
        """Get codebase capability profile.

        Returns:
            CodebaseProfile for the monorepo.
        """
        return self._codebase_profile_factory(self.monorepo_root)

    def scan_libs(self) -> tuple[LibInfo, ...]:
        """Scan libs directory.

        Returns:
            Tuple of LibInfo for each library.
        """
        return self._libs_scanner(self.monorepo_root)

    def scan_services(self) -> tuple[ServiceInfo, ...]:
        """Scan services directory.

        Returns:
            Tuple of ServiceInfo for each service.
        """
        return self._services_scanner(self.monorepo_root)


def _default_codebase_profile_factory(root: Path) -> CodebaseProfile:
    """Build codebase profile from monorepo root.

    Args:
        root: Path to monorepo root.

    Returns:
        CodebaseProfile with detected capabilities.
    """
    from platform_kaggle import get_codebase_profile

    return get_codebase_profile(root)


def create_production_container(
    settings: OpportunityRadarSettings,
    monorepo_root: Path | None = None,
) -> ServiceContainer:
    """Create container with production dependencies.

    When GITHUB_TOKEN and GITHUB_REPO are set in settings, uses GitHub API
    to scan the repository. Otherwise, scans the local filesystem.

    Args:
        settings: Service settings including optional GitHub config.
        monorepo_root: Path to monorepo root. If None, auto-detects.

    Returns:
        Configured ServiceContainer.
    """
    github_token = settings["github_token"]
    github_repo = settings["github_repo"]

    # Check if GitHub scanning is configured
    if github_token is not None and github_repo is not None:
        # Use GitHub-based scanning
        from opportunity_radar_api import _test_hooks

        owner, repo = parse_github_repo(github_repo)
        github_client: GitHubClientProtocol
        if _test_hooks.container_github_client_factory is not None:
            github_client = _test_hooks.container_github_client_factory(github_token)
        else:
            github_client = GitHubClient(github_token)

        def github_libs_scanner(root: Path) -> tuple[LibInfo, ...]:
            _ = root  # Unused when scanning via GitHub
            return scan_libs_from_github(github_client, owner, repo)

        def github_services_scanner(root: Path) -> tuple[ServiceInfo, ...]:
            _ = root  # Unused when scanning via GitHub
            return scan_services_from_github(github_client, owner, repo)

        def github_profile_factory(root: Path) -> CodebaseProfile:
            _ = root  # Unused when scanning via GitHub
            libs = scan_libs_from_github(github_client, owner, repo)
            services = scan_services_from_github(github_client, owner, repo)
            return build_profile(libs, services)

        # Use a dummy path since we're not using filesystem
        effective_root = Path("/github") / github_repo

        return ServiceContainer(
            monorepo_root=effective_root,
            kaggle_client_factory=KaggleClient,
            devpost_client_factory=DevpostClient,
            codebase_profile_factory=github_profile_factory,
            libs_scanner=github_libs_scanner,
            services_scanner=github_services_scanner,
        )

    # Fall back to local filesystem scanning
    if monorepo_root is None:
        monorepo_root = _find_monorepo_root()

    return ServiceContainer(
        monorepo_root=monorepo_root,
        kaggle_client_factory=KaggleClient,
        devpost_client_factory=DevpostClient,
        codebase_profile_factory=_default_codebase_profile_factory,
        libs_scanner=codebase_scan_libs,
        services_scanner=codebase_scan_services,
    )


def _find_monorepo_root_impl() -> Path:
    """Find monorepo root by looking for libs directory.

    Returns:
        Path to monorepo root.

    Raises:
        RuntimeError: If monorepo root not found.
    """
    current = Path(__file__).resolve()
    while True:
        if (current / "libs").is_dir():
            return current
        if current.parent == current:
            raise RuntimeError("Could not find monorepo root with 'libs' directory")
        current = current.parent


def _find_monorepo_root() -> Path:
    """Find monorepo root, using hook if set.

    Returns:
        Path to monorepo root.
    """
    from opportunity_radar_api import _test_hooks

    if _test_hooks.container_find_monorepo_root is not None:
        return _test_hooks.container_find_monorepo_root()
    return _find_monorepo_root_impl()
