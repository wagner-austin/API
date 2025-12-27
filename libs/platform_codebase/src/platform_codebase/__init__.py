"""Codebase capability detection and profiling for monorepos.

This library provides shared types and utilities for scanning monorepo
codebases to detect capabilities, technologies, and frameworks.
"""

from __future__ import annotations

from platform_codebase.github_scanner import (
    GitHubClient,
    GitHubClientProtocol,
    parse_github_repo,
    scan_libs_from_github,
    scan_services_from_github,
)
from platform_codebase.scanner import (
    collect_all_dependencies,
    has_dependency,
    scan_libs,
    scan_services,
)
from platform_codebase.testing import (
    FakeGitHubClient,
    FakeHttpxClient,
    FakeHttpxResponse,
    make_fake_capability,
    make_fake_lib_info,
    make_fake_profile,
    make_fake_service_info,
)
from platform_codebase.toml import (
    extract_poetry_dependencies,
    extract_poetry_name,
    parse_pyproject,
)
from platform_codebase.types import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    LibInfo,
    MatchRecommendation,
    ServiceInfo,
    decode_capability,
    decode_profile,
    encode_capability,
    encode_lib_info,
    encode_profile,
    encode_service_info,
    require_recommendation,
    require_strength,
)

__all__ = [
    "CapabilityStrength",
    "CodebaseCapability",
    "CodebaseProfile",
    "FakeGitHubClient",
    "FakeHttpxClient",
    "FakeHttpxResponse",
    "GitHubClient",
    "GitHubClientProtocol",
    "LibInfo",
    "MatchRecommendation",
    "ServiceInfo",
    "collect_all_dependencies",
    "decode_capability",
    "decode_profile",
    "encode_capability",
    "encode_lib_info",
    "encode_profile",
    "encode_service_info",
    "extract_poetry_dependencies",
    "extract_poetry_name",
    "has_dependency",
    "make_fake_capability",
    "make_fake_lib_info",
    "make_fake_profile",
    "make_fake_service_info",
    "parse_github_repo",
    "parse_pyproject",
    "require_recommendation",
    "require_strength",
    "scan_libs",
    "scan_libs_from_github",
    "scan_services",
    "scan_services_from_github",
]
