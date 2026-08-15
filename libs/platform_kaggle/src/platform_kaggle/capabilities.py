"""Codebase capability detection for platform_kaggle.

The individual detectors live in the private _capabilities_* modules, grouped
by the kind of capability they look for; this module composes them into a
profile and exposes the public entry points.
"""

from __future__ import annotations

from pathlib import Path

from platform_codebase import (
    CodebaseProfile,
    LibInfo,
    ServiceInfo,
    scan_libs,
    scan_services,
)

from platform_kaggle._capabilities_domain import (
    _detect_domain_capabilities,
    _detect_observability_capabilities,
    _detect_streaming_capabilities,
)
from platform_kaggle._capabilities_llm import (
    _detect_cloud_capabilities,
    _detect_llm_api_capabilities,
    _detect_technologies,
    _detect_web_frameworks,
)
from platform_kaggle._capabilities_ml import (
    _detect_cv_capabilities,
    _detect_ml_capabilities,
    _detect_transformers_capabilities,
)
from platform_kaggle._capabilities_nlp import (
    _detect_data_formats,
    _detect_ml_backends,
    _detect_nlp_capabilities,
    _detect_task_types,
)

# -----------------------------------------------------------------------------
# Capability Detection
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def build_profile(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> CodebaseProfile:
    """Build capability profile from pre-scanned libs and services.

    This function accepts already-scanned data, enabling use with data
    from GitHub API or other sources beyond local filesystem.

    Args:
        libs: Tuple of LibInfo from libs directory.
        services: Tuple of ServiceInfo from services directory.

    Returns:
        CodebaseProfile with detected capabilities.
    """
    ml_caps = _detect_ml_capabilities(libs, services)
    cv_caps = _detect_cv_capabilities(libs, services)
    transformers_caps = _detect_transformers_capabilities(libs, services)
    nlp_caps = _detect_nlp_capabilities(libs, services)
    domain_caps = _detect_domain_capabilities(libs, services)
    observability_caps = _detect_observability_capabilities(libs, services)
    streaming_caps = _detect_streaming_capabilities(libs, services)
    llm_api_caps = _detect_llm_api_capabilities(libs, services)
    cloud_caps = _detect_cloud_capabilities(libs, services)

    all_caps = (
        ml_caps
        + cv_caps
        + transformers_caps
        + nlp_caps
        + domain_caps
        + observability_caps
        + streaming_caps
        + llm_api_caps
        + cloud_caps
    )

    return CodebaseProfile(
        capabilities=all_caps,
        technologies=_detect_technologies(libs, services),
        frameworks=_detect_web_frameworks(libs, services),
        ml_backends=_detect_ml_backends(libs, services),
        data_formats=_detect_data_formats(libs, services),
        task_types=_detect_task_types(libs, services),
    )


def scan_codebase(root: Path) -> CodebaseProfile:
    """Scan codebase and return capability profile.

    Scans the libs/ and services/ directories to detect installed
    dependencies and infer ML/NLP capabilities.

    Args:
        root: Path to monorepo root directory.

    Returns:
        CodebaseProfile with detected capabilities.
    """
    libs = scan_libs(root)
    services = scan_services(root)
    return build_profile(libs, services)


__all__ = [
    "build_profile",
    "scan_codebase",
]
