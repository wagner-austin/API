"""Codebase capability detection for hackathon matching.

This module scans pyproject.toml files to detect technologies and frameworks
that can be used to match hackathons.
"""

from __future__ import annotations

from pathlib import Path

from platform_codebase import (
    CodebaseCapability,
    CodebaseProfile,
    LibInfo,
    ServiceInfo,
    collect_all_dependencies,
    has_dependency,
    scan_libs,
    scan_services,
)

from platform_devpost.types import CapabilityStrength

# -----------------------------------------------------------------------------
# Capability Detection Rules
# -----------------------------------------------------------------------------

# Technology detection: (dep, tech, cap_name, strength, tags, desc)
_TechRule = tuple[str, str, str, CapabilityStrength, tuple[str, ...], str]
TECHNOLOGY_RULES: tuple[_TechRule, ...] = (
    (
        "python",
        "python",
        "python_development",
        CapabilityStrength.STRONG,
        ("python", "backend"),
        "Python",
    ),
    (
        "flask",
        "python",
        "web_development",
        CapabilityStrength.MODERATE,
        ("web", "api", "backend"),
        "Flask",
    ),
    (
        "django",
        "python",
        "web_development",
        CapabilityStrength.STRONG,
        ("web", "api", "backend"),
        "Django",
    ),
    (
        "fastapi",
        "python",
        "web_development",
        CapabilityStrength.STRONG,
        ("web", "api", "backend"),
        "FastAPI",
    ),
    (
        "react",
        "javascript",
        "frontend_development",
        CapabilityStrength.STRONG,
        ("web", "frontend", "js"),
        "React",
    ),
    (
        "vue",
        "javascript",
        "frontend_development",
        CapabilityStrength.MODERATE,
        ("web", "frontend", "js"),
        "Vue.js",
    ),
    (
        "pytorch",
        "python",
        "machine_learning",
        CapabilityStrength.STRONG,
        ("ml", "ai", "deep-learning"),
        "PyTorch",
    ),
    (
        "tensorflow",
        "python",
        "machine_learning",
        CapabilityStrength.STRONG,
        ("ml", "ai", "deep-learning"),
        "TF",
    ),
    (
        "scikit-learn",
        "python",
        "machine_learning",
        CapabilityStrength.MODERATE,
        ("ml", "ai", "data"),
        "sklearn",
    ),
    (
        "xgboost",
        "python",
        "machine_learning",
        CapabilityStrength.STRONG,
        ("ml", "tabular", "clf"),
        "XGBoost",
    ),
    (
        "lightgbm",
        "python",
        "machine_learning",
        CapabilityStrength.STRONG,
        ("ml", "tabular", "clf"),
        "LightGBM",
    ),
    (
        "openai",
        "python",
        "ai_integration",
        CapabilityStrength.MODERATE,
        ("ai", "llm", "api"),
        "OpenAI",
    ),
    (
        "langchain",
        "python",
        "ai_integration",
        CapabilityStrength.STRONG,
        ("ai", "llm", "rag"),
        "LangChain",
    ),
    (
        "httpx",
        "python",
        "api_development",
        CapabilityStrength.MODERATE,
        ("api", "http", "client"),
        "httpx",
    ),
    (
        "requests",
        "python",
        "api_development",
        CapabilityStrength.MODERATE,
        ("api", "http", "client"),
        "requests",
    ),
    (
        "sqlalchemy",
        "python",
        "database",
        CapabilityStrength.STRONG,
        ("database", "sql", "orm"),
        "SQLAlchemy",
    ),
    (
        "pandas",
        "python",
        "data_analysis",
        CapabilityStrength.STRONG,
        ("data", "analytics", "tabular"),
        "pandas",
    ),
    (
        "polars",
        "python",
        "data_analysis",
        CapabilityStrength.STRONG,
        ("data", "analytics", "tabular"),
        "polars",
    ),
    (
        "numpy",
        "python",
        "data_analysis",
        CapabilityStrength.MODERATE,
        ("data", "numerical", "sci"),
        "NumPy",
    ),
)


def _detect_capabilities(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[CodebaseCapability, ...]:
    """Detect capabilities from scanned libs/services.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of detected capabilities.
    """
    capabilities: list[CodebaseCapability] = []
    seen_names: set[str] = set()
    all_deps = collect_all_dependencies(libs, services)

    for dep, _tech, cap_name, strength, tags, description in TECHNOLOGY_RULES:
        if has_dependency(all_deps, dep) and cap_name not in seen_names:
            capabilities.append(
                CodebaseCapability(
                    name=cap_name,
                    strength=strength,
                    tags=tags,
                    description=description,
                )
            )
            seen_names.add(cap_name)

    return tuple(capabilities)


def _detect_technologies(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[str, ...]:
    """Detect technologies from scanned libs/services.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of technology names.
    """
    technologies: set[str] = set()
    all_deps = collect_all_dependencies(libs, services)

    for dep, tech, _cap, _strength, _tags, _desc in TECHNOLOGY_RULES:
        if has_dependency(all_deps, dep):
            technologies.add(tech)

    return tuple(sorted(technologies))


def _detect_frameworks(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[str, ...]:
    """Detect frameworks from scanned libs/services.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of framework names.
    """
    framework_deps = {
        "flask",
        "django",
        "fastapi",
        "pytorch",
        "tensorflow",
        "react",
        "vue",
        "langchain",
        "sqlalchemy",
    }

    all_deps = collect_all_dependencies(libs, services)
    frameworks: list[str] = []

    for dep in framework_deps:
        if has_dependency(all_deps, dep):
            frameworks.append(dep)

    return tuple(sorted(frameworks))


def scan_codebase(root: Path) -> CodebaseProfile:
    """Scan codebase and return capability profile.

    This function scans pyproject.toml files in libs/ and services/
    directories to detect technologies and frameworks.

    Args:
        root: Path to monorepo root.

    Returns:
        CodebaseProfile with detected capabilities.
    """
    libs = scan_libs(root)
    services = scan_services(root)

    return CodebaseProfile(
        capabilities=_detect_capabilities(libs, services),
        technologies=_detect_technologies(libs, services),
        frameworks=_detect_frameworks(libs, services),
    )
