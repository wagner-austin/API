"""Codebase scanning utilities.

This module provides utilities for scanning monorepo libs/ and services/
directories to detect installed dependencies.
"""

from __future__ import annotations

from pathlib import Path

from platform_codebase.toml import parse_pyproject
from platform_codebase.types import LibInfo, ServiceInfo


def scan_libs(root: Path) -> tuple[LibInfo, ...]:
    """Scan libs/ directory for pyproject.toml files.

    Args:
        root: Path to monorepo root.

    Returns:
        Tuple of LibInfo for each library found.
    """
    libs_dir = root / "libs"
    if not libs_dir.exists():
        return ()

    result: list[LibInfo] = []
    for lib_dir in libs_dir.iterdir():
        if not lib_dir.is_dir():
            continue
        pyproject = lib_dir / "pyproject.toml"
        if not pyproject.exists():
            continue

        name, deps = parse_pyproject(pyproject)
        result.append(
            LibInfo(
                name=name,
                path=lib_dir,
                dependencies=deps,
            )
        )

    return tuple(result)


def scan_services(root: Path) -> tuple[ServiceInfo, ...]:
    """Scan services/ directory for pyproject.toml files.

    Args:
        root: Path to monorepo root.

    Returns:
        Tuple of ServiceInfo for each service found.
    """
    services_dir = root / "services"
    if not services_dir.exists():
        return ()

    result: list[ServiceInfo] = []
    for service_dir in services_dir.iterdir():
        if not service_dir.is_dir():
            continue
        pyproject = service_dir / "pyproject.toml"
        if not pyproject.exists():
            continue

        name, deps = parse_pyproject(pyproject)

        # Check for .rules files (transliteration rules)
        has_rules = any(service_dir.rglob("*.rules"))

        result.append(
            ServiceInfo(
                name=name,
                path=service_dir,
                dependencies=deps,
                has_rules_files=has_rules,
            )
        )

    return tuple(result)


def has_dependency(deps: tuple[str, ...], name: str) -> bool:
    """Check if dependency is in list.

    Args:
        deps: Tuple of dependency names.
        name: Dependency to check for.

    Returns:
        True if dependency is present.
    """
    return any(dep == name or dep.startswith(name + "[") for dep in deps)


def collect_all_dependencies(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[str, ...]:
    """Collect all unique dependencies from libs and services.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of unique dependency names.
    """
    all_deps: set[str] = set()

    for lib in libs:
        for dep in lib.dependencies:
            all_deps.add(dep)
    for service in services:
        for dep in service.dependencies:
            all_deps.add(dep)

    return tuple(sorted(all_deps))
