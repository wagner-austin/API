"""Core types for codebase capability detection.

This module defines the shared types used for scanning monorepo codebases
to detect capabilities, technologies, and frameworks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_list,
    require_str,
)

# -----------------------------------------------------------------------------
# Literal Types
# -----------------------------------------------------------------------------

CapabilityStrength = Literal["strong", "moderate", "basic"]
MatchRecommendation = Literal["strong_fit", "good_fit", "stretch", "new_territory"]


# -----------------------------------------------------------------------------
# Internal Validation Helpers
# -----------------------------------------------------------------------------


def _require_list_str(obj: JSONObject, key: str) -> list[str]:
    """Extract required list of strings from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        List of strings.

    Raises:
        JSONTypeError: If field is missing or contains non-strings.
    """
    items = require_list(obj, key)
    result: list[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise JSONTypeError(f"Field '{key}[{i}]' must be a string, got {type(item).__name__}")
        result.append(item)
    return result


def _require_dict_value(value: JSONValue, context: str) -> JSONObject:
    """Require value to be a dict.

    Args:
        value: JSON value to check.
        context: Context for error message.

    Returns:
        The value as JSONObject.

    Raises:
        JSONTypeError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{context} must be an object, got {type(value).__name__}")
    return value


def require_strength(obj: JSONObject, key: str) -> CapabilityStrength:
    """Extract and validate CapabilityStrength from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        Validated CapabilityStrength.

    Raises:
        JSONTypeError: If field is missing or not a valid strength.
    """
    value = require_str(obj, key)
    if value == "strong":
        return "strong"
    if value == "moderate":
        return "moderate"
    if value == "basic":
        return "basic"
    raise JSONTypeError(f"Field '{key}' must be strong/moderate/basic, got '{value}'")


def require_recommendation(obj: JSONObject, key: str) -> MatchRecommendation:
    """Extract and validate MatchRecommendation from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        Validated MatchRecommendation.

    Raises:
        JSONTypeError: If field is missing or not a valid recommendation.
    """
    value = require_str(obj, key)
    if value == "strong_fit":
        return "strong_fit"
    if value == "good_fit":
        return "good_fit"
    if value == "stretch":
        return "stretch"
    if value == "new_territory":
        return "new_territory"
    raise JSONTypeError(f"Field '{key}' must be a valid recommendation, got '{value}'")


# -----------------------------------------------------------------------------
# CodebaseCapability
# -----------------------------------------------------------------------------


class CodebaseCapability:
    """A capability the codebase has.

    Attributes:
        name: Capability identifier (e.g., "tabular_classification").
        strength: Capability strength level.
        tags: Tuple of tags this capability matches.
        description: Human-readable description.
    """

    __slots__ = ("description", "name", "strength", "tags")

    def __init__(
        self,
        *,
        name: str,
        strength: CapabilityStrength,
        tags: tuple[str, ...],
        description: str,
    ) -> None:
        """Initialize capability.

        Args:
            name: Capability identifier.
            strength: Capability strength level.
            tags: Tuple of tags this capability matches.
            description: Human-readable description.
        """
        self.name = name
        self.strength = strength
        self.tags = tags
        self.description = description


def encode_capability(cap: CodebaseCapability) -> JSONObject:
    """Encode CodebaseCapability to JSON-serializable dict.

    Args:
        cap: CodebaseCapability to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "name": cap.name,
        "strength": cap.strength,
        "tags": list(cap.tags),
        "description": cap.description,
    }
    return result


def decode_capability(data: JSONObject) -> CodebaseCapability:
    """Decode CodebaseCapability from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CodebaseCapability.

    Raises:
        JSONTypeError: If validation fails.
    """
    return CodebaseCapability(
        name=require_str(data, "name"),
        strength=require_strength(data, "strength"),
        tags=tuple(_require_list_str(data, "tags")),
        description=require_str(data, "description"),
    )


# -----------------------------------------------------------------------------
# CodebaseProfile
# -----------------------------------------------------------------------------


class CodebaseProfile:
    """Full profile of codebase capabilities.

    This is a unified profile structure supporting both platform_kaggle and
    platform_devpost. Different platforms may populate different fields.

    Attributes:
        capabilities: Tuple of detected capabilities.
        technologies: Tuple of technology names (e.g., "python", "javascript").
        frameworks: Tuple of framework names (e.g., "flask", "django").
        ml_backends: Tuple of ML backend names (e.g., "xgboost", "lightgbm").
        data_formats: Tuple of supported data formats (e.g., "csv", "parquet").
        task_types: Tuple of supported task types (e.g., "binary_classification").
    """

    __slots__ = (
        "capabilities",
        "data_formats",
        "frameworks",
        "ml_backends",
        "task_types",
        "technologies",
    )

    def __init__(
        self,
        *,
        capabilities: tuple[CodebaseCapability, ...],
        technologies: tuple[str, ...] = (),
        frameworks: tuple[str, ...] = (),
        ml_backends: tuple[str, ...] = (),
        data_formats: tuple[str, ...] = (),
        task_types: tuple[str, ...] = (),
    ) -> None:
        """Initialize profile.

        Args:
            capabilities: Tuple of detected capabilities.
            technologies: Tuple of technology names.
            frameworks: Tuple of framework names.
            ml_backends: Tuple of ML backend names.
            data_formats: Tuple of supported data formats.
            task_types: Tuple of supported task types.
        """
        self.capabilities = capabilities
        self.technologies = technologies
        self.frameworks = frameworks
        self.ml_backends = ml_backends
        self.data_formats = data_formats
        self.task_types = task_types


def encode_profile(profile: CodebaseProfile) -> JSONObject:
    """Encode CodebaseProfile to JSON-serializable dict.

    Args:
        profile: CodebaseProfile to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "capabilities": [encode_capability(c) for c in profile.capabilities],
        "technologies": list(profile.technologies),
        "frameworks": list(profile.frameworks),
        "ml_backends": list(profile.ml_backends),
        "data_formats": list(profile.data_formats),
        "task_types": list(profile.task_types),
    }
    return result


def decode_profile(data: JSONObject) -> CodebaseProfile:
    """Decode CodebaseProfile from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CodebaseProfile.

    Raises:
        JSONTypeError: If validation fails.
    """
    caps_raw = require_list(data, "capabilities")
    return CodebaseProfile(
        capabilities=tuple(
            decode_capability(_require_dict_value(c, f"capabilities[{i}]"))
            for i, c in enumerate(caps_raw)
        ),
        technologies=tuple(_require_list_str(data, "technologies")),
        frameworks=tuple(_require_list_str(data, "frameworks")),
        ml_backends=tuple(_require_list_str(data, "ml_backends")),
        data_formats=tuple(_require_list_str(data, "data_formats")),
        task_types=tuple(_require_list_str(data, "task_types")),
    )


# -----------------------------------------------------------------------------
# LibInfo and ServiceInfo
# -----------------------------------------------------------------------------


class LibInfo:
    """Information about a scanned library.

    Attributes:
        name: Library name.
        path: Path to library directory.
        dependencies: Tuple of dependency names.
    """

    __slots__ = ("dependencies", "name", "path")

    def __init__(
        self,
        *,
        name: str,
        path: Path,
        dependencies: tuple[str, ...],
    ) -> None:
        """Initialize library info.

        Args:
            name: Library name.
            path: Path to library directory.
            dependencies: Tuple of dependency names.
        """
        self.name = name
        self.path = path
        self.dependencies = dependencies


class ServiceInfo:
    """Information about a scanned service.

    Attributes:
        name: Service name.
        path: Path to service directory.
        dependencies: Tuple of dependency names.
        has_rules_files: Whether service has .rules files (for transliteration).
    """

    __slots__ = ("dependencies", "has_rules_files", "name", "path")

    def __init__(
        self,
        *,
        name: str,
        path: Path,
        dependencies: tuple[str, ...],
        has_rules_files: bool,
    ) -> None:
        """Initialize service info.

        Args:
            name: Service name.
            path: Path to service directory.
            dependencies: Tuple of dependency names.
            has_rules_files: Whether service has .rules files.
        """
        self.name = name
        self.path = path
        self.dependencies = dependencies
        self.has_rules_files = has_rules_files


def encode_lib_info(lib: LibInfo) -> JSONObject:
    """Encode LibInfo to JSON-serializable dict.

    Args:
        lib: LibInfo to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "name": lib.name,
        "path": str(lib.path),
        "dependencies": list(lib.dependencies),
    }
    return result


def encode_service_info(svc: ServiceInfo) -> JSONObject:
    """Encode ServiceInfo to JSON-serializable dict.

    Args:
        svc: ServiceInfo to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "name": svc.name,
        "path": str(svc.path),
        "dependencies": list(svc.dependencies),
        "has_rules_files": svc.has_rules_files,
    }
    return result
