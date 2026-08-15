"""Tests for types: LibInfo."""

from __future__ import annotations

from pathlib import Path

from platform_kaggle.types import (
    LibInfo,
    ServiceInfo,
)
from tests._types_equality import _libinfos_equal, _serviceinfos_equal


class TestLibInfo:
    """Tests for LibInfo type."""

    def test_libinfo_creation(self) -> None:
        """Test creating a LibInfo instance."""
        info = LibInfo(
            name="sample-lib",
            path=Path("/libs/sample_lib"),
            dependencies=("xgboost", "pandas"),
        )
        assert info.name == "sample-lib"
        assert info.path == Path("/libs/sample_lib")
        assert info.dependencies == ("xgboost", "pandas")

    def test_libinfo_equality(self) -> None:
        """Test LibInfo equality comparison."""
        info1 = LibInfo(
            name="test",
            path=Path("/test"),
            dependencies=("dep1",),
        )
        info2 = LibInfo(
            name="test",
            path=Path("/test"),
            dependencies=("dep1",),
        )
        info3 = LibInfo(
            name="other",
            path=Path("/test"),
            dependencies=("dep1",),
        )
        assert _libinfos_equal(info1, info2)
        assert not _libinfos_equal(info1, info3)


class TestServiceInfo:
    """Tests for ServiceInfo type."""

    def test_serviceinfo_creation(self) -> None:
        """Test creating a ServiceInfo instance."""
        info = ServiceInfo(
            name="sample-service",
            path=Path("/services/sample_service"),
            dependencies=("openai", "fastapi"),
            has_rules_files=True,
        )
        assert info.name == "sample-service"
        assert info.path == Path("/services/sample_service")
        assert info.dependencies == ("openai", "fastapi")
        assert info.has_rules_files is True

    def test_serviceinfo_equality(self) -> None:
        """Test ServiceInfo equality comparison."""
        info1 = ServiceInfo(
            name="test",
            path=Path("/test"),
            dependencies=("dep1",),
            has_rules_files=False,
        )
        info2 = ServiceInfo(
            name="test",
            path=Path("/test"),
            dependencies=("dep1",),
            has_rules_files=False,
        )
        info3 = ServiceInfo(
            name="other",
            path=Path("/test"),
            dependencies=("dep1",),
            has_rules_files=False,
        )
        assert _serviceinfos_equal(info1, info2)
        assert not _serviceinfos_equal(info1, info3)
