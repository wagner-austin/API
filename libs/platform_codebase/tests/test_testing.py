"""Tests for platform_codebase.testing module."""

from __future__ import annotations

from pathlib import Path

from platform_codebase.testing import (
    make_fake_capability,
    make_fake_lib_info,
    make_fake_profile,
    make_fake_service_info,
)


class TestMakeFakeCapability:
    """Tests for make_fake_capability function."""

    def test_default_values(self) -> None:
        """Test with default values."""
        cap = make_fake_capability()

        assert cap.name == "test_capability"
        assert cap.strength == "moderate"
        assert cap.tags == ("test",)
        assert cap.description == "Test capability"

    def test_custom_values(self) -> None:
        """Test with custom values."""
        cap = make_fake_capability(
            name="custom_cap",
            strength="strong",
            tags=("ml", "tabular"),
            description="Custom description",
        )

        assert cap.name == "custom_cap"
        assert cap.strength == "strong"
        assert cap.tags == ("ml", "tabular")
        assert cap.description == "Custom description"


class TestMakeFakeProfile:
    """Tests for make_fake_profile function."""

    def test_default_values(self) -> None:
        """Test with default values."""
        profile = make_fake_profile()

        assert profile.capabilities == ()
        assert profile.technologies == ()
        assert profile.frameworks == ()
        assert profile.ml_backends == ()
        assert profile.data_formats == ()
        assert profile.task_types == ()

    def test_custom_values(self) -> None:
        """Test with custom values."""
        cap = make_fake_capability()
        profile = make_fake_profile(
            capabilities=(cap,),
            technologies=("python",),
            frameworks=("fastapi",),
            ml_backends=("xgboost",),
            data_formats=("csv",),
            task_types=("classification",),
        )

        assert len(profile.capabilities) == 1
        assert profile.technologies == ("python",)
        assert profile.frameworks == ("fastapi",)
        assert profile.ml_backends == ("xgboost",)
        assert profile.data_formats == ("csv",)
        assert profile.task_types == ("classification",)


class TestMakeFakeLibInfo:
    """Tests for make_fake_lib_info function."""

    def test_default_values(self) -> None:
        """Test with default values."""
        lib = make_fake_lib_info()

        assert lib.name == "test-lib"
        assert lib.path == Path("libs/test-lib")
        assert lib.dependencies == ()

    def test_custom_values(self) -> None:
        """Test with custom values."""
        lib = make_fake_lib_info(
            name="custom-lib",
            path=Path("libs/custom"),
            dependencies=("dep1", "dep2"),
        )

        assert lib.name == "custom-lib"
        assert lib.path == Path("libs/custom")
        assert lib.dependencies == ("dep1", "dep2")


class TestMakeFakeServiceInfo:
    """Tests for make_fake_service_info function."""

    def test_default_values(self) -> None:
        """Test with default values."""
        service = make_fake_service_info()

        assert service.name == "test-service"
        assert service.path == Path("services/test-service")
        assert service.dependencies == ()
        assert service.has_rules_files is False

    def test_custom_values(self) -> None:
        """Test with custom values."""
        service = make_fake_service_info(
            name="custom-service",
            path=Path("services/custom"),
            dependencies=("fastapi",),
            has_rules_files=True,
        )

        assert service.name == "custom-service"
        assert service.path == Path("services/custom")
        assert service.dependencies == ("fastapi",)
        assert service.has_rules_files is True
