"""Tests for capabilities: BuildProfile."""

from __future__ import annotations

from pathlib import Path

from platform_codebase import LibInfo, ServiceInfo

from platform_kaggle.capabilities import build_profile


class TestBuildProfile:
    """Tests for build_profile function."""

    def test_build_profile_with_xgboost(self) -> None:
        """Test building profile from libs with XGBoost dependency."""
        libs: tuple[LibInfo, ...] = (
            LibInfo(
                name="ml-lib",
                path=Path("libs/ml_lib"),
                dependencies=("xgboost", "pandas"),
            ),
        )
        services: tuple[ServiceInfo, ...] = ()

        profile = build_profile(libs, services)

        assert "xgboost" in profile.ml_backends
        assert "csv" in profile.data_formats
        cap_names = [c.name for c in profile.capabilities]
        assert "xgboost_tabular" in cap_names

    def test_build_profile_with_multiple_deps(self) -> None:
        """Test building profile with multiple ML dependencies."""
        libs: tuple[LibInfo, ...] = (
            LibInfo(
                name="ml-lib",
                path=Path("libs/ml_lib"),
                dependencies=("xgboost", "lightgbm", "torch", "optuna", "polars"),
            ),
        )
        services: tuple[ServiceInfo, ...] = ()

        profile = build_profile(libs, services)

        assert "xgboost" in profile.ml_backends
        assert "lightgbm" in profile.ml_backends
        assert "pytorch" in profile.ml_backends
        assert "parquet" in profile.data_formats
        cap_names = [c.name for c in profile.capabilities]
        assert "xgboost_tabular" in cap_names
        assert "lightgbm_tabular" in cap_names
        assert "pytorch_deep_learning" in cap_names
        assert "hyperparameter_optimization" in cap_names

    def test_build_profile_with_nlp_deps(self) -> None:
        """Test building profile with NLP dependencies."""
        libs: tuple[LibInfo, ...] = (
            LibInfo(
                name="nlp-lib",
                path=Path("libs/nlp_lib"),
                dependencies=("fasttext", "openai"),
            ),
        )
        services: tuple[ServiceInfo, ...] = ()

        profile = build_profile(libs, services)

        cap_names = [c.name for c in profile.capabilities]
        assert "language_identification" in cap_names
        assert "speech_to_text" in cap_names

    def test_build_profile_with_rules_files(self) -> None:
        """Test building profile from service with rules files."""
        libs: tuple[LibInfo, ...] = ()
        services: tuple[ServiceInfo, ...] = (
            ServiceInfo(
                name="turkic-api",
                path=Path("services/turkic-api"),
                dependencies=("fasttext",),
                has_rules_files=True,
            ),
        )

        profile = build_profile(libs, services)

        cap_names = [c.name for c in profile.capabilities]
        assert "transliteration" in cap_names

    def test_build_profile_empty(self) -> None:
        """Test building profile with no libs or services."""
        profile = build_profile((), ())

        assert profile.capabilities == ()
        assert profile.ml_backends == ()
        assert profile.data_formats == ()
        assert profile.task_types == ()
