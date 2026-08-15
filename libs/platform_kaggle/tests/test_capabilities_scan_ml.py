"""Tests for capabilities: scanning for machine-learning libraries."""

from __future__ import annotations

from pathlib import Path

from platform_kaggle.capabilities import scan_codebase


class TestScanCodebaseMachineLearning:
    """Scanning for machine-learning libraries."""

    def test_scan_codebase_with_xgboost(self, tmp_path: Path) -> None:
        """Test scanning codebase with XGBoost dependency."""
        # Create libs directory structure
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "ml_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "ml-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
xgboost = "^2.0.0"
pandas = "^2.0.0"
"""
        )

        # Create empty services directory
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert "xgboost" in profile.ml_backends
        assert "csv" in profile.data_formats
        assert "excel" in profile.data_formats

        # Check for xgboost capability
        cap_names = [c.name for c in profile.capabilities]
        assert "xgboost_tabular" in cap_names

    def test_scan_codebase_with_lightgbm(self, tmp_path: Path) -> None:
        """Test scanning codebase with LightGBM dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "lgb_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "lgb-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
lightgbm = "^4.0.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert "lightgbm" in profile.ml_backends
        cap_names = [c.name for c in profile.capabilities]
        assert "lightgbm_tabular" in cap_names

    def test_scan_codebase_with_pytorch(self, tmp_path: Path) -> None:
        """Test scanning codebase with PyTorch dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "dl_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "dl-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
torch = "^2.0.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert "pytorch" in profile.ml_backends
        assert "time_series" in profile.task_types
        assert "sequence_modeling" in profile.task_types

        cap_names = [c.name for c in profile.capabilities]
        assert "pytorch_deep_learning" in cap_names

    def test_scan_codebase_with_optuna(self, tmp_path: Path) -> None:
        """Test scanning codebase with Optuna dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "opt_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "opt-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
optuna = "^3.0.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "hyperparameter_optimization" in cap_names

    def test_scan_codebase_with_sklearn(self, tmp_path: Path) -> None:
        """Test scanning codebase with scikit-learn dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "sklearn_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "sklearn-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
scikit-learn = "^1.0.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert "sklearn" in profile.ml_backends
        assert "clustering" in profile.task_types

        cap_names = [c.name for c in profile.capabilities]
        assert "sklearn_ml" in cap_names

    def test_scan_codebase_with_fasttext(self, tmp_path: Path) -> None:
        """Test scanning codebase with FastText dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "nlp_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "nlp-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
fasttext = "^0.9.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "language_identification" in cap_names

    def test_scan_codebase_with_openai(self, tmp_path: Path) -> None:
        """Test scanning codebase with OpenAI dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "stt_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "stt-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
openai = "^1.0.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "speech_to_text" in cap_names

    def test_scan_codebase_with_rules_files(self, tmp_path: Path) -> None:
        """Test scanning codebase with .rules files for transliteration."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "transliteration_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "transliteration-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
"""
        )

        # Create a .rules file (transliteration rules file)
        rules_file = service_dir / "latin_to_cyrillic.rules"
        rules_file.write_text("a -> cyrillic_a")

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "transliteration" in cap_names

    def test_scan_codebase_with_polars(self, tmp_path: Path) -> None:
        """Test scanning codebase with Polars dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "data_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "data-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
polars = "^0.20.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert "csv" in profile.data_formats
        assert "parquet" in profile.data_formats
