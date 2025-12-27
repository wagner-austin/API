"""Tests for platform_kaggle.capabilities module."""

from __future__ import annotations

from pathlib import Path

from platform_codebase import LibInfo, ServiceInfo

from platform_kaggle.capabilities import build_profile, scan_codebase


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


class TestScanCodebase:
    """Tests for scan_codebase function."""

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

    def test_scan_codebase_empty(self, tmp_path: Path) -> None:
        """Test scanning empty codebase."""
        # Create empty libs and services directories
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()
        assert profile.ml_backends == ()
        assert profile.data_formats == ()
        assert profile.task_types == ()

    def test_scan_codebase_no_libs_dir(self, tmp_path: Path) -> None:
        """Test scanning codebase without libs directory."""
        # Only create services directory
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()

    def test_scan_codebase_no_services_dir(self, tmp_path: Path) -> None:
        """Test scanning codebase without services directory."""
        # Only create libs directory
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()

    def test_scan_codebase_skips_non_dirs(self, tmp_path: Path) -> None:
        """Test scanning codebase skips files in libs/services directories."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        # Create a file instead of a directory
        not_a_dir = libs_dir / "not_a_lib.txt"
        not_a_dir.write_text("This is not a library")

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()

    def test_scan_codebase_skips_missing_pyproject(self, tmp_path: Path) -> None:
        """Test scanning codebase skips directories without pyproject.toml."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        # Create a directory without pyproject.toml
        lib_dir = libs_dir / "incomplete_lib"
        lib_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert profile.capabilities == ()

    def test_scan_codebase_with_tensorflow(self, tmp_path: Path) -> None:
        """Test scanning codebase with TensorFlow dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "tf_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "tf-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
tensorflow = "^2.0.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert "tensorflow" in profile.ml_backends

    def test_scan_codebase_pyproject_no_name(self, tmp_path: Path) -> None:
        """Test scanning pyproject.toml without name field."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "noname_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        # Missing [tool.poetry] name field
        pyproject.write_text(
            """
[tool.poetry]
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
pandas = "^2.0.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        # Should still work, just have empty name
        profile = scan_codebase(tmp_path)
        # pandas was detected
        assert "csv" in profile.data_formats

    def test_scan_codebase_pyproject_no_dependencies(self, tmp_path: Path) -> None:
        """Test scanning pyproject.toml without dependencies section."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "nodeps_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        # No [tool.poetry.dependencies] section
        pyproject.write_text(
            """
[tool.poetry]
name = "nodeps-lib"
version = "0.1.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        # Should still work, just have no dependencies
        profile = scan_codebase(tmp_path)
        assert profile.capabilities == ()

    def test_scan_codebase_service_skips_non_dirs(self, tmp_path: Path) -> None:
        """Test scanning skips files in services directory."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        # Create a file instead of a directory in services
        not_a_dir = services_dir / "not_a_service.txt"
        not_a_dir.write_text("This is not a service")

        profile = scan_codebase(tmp_path)
        assert profile.capabilities == ()

    def test_scan_codebase_service_skips_missing_pyproject(self, tmp_path: Path) -> None:
        """Test scanning skips service directories without pyproject.toml."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        # Create a directory without pyproject.toml
        service_dir = services_dir / "incomplete_service"
        service_dir.mkdir()

        profile = scan_codebase(tmp_path)
        assert profile.capabilities == ()

    def test_scan_codebase_with_torchvision(self, tmp_path: Path) -> None:
        """Test scanning codebase with TorchVision dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "cv_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "cv-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
torch = "^2.0.0"
torchvision = "^0.15.0"
"""
        )

        profile = scan_codebase(tmp_path)

        assert "torchvision" in profile.ml_backends
        assert "pytorch" in profile.ml_backends
        assert "image_classification" in profile.task_types
        assert "object_detection" in profile.task_types

        cap_names = [c.name for c in profile.capabilities]
        assert "torchvision_cv" in cap_names

    def test_scan_codebase_with_pillow(self, tmp_path: Path) -> None:
        """Test scanning codebase with Pillow dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "img_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "img-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
pillow = "^10.0.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "image_processing" in cap_names

    def test_scan_codebase_with_opencv(self, tmp_path: Path) -> None:
        """Test scanning codebase with OpenCV dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "cv_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "cv-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
opencv-python = "^4.8.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "opencv_cv" in cap_names

    def test_scan_codebase_with_transformers(self, tmp_path: Path) -> None:
        """Test scanning codebase with Hugging Face Transformers dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "nlp_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "nlp-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
transformers = "^4.30.0"
datasets = "^2.14.0"
tokenizers = "^0.13.0"
sentencepiece = "^0.1.99"
"""
        )

        profile = scan_codebase(tmp_path)

        assert "transformers" in profile.ml_backends
        assert "text_classification" in profile.task_types
        assert "text_generation" in profile.task_types
        assert "summarization" in profile.task_types

        cap_names = [c.name for c in profile.capabilities]
        assert "huggingface_transformers" in cap_names
        assert "huggingface_datasets" in cap_names
        assert "tokenization" in cap_names
        assert "sentencepiece_tokenization" in cap_names

    def test_scan_codebase_with_catboost(self, tmp_path: Path) -> None:
        """Test scanning codebase with CatBoost dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "cb_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "cb-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
catboost = "^1.2.0"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        assert "catboost" in profile.ml_backends

    def test_scan_codebase_with_fasttext_wheel(self, tmp_path: Path) -> None:
        """Test scanning codebase with fasttext-wheel dependency (alternate name)."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        lib_dir = libs_dir / "lang_lib"
        lib_dir.mkdir()

        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "lang-lib"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
fasttext-wheel = "^0.9.2"
"""
        )

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "language_identification" in cap_names
