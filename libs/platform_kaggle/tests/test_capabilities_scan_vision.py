"""Tests for capabilities: scanning for vision and transformer libraries."""

from __future__ import annotations

from pathlib import Path

from platform_kaggle.capabilities import scan_codebase


class TestScanCodebaseVision:
    """Scanning for vision and transformer libraries."""

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
