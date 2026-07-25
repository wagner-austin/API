"""Tests for containment-checked model path resolution.

Exercises the real resolver against real filesystem paths under tmp_path — no
fakes, since the behaviour under test is filesystem resolution itself.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

from pytest import raises

from covenant_radar_api.core.model_paths import resolve_model_path


class TestResolveModelPathAccepts:
    """Paths that resolve inside the models root are accepted."""

    def test_direct_child_is_accepted(self, tmp_path: Path) -> None:
        """A model directly inside the root resolves to its absolute path.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        models_root = tmp_path / "models"
        models_root.mkdir()
        model = models_root / "active_xgb.ubj"
        model.write_bytes(b"stub")

        resolved = resolve_model_path(str(model), models_root)

        assert resolved == model.resolve()

    def test_nested_subdirectory_is_accepted(self, tmp_path: Path) -> None:
        """A model in a subdirectory of the root is accepted.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        models_root = tmp_path / "models"
        nested = models_root / "xgboost" / "v3"
        nested.mkdir(parents=True)
        model = nested / "model.ubj"
        model.write_bytes(b"stub")

        resolved = resolve_model_path(str(model), models_root)

        assert resolved == model.resolve()

    def test_traversal_that_lands_back_inside_root_is_accepted(self, tmp_path: Path) -> None:
        """`..` segments are fine as long as the result stays under the root.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        models_root = tmp_path / "models"
        (models_root / "a").mkdir(parents=True)
        model = models_root / "model.ubj"
        model.write_bytes(b"stub")

        resolved = resolve_model_path(str(models_root / "a" / ".." / "model.ubj"), models_root)

        assert resolved == model.resolve()

    def test_nonexistent_path_inside_root_is_accepted(self, tmp_path: Path) -> None:
        """Containment is a path check, not an existence check.

        Existence is enforced by the loaders, which raise FileNotFoundError.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        models_root = tmp_path / "models"
        models_root.mkdir()

        resolved = resolve_model_path(str(models_root / "absent.ubj"), models_root)

        assert resolved == (models_root / "absent.ubj").resolve()


class TestResolveModelPathRejects:
    """Paths that escape the models root are rejected."""

    def test_parent_traversal_is_rejected(self, tmp_path: Path) -> None:
        """A `..` sequence escaping the root raises ValueError.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        models_root = tmp_path / "models"
        models_root.mkdir()
        secret = tmp_path / "secret.ubj"
        secret.write_bytes(b"stub")

        with raises(ValueError, match="must resolve inside the models root"):
            resolve_model_path(str(models_root / ".." / "secret.ubj"), models_root)

    def test_absolute_path_outside_root_is_rejected(self, tmp_path: Path) -> None:
        """An unrelated absolute path raises ValueError.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        models_root = tmp_path / "models"
        models_root.mkdir()
        outside = tmp_path / "elsewhere" / "model.ubj"
        outside.parent.mkdir()
        outside.write_bytes(b"stub")

        with raises(ValueError, match="must resolve inside the models root"):
            resolve_model_path(str(outside), models_root)

    def test_sibling_prefix_directory_is_rejected(self, tmp_path: Path) -> None:
        """A sibling whose name merely starts with the root's name is rejected.

        Guards against a containment check written as a string prefix
        comparison, which would wrongly accept `models_evil/`.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        models_root = tmp_path / "models"
        models_root.mkdir()
        sibling = tmp_path / "models_evil"
        sibling.mkdir()
        model = sibling / "model.ubj"
        model.write_bytes(b"stub")

        with raises(ValueError, match="must resolve inside the models root"):
            resolve_model_path(str(model), models_root)

    def test_error_names_both_paths(self, tmp_path: Path) -> None:
        """The error message carries the offending path and the root.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        models_root = tmp_path / "models"
        models_root.mkdir()
        outside = tmp_path / "outside.ubj"

        with raises(ValueError) as excinfo:
            resolve_model_path(str(outside), models_root)

        message = str(excinfo.value)
        assert str(outside.resolve()) in message
        assert str(models_root.resolve()) in message
