"""Tests for scripts/optimize runner functions.

Tests unified run_backend function for all backends.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import scripts._test_hooks as _hooks
from scripts._test_hooks import (
    LoadingProgressCallbackProtocol,
    PhaseProgressCallbackProtocol,
    TrialProgressCallbackProtocol,
    UnifiedOptimizationResult,
)
from scripts.optimize.runner import (
    get_project_root,
    run_backend,
)

from .conftest import make_fake_result


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_returns_parent_of_scripts(self) -> None:
        """Test default project root hook returns parent of scripts directory."""
        from scripts._test_hooks import _default_project_root

        root: Path = _default_project_root()
        assert root.name == "covenant-radar-api"
        assert (root / "scripts").exists()

    def test_hook_is_used(self, tmp_path: Path) -> None:
        """Test get_project_root delegates to project_root_hook."""
        root: Path = get_project_root()
        assert root == tmp_path


class TestRunBackend:
    """Tests for unified run_backend function."""

    def _make_fake_runner(
        self,
        result: UnifiedOptimizationResult,
        call_args: list[tuple[str, Path, Path]],
    ) -> _hooks.OptimizationRunnerProtocol:
        """Create a fake runner that records calls.

        Args:
            result: Result to return.
            call_args: List to append (config_json, external_dir, output_dir).

        Returns:
            Fake runner matching OptimizationRunnerProtocol.
        """

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            _ = progress_callback
            _ = phase_callback
            _ = loading_progress_callback
            call_args.append((config_json, external_dir, output_dir))
            return result

        return fake_runner

    def test_runs_xgboost_with_hook(self) -> None:
        """Test run_backend for xgboost uses optimization_runner hook."""
        fake_result = make_fake_result()
        call_args: list[tuple[str, Path, Path]] = []

        original = _hooks.optimization_runner
        _hooks.optimization_runner = self._make_fake_runner(fake_result, call_args)
        try:
            result = run_backend("xgboost", "taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "taiwan" in config_json
            assert "10" in config_json
            assert "full" in config_json
            assert '"xgboost"' in config_json
        finally:
            _hooks.optimization_runner = original

    def test_runs_mlp_with_hook(self) -> None:
        """Test run_backend for mlp uses optimization_runner hook."""
        fake_result = make_fake_result(backend="mlp")
        call_args: list[tuple[str, Path, Path]] = []

        original = _hooks.optimization_runner
        _hooks.optimization_runner = self._make_fake_runner(fake_result, call_args)
        try:
            result = run_backend("mlp", "taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert '"mlp"' in config_json
        finally:
            _hooks.optimization_runner = original

    def test_runs_lightgbm_with_hook(self) -> None:
        """Test run_backend for lightgbm uses optimization_runner hook."""
        fake_result = make_fake_result(backend="lightgbm")
        call_args: list[tuple[str, Path, Path]] = []

        original = _hooks.optimization_runner
        _hooks.optimization_runner = self._make_fake_runner(fake_result, call_args)
        try:
            result = run_backend("lightgbm", "taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert '"lightgbm"' in config_json
        finally:
            _hooks.optimization_runner = original

    def test_runs_lstm_with_hook(self) -> None:
        """Test run_backend for lstm uses optimization_runner hook."""
        fake_result = make_fake_result(backend="lstm")
        call_args: list[tuple[str, Path, Path]] = []

        original = _hooks.optimization_runner
        _hooks.optimization_runner = self._make_fake_runner(fake_result, call_args)
        try:
            result = run_backend("lstm", "taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert '"lstm"' in config_json
        finally:
            _hooks.optimization_runner = original

    def test_runs_cleargbm_with_hook(self) -> None:
        """Test run_backend for cleargbm uses optimization_runner hook."""
        fake_result = make_fake_result(backend="cleargbm")
        call_args: list[tuple[str, Path, Path]] = []

        original = _hooks.optimization_runner
        _hooks.optimization_runner = self._make_fake_runner(fake_result, call_args)
        try:
            result = run_backend("cleargbm", "taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert '"cleargbm"' in config_json
        finally:
            _hooks.optimization_runner = original

    def test_includes_timeout_when_provided(self) -> None:
        """Test run_backend includes timeout in config when provided."""
        fake_result = make_fake_result()
        call_args: list[tuple[str, Path, Path]] = []

        original = _hooks.optimization_runner
        _hooks.optimization_runner = self._make_fake_runner(fake_result, call_args)
        try:
            run_backend("xgboost", "taiwan", 10, "full", "cpu", 60)
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "timeout_seconds" in config_json
            assert "60" in config_json
        finally:
            _hooks.optimization_runner = original
