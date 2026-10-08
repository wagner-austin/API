"""Tests for Kohya subprocess runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from art_trainer.core.services.training.backends.kohya import _test_hooks
from art_trainer.core.services.training.backends.kohya.runner import run_subprocess

from .testing import FakeKohyaRunner


def test_run_subprocess_with_fake_runner() -> None:
    """Test run_subprocess uses hook when set."""
    fake_runner = FakeKohyaRunner(should_succeed=True, final_loss=0.03)
    _test_hooks.Hooks.subprocess_runner = fake_runner

    result = run_subprocess(
        ["python", "train.py", "--config", "config.toml"],
        cwd=Path("/test/cwd"),
        timeout=3600,
    )

    assert result.returncode == 0
    assert result.stdout == "Training complete. loss=0.03"
    assert fake_runner.calls == [
        (["python", "train.py", "--config", "config.toml"], Path("/test/cwd"))
    ]


def test_run_subprocess_failure() -> None:
    """Test run_subprocess with failing command."""
    fake_runner = FakeKohyaRunner(
        should_succeed=False,
        returncode=1,
        stderr="CUDA out of memory",
    )
    _test_hooks.Hooks.subprocess_runner = fake_runner

    result = run_subprocess(["python", "train.py"], cwd=None, timeout=None)

    assert result.returncode == 1
    assert result.stderr == "CUDA out of memory"
    assert result.stdout == "Training failed"


def test_subprocess_result_impl_properties() -> None:
    """Test SubprocessResultImpl has correct property values."""
    from art_trainer.core.services.training.backends.kohya.runner import SubprocessResultImpl

    result = SubprocessResultImpl(
        returncode=42,
        stdout="test stdout",
        stderr="test stderr",
    )

    assert result.returncode == 42
    assert result.stdout == "test stdout"
    assert result.stderr == "test stderr"


def test_run_subprocess_real_subprocess(tmp_path: Path) -> None:
    """Test run_subprocess executes a real subprocess through the bound hook."""
    _test_hooks.reset_hooks()

    # Run a simple echo command
    result = run_subprocess(
        ["python", "-c", "print('hello_exact_output_12345')"],
        cwd=tmp_path,
        timeout=30,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "hello_exact_output_12345"


def test_run_subprocess_real_subprocess_no_cwd() -> None:
    """Test run_subprocess without cwd uses None."""
    _test_hooks.reset_hooks()

    result = run_subprocess(
        ["python", "-c", "print('test_exact_output_67890')"],
        cwd=None,
        timeout=30,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "test_exact_output_67890"


def test_real_runner_returns_a_failing_kohya_run_with_its_words(tmp_path: Path) -> None:
    """A training script that fails comes back as a status, never raised."""
    result = _test_hooks._real_subprocess_runner(
        [sys.executable, "-c", "import sys; sys.stderr.write('CUDA out of memory'); sys.exit(3)"],
        cwd=tmp_path,
        timeout=30,
    )

    assert result.returncode == 3
    assert result.stdout == ""
    assert result.stderr == "CUDA out of memory"


def test_real_runner_raises_when_a_run_outlasts_its_timeout() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        _test_hooks._real_subprocess_runner(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1,
        )
