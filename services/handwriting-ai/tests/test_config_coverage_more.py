from __future__ import annotations

from pathlib import Path

from handwriting_ai import _test_hooks
from handwriting_ai import config as cfg


def test_default_settings_create_dirs_false_does_not_create(tmp_path: Path) -> None:
    # The conftest's reset fixture restores the hook after the case.
    _test_hooks.working_dir = lambda: tmp_path
    out = cfg._default_settings(create_dirs=False)
    base = (tmp_path / ".handwriting-ai").resolve()
    assert not (base / "data").exists()
    assert not (base / "artifacts").exists()
    assert not (base / "logs").exists()
    assert not (base / "models").exists()
    assert out["app"]["data_root"] == base / "data"
    assert out["digits"]["model_dir"] == base / "models"


def test_default_settings_create_dirs_true_creates_under_the_working_dir(
    tmp_path: Path,
) -> None:
    _test_hooks.working_dir = lambda: tmp_path
    out = cfg._default_settings(create_dirs=True)
    base = (tmp_path / ".handwriting-ai").resolve()
    for name in ("data", "artifacts", "logs", "models"):
        assert (base / name).is_dir()
    assert out["app"]["logs_root"] == base / "logs"


def test_ensure_settings_defaults_its_roots_under_the_working_dir(tmp_path: Path) -> None:
    _test_hooks.working_dir = lambda: tmp_path
    partial: cfg.Settings = {
        "app": {"threads": 1, "port": 1},
        "digits": {"active_model": "default"},
        "security": {"api_key": ""},
    }
    out = cfg.ensure_settings(partial, create_dirs=False)
    base = (tmp_path / ".handwriting-ai").resolve()
    assert out["app"]["data_root"] == base / "data"
    assert out["app"]["artifacts_root"] == base / "artifacts"
    assert out["digits"]["model_dir"] == base / "models"
    assert not base.exists()


def test_the_working_dir_hook_is_the_process_directory() -> None:
    assert _test_hooks.working_dir() == Path.cwd()


def test_digits_job_load_settings_wrapper_invokes_loader() -> None:
    from handwriting_ai.jobs import digits as job_digits

    # Call wrapper and assert structure; this executes the shim and covers the line
    s = job_digits._load_settings()
    assert isinstance(s, dict) and "app" in s and "digits" in s and "security" in s
