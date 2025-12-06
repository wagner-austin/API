from __future__ import annotations

from pathlib import Path

import pytest

from handwriting_ai import config as cfg


def test_default_settings_create_dirs_false_does_not_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = cfg._default_settings(create_dirs=False)
    base = tmp_path / ".handwriting-ai"
    assert not (base / "data").exists()
    assert not (base / "artifacts").exists()
    assert not (base / "logs").exists()
    assert not (base / "models").exists()
    assert out["app"]["data_root"].as_posix().endswith(".handwriting-ai/data")
    assert out["digits"]["model_dir"].as_posix().endswith(".handwriting-ai/models")


def test_digits_job_load_settings_wrapper_invokes_loader() -> None:
    from handwriting_ai.jobs import digits as job_digits

    # Call wrapper and assert structure; this executes the shim and covers the line
    s = job_digits._load_settings()
    assert isinstance(s, dict) and "app" in s and "digits" in s and "security" in s
