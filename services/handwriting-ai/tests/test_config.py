from __future__ import annotations

from pathlib import Path

import pytest

from handwriting_ai.config import load_settings


def test_app_port_out_of_range_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"APP__PORT": "70000"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(RuntimeError):
        _ = load_settings()


def test_env_overrides_happy_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = {
        "DIGITS__MODEL_DIR": (tmp_path / "models").as_posix(),
        "DIGITS__VISUALIZE_MAX_KB": "2",
        "DIGITS__PREDICT_TIMEOUT_SECONDS": "1",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = load_settings()
    assert s["digits"]["model_dir"].as_posix().endswith("models")
    assert int(s["digits"]["visualize_max_kb"]) == 2
    assert int(s["digits"]["predict_timeout_seconds"]) == 1
