from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.config import _test_hooks as config_test_hooks
from platform_core.testing import make_fake_env

from handwriting_ai.config import load_settings


def test_app_port_out_of_range_raises() -> None:
    config_test_hooks.get_env = make_fake_env({"APP__PORT": "70000"})
    with pytest.raises(RuntimeError):
        _ = load_settings()


def test_env_overrides_happy_paths(tmp_path: Path) -> None:
    config_test_hooks.get_env = make_fake_env(
        {
            "DIGITS__MODEL_DIR": (tmp_path / "models").as_posix(),
            "DIGITS__VISUALIZE_MAX_KB": "2",
            "DIGITS__PREDICT_TIMEOUT_SECONDS": "1",
        }
    )
    s = load_settings()
    assert s["digits"]["model_dir"].as_posix().endswith("models")
    assert int(s["digits"]["visualize_max_kb"]) == 2
    assert int(s["digits"]["predict_timeout_seconds"]) == 1
