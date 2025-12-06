from __future__ import annotations

import io
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from platform_core.json_utils import dump_json_str
from platform_core.logging import JsonFormatter, get_logger

from handwriting_ai.config import (
    AppConfig,
    DigitsConfig,
    SecurityConfig,
    Settings,
)
from handwriting_ai.inference.engine import InferenceEngine, build_fresh_state_dict
from handwriting_ai.preprocess import preprocess_signature

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def _make_engine_with_root(root: Path, active: str) -> InferenceEngine:
    dig: DigitsConfig = {
        "model_dir": root,
        "active_model": active,
        "tta": False,
        "uncertain_threshold": 0.70,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    app: AppConfig = {
        "data_root": Path("/tmp/data"),
        "artifacts_root": Path("/tmp/artifacts"),
        "logs_root": Path("/tmp/logs"),
        "threads": 0,
        "port": 8081,
    }
    sec: SecurityConfig = {"api_key": ""}
    s: Settings = {"app": app, "digits": dig, "security": sec}
    return InferenceEngine(s)


def test_try_load_active_model_build_failure_is_logged_and_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models"
        active = "fail_build"
        active_dir = model_dir / active
        active_dir.mkdir(parents=True, exist_ok=True)

        man = {
            "schema_version": "v1.1",
            "model_id": active,
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "preprocess_hash": preprocess_signature(),
            "val_acc": 0.99,
            "temperature": 1.0,
        }
        (active_dir / "manifest.json").write_text(dump_json_str(man), encoding="utf-8")
        sd = build_fresh_state_dict("resnet18", 10)
        torch.save(sd, (active_dir / "model.pt").as_posix())

        # Force _build_model to raise
        from handwriting_ai import inference as _inf

        def _boom(arch: str, n_classes: int) -> None:
            raise RuntimeError("broken torchvision")

        monkeypatch.setattr(_inf.engine, "_build_model", _boom, raising=True)

        eng = _make_engine_with_root(model_dir, active)
        buf = io.StringIO()
        logger = get_logger("handwriting_ai")
        # Use standard logging handler with JSON formatter
        import logging as _logging

        handler = _logging.StreamHandler(buf)
        handler.setFormatter(JsonFormatter(static_fields={}, extra_field_names=[]))
        logger.addHandler(handler)
        try:
            # Should raise after logging
            with pytest.raises(RuntimeError, match="broken torchvision"):
                eng.try_load_active()
        finally:
            logger.removeHandler(handler)
        out = buf.getvalue()
        assert "model_build_failed" in out
        assert eng.ready is False
