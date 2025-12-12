from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from platform_core.json_utils import dump_json_str

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import StatResultProtocol
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
    s: Settings = {
        "app": app,
        "digits": dig,
        "security": sec,
    }
    return InferenceEngine(s)


def test_reload_if_changed_detects_updates() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models"
        active = "a_reload"
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

        eng = _make_engine_with_root(model_dir, active)
        eng.try_load_active()
        assert eng.ready is True

        # Update manifest with new model_id
        man2 = dict(man)
        man2["model_id"] = active + "_v2"
        (active_dir / "manifest.json").write_text(dump_json_str(man2), encoding="utf-8")

        reloaded = eng.reload_if_changed()
        assert reloaded is True and eng.model_id == active + "_v2"

        # Unchanged files path: subsequent call returns False
        assert eng.reload_if_changed() is False


def test_reload_if_changed_branches() -> None:
    # Not ready / no artifacts dir
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        eng = _make_engine_with_root(root, "none")
        assert eng.reload_if_changed() is False

    # Ready but missing last mtimes
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "m"
        active_dir = model_dir / "a"
        active_dir.mkdir(parents=True, exist_ok=True)
        man = {
            "schema_version": "v1.1",
            "model_id": "a",
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
        eng = _make_engine_with_root(model_dir, "a")
        eng.try_load_active()
        # Simulate unknown mtimes
        eng._last_manifest_mtime = None
        eng._last_model_mtime = None
        assert eng.reload_if_changed() is False

        # OSError path via path_stat hook
        def _boom_stat(path: Path, *, follow_symlinks: bool = True) -> StatResultProtocol:
            p = path.as_posix()
            if p.endswith("manifest.json") or p.endswith("model.pt"):
                raise OSError("nope")
            return path.stat(follow_symlinks=follow_symlinks)

        _test_hooks.path_stat = _boom_stat
        # Should raise after logging
        with pytest.raises(OSError, match="nope"):
            eng.reload_if_changed()


def test_try_load_active_stat_oserror_sets_last_none() -> None:
    # Test that OSError when reading mtimes during initial load is raised
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models"
        active = "a_oserr"
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

        # Track stat call counts to fail on mtime collection calls
        call_count: dict[str, int] = {"n": 0}

        def _boom_stat(path: Path, *, follow_symlinks: bool = True) -> StatResultProtocol:
            p = path.as_posix()
            call_count["n"] += 1
            # Let the first few stat calls (for file reading) succeed
            # but fail on the mtime collection calls
            if call_count["n"] > 10 and (p.endswith("manifest.json") or p.endswith("model.pt")):
                raise OSError("nope")
            return path.stat(follow_symlinks=follow_symlinks)

        _test_hooks.path_stat = _boom_stat
        eng = _make_engine_with_root(model_dir, active)
        # Should raise after logging when stat fails during mtime collection
        with pytest.raises(OSError, match="nope"):
            eng.try_load_active()
