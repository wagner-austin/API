"""Run ids must be unique even for identical, rapid submissions.

The id previously carried only a one-second timestamp, so submitting several
runs of the same family and size in a loop -- which is exactly what a
multi-arm experiment does -- produced one id shared by every run. Combined
with ``exist_ok=True`` on the artifact directory, the runs silently shared a
directory and overwrote each other's manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str

from model_trainer.infra.storage.run_store import RunStore


def test_rapid_identical_submissions_get_distinct_ids(tmp_path: Path) -> None:
    """Ten same-second creations of the same family and size must not collide."""
    store = RunStore(str(tmp_path))
    ids = [store.create_run("hf_lm", "small") for _ in range(10)]
    assert len(set(ids)) == 10


def test_each_run_gets_its_own_artifact_directory(tmp_path: Path) -> None:
    """Distinct ids must map to distinct directories, each with its manifest."""
    store = RunStore(str(tmp_path))
    first = store.create_run("hf_lm", "small")
    second = store.create_run("hf_lm", "small")
    assert first != second

    models_root = tmp_path / "models"
    assert sorted(p.name for p in models_root.iterdir()) == sorted([first, second])

    for run_id in (first, second):
        manifest = load_json_str((models_root / run_id / "manifest.json").read_text("utf-8"))
        if not isinstance(manifest, dict):
            raise AssertionError(f"expected dict manifest, got {type(manifest)}")
        assert manifest["run_id"] == run_id


def test_id_retains_family_and_size_prefix(tmp_path: Path) -> None:
    """The id stays human-readable, so operators can still identify a run."""
    run_id = RunStore(str(tmp_path)).create_run("hf_lm", "small")
    assert run_id.startswith("hf_lm-small-")


def test_colliding_directory_fails_loudly(tmp_path: Path) -> None:
    """A pre-existing artifact directory must raise, never be reused."""
    store = RunStore(str(tmp_path))
    run_id = store.create_run("hf_lm", "small")
    (tmp_path / "models" / run_id / "manifest.json").write_text("sentinel", encoding="utf-8")

    # Recreating the same directory is the failure the old exist_ok=True hid.
    with pytest.raises(FileExistsError):
        (tmp_path / "models" / run_id).mkdir(parents=True, exist_ok=False)

    # The original manifest is untouched, which is what silent sharing destroyed.
    assert (tmp_path / "models" / run_id / "manifest.json").read_text("utf-8") == "sentinel"
