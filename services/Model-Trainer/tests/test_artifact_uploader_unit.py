"""Saving a finished run's model directory through the production artifact store.

``_upload_and_persist_pointer`` packs the run's output into the store under
``artifacts_root/store`` and records the stored tarball's digest as the run's
artifact pointer. These tests drive the real filesystem store the default hook
builds, so what they read back is a tarball on disk, not a fake's answer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from platform_core.trainer_keys import artifact_file_id_key
from platform_workers.testing import FakeRedis

from model_trainer.cluster.stores import LocalArtifacts
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.data.store_roots import artifact_store_root
from model_trainer.worker import train_job_lifecycle


def test_the_pointer_names_the_tarball_written_under_the_store_root(
    tmp_path: Path, settings_with_paths: Settings
) -> None:
    base = tmp_path / "model-run1"
    base.mkdir()
    (base / "weights.bin").write_bytes(b"weights")
    r = FakeRedis()

    file_id, size = train_job_lifecycle._upload_and_persist_pointer(
        settings_with_paths, r, run_id="run1", out_dir=str(base)
    )

    store_root = artifact_store_root(settings_with_paths)
    tarballs = sorted(store_root.glob("*.tar.gz"))
    assert [t.name for t in tarballs] == [f"model-run1-{file_id[:12]}.tar.gz"]
    assert hashlib.sha256(tarballs[0].read_bytes()).hexdigest() == file_id
    assert size == tarballs[0].stat().st_size
    assert r.get(artifact_file_id_key("run1")) == file_id
    r.assert_only_called({"set", "get"})


def test_the_stored_model_reads_back_through_the_same_store(
    tmp_path: Path, settings_with_paths: Settings
) -> None:
    base = tmp_path / "model-run2"
    base.mkdir()
    (base / "weights.bin").write_bytes(b"trained")
    r = FakeRedis()

    file_id, _ = train_job_lifecycle._upload_and_persist_pointer(
        settings_with_paths, r, run_id="run2", out_dir=str(base)
    )

    store = LocalArtifacts(artifact_store_root(settings_with_paths))
    restored = store.download_artifact(
        file_id, dest_dir=tmp_path / "restored", request_id="run2", expected_root="model-run2"
    )
    assert (restored / "weights.bin").read_bytes() == b"trained"
    r.assert_only_called({"set"})


def test_a_missing_output_directory_records_no_pointer_and_leaves_no_staging_file(
    tmp_path: Path, settings_with_paths: Settings
) -> None:
    r = FakeRedis()

    with pytest.raises(FileNotFoundError):
        train_job_lifecycle._upload_and_persist_pointer(
            settings_with_paths, r, run_id="run3", out_dir=str(tmp_path / "never-written")
        )

    assert sorted(p.name for p in artifact_store_root(settings_with_paths).iterdir()) == []
    r.assert_only_called(set())
