"""Tests for the checks that prove a run can finish before it starts.

Two of these reproduce failures that actually cost A100 time, and they are
written as the failure rather than as the fix: a directory that exists but
cannot be written, and an artifact store whose credentials are absent. Both
were checkable in under a second and both were discovered only after the
expensive work was already done.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.cluster import preflight
from model_trainer.cluster.stores import LocalArtifacts


class _RefusingStore:
    """An artifact store that accepts an upload and returns the wrong bytes.

    Stands for every way a store can be reachable and still not do its job --
    the class of failure a configuration check cannot see.
    """

    __slots__ = ("root",)

    def __init__(self, root: pathlib.Path) -> None:
        """Record where the real store would have written.

        Args:
            root: Directory the honest store uses.
        """
        self.root = root

    def upload_artifact(
        self, dir_path: pathlib.Path, *, artifact_name: str, request_id: str
    ) -> FileUploadResponse:
        """Claim success without storing anything retrievable.

        Args:
            dir_path: Directory that would have been packed.
            artifact_name: Name for the artifact.
            request_id: Correlation id.

        Returns:
            A plausible response naming bytes that are not there.
        """
        return FileUploadResponse(
            file_id="0" * 64,
            size=1,
            sha256="0" * 64,
            content_type="application/gzip",
            created_at=None,
        )

    def download_artifact(
        self, file_id: str, *, dest_dir: pathlib.Path, request_id: str, expected_root: str
    ) -> pathlib.Path:
        """Return a directory holding different bytes than were uploaded.

        Args:
            file_id: Digest requested.
            dest_dir: Directory to extract into.
            request_id: Correlation id.
            expected_root: Directory name expected inside.

        Returns:
            A path whose probe file has the wrong contents.
        """
        out = dest_dir / expected_root
        out.mkdir(parents=True, exist_ok=True)
        (out / "probe.txt").write_bytes(b"not what went in\n")
        return out


class TestCheckWritable:
    def test_writable_roots_pass(self, tmp_path: pathlib.Path) -> None:
        preflight.check_writable({"artifacts": tmp_path / "a", "runs": tmp_path / "b"})
        assert (tmp_path / "a").is_dir()
        assert (tmp_path / "b").is_dir()

    def test_it_leaves_no_probe_behind(self, tmp_path: pathlib.Path) -> None:
        preflight.check_writable({"artifacts": tmp_path / "a"})
        assert list((tmp_path / "a").iterdir()) == []

    def test_a_root_that_cannot_be_created_is_refused(self, tmp_path: pathlib.Path) -> None:
        """The failure that actually happened: PermissionError on /data/artifacts,
        discovered at the first epoch boundary of a 20-epoch run."""
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("I am a file", encoding="utf-8")
        with pytest.raises(AppError) as excinfo:
            preflight.check_writable({"APP__ARTIFACTS_ROOT": blocker / "artifacts"})
        assert excinfo.value.code is ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED

    def test_the_refusal_names_the_setting_not_only_the_path(self, tmp_path: pathlib.Path) -> None:
        """The operator has to change a setting, so the message names it."""
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(AppError) as excinfo:
            preflight.check_writable({"APP__RUNS_ROOT": blocker / "runs"})
        assert "APP__RUNS_ROOT" in excinfo.value.message
        assert "train to completion and then fail saving" in excinfo.value.message

    def test_every_root_is_checked_not_just_the_first(self, tmp_path: pathlib.Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(AppError) as excinfo:
            preflight.check_writable(
                {"good": tmp_path / "fine", "APP__LOGS_ROOT": blocker / "logs"}
            )
        assert "APP__LOGS_ROOT" in excinfo.value.message


class TestCheckArtifactRoundTrip:
    def test_a_working_store_passes(self, tmp_path: pathlib.Path) -> None:
        store = LocalArtifacts(tmp_path / "artifacts")
        preflight.check_artifact_round_trip(store, tmp_path / "scratch", tmp_path / "artifacts")

    def test_it_leaves_nothing_behind_in_the_output_directory(self, tmp_path: pathlib.Path) -> None:
        """A check that litters makes the run's own output harder to read
        every time it passes -- which is every time. The first version left a
        300-byte probe tarball beside two 462 MB models."""
        artifacts = tmp_path / "artifacts"
        preflight.check_artifact_round_trip(
            LocalArtifacts(artifacts), tmp_path / "scratch", artifacts
        )
        assert list(artifacts.iterdir()) == []
        assert not (tmp_path / "scratch").exists()

    def test_cleanup_does_not_touch_the_run_s_real_output(self, tmp_path: pathlib.Path) -> None:
        """The sweep is scoped to the probe's own name. A cleanup that took
        the run's model with it would be far worse than the litter."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        real = artifacts / "model-abl-armC-a-seed42-8821d462f859.tar.gz"
        real.write_bytes(b"a real trained model\n")

        preflight.check_artifact_round_trip(
            LocalArtifacts(artifacts), tmp_path / "scratch", artifacts
        )
        assert real.read_bytes() == b"a real trained model\n"
        assert [p.name for p in artifacts.iterdir()] == [real.name]

    def test_a_store_that_returns_the_wrong_bytes_is_refused(self, tmp_path: pathlib.Path) -> None:
        """A configuration check cannot see this: the store answers, it just
        answers wrong. A finished run would be saved incorrectly rather than
        not at all, which is the worse of the two."""
        with pytest.raises(AppError) as excinfo:
            preflight.check_artifact_round_trip(
                _RefusingStore(tmp_path / "artifacts"),
                tmp_path / "scratch",
                tmp_path / "artifacts",
            )
        assert excinfo.value.code is ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED
        assert "different bytes" in excinfo.value.message

    def test_an_unconfigured_http_store_is_refused_at_construction(self) -> None:
        """The failure that cost 49 minutes. The credential check now belongs
        to the store that needs credentials, so it fires when that store is
        BUILT -- during preflight -- rather than after training finishes."""
        from model_trainer.core._hook_defaults import _default_artifact_store

        with pytest.raises(AppError) as excinfo:
            _default_artifact_store("", "")
        assert excinfo.value.code is ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED

    def test_a_store_needing_no_credentials_is_not_refused(self, tmp_path: pathlib.Path) -> None:
        """The other half: a filesystem store was refused for lacking
        credentials it never uses. That refusal came from the CALLER, which
        could not know what the store required."""
        store = LocalArtifacts(tmp_path / "artifacts")
        preflight.check_artifact_round_trip(store, tmp_path / "scratch", tmp_path / "artifacts")
        assert (tmp_path / "artifacts").is_dir()
