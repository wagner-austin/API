"""Tests for the checks that prove a run can finish before it starts.

Two of these reproduce failures that actually cost A100 time, and they are
written as the failure rather than as the fix: a directory that exists but
cannot be written, and an artifact store whose credentials are absent. Both
were checkable in under a second and both were discovered only after the
expensive work was already done.
"""

from __future__ import annotations

import hashlib
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
        preflight.check_writable({"artifacts": tmp_path / "a", "runs": tmp_path / "b"}, token="r1")
        assert (tmp_path / "a").is_dir()
        assert (tmp_path / "b").is_dir()

    def test_it_leaves_no_probe_behind(self, tmp_path: pathlib.Path) -> None:
        preflight.check_writable({"artifacts": tmp_path / "a"}, token="r1")
        assert list((tmp_path / "a").iterdir()) == []

    def test_a_root_that_cannot_be_created_is_refused(self, tmp_path: pathlib.Path) -> None:
        """The failure that actually happened: PermissionError on /data/artifacts,
        discovered at the first epoch boundary of a 20-epoch run."""
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("I am a file", encoding="utf-8")
        with pytest.raises(AppError) as excinfo:
            preflight.check_writable({"APP__ARTIFACTS_ROOT": blocker / "artifacts"}, token="r1")
        assert excinfo.value.code is ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED

    def test_the_refusal_names_the_setting_not_only_the_path(self, tmp_path: pathlib.Path) -> None:
        """The operator has to change a setting, so the message names it."""
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(AppError) as excinfo:
            preflight.check_writable({"APP__RUNS_ROOT": blocker / "runs"}, token="r1")
        assert "APP__RUNS_ROOT" in excinfo.value.message
        assert "train to completion and then fail saving" in excinfo.value.message

    def test_two_runs_sharing_a_root_do_not_delete_each_other_s_probe(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The failure that killed arm B of the Kazakh A/B, 19 seconds in.

        Both arms wrote `.preflight-probe` into the same artifacts root on a
        shared filesystem, seconds apart. Arm A's cleanup removed the file arm
        B had just written, so B's own unlink raised FileNotFoundError and the
        check meant to protect the run is what ended it.

        Interleaved deliberately: run A writes, run B writes, A finishes, B
        finishes. With one shared name that ordering cannot survive.
        """
        shared = {"APP__ARTIFACTS_ROOT": tmp_path / "shared"}
        (tmp_path / "shared").mkdir()

        probe_a = tmp_path / "shared" / f"{preflight.PROBE_NAME}-armA"
        probe_b = tmp_path / "shared" / f"{preflight.PROBE_NAME}-armB"
        assert probe_a != probe_b

        preflight.check_writable(shared, token="armA")
        preflight.check_writable(shared, token="armB")

        assert list((tmp_path / "shared").iterdir()) == []

    def test_every_root_is_checked_not_just_the_first(self, tmp_path: pathlib.Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(AppError) as excinfo:
            preflight.check_writable(
                {"good": tmp_path / "fine", "APP__LOGS_ROOT": blocker / "logs"}, token="r1"
            )
        assert "APP__LOGS_ROOT" in excinfo.value.message


class TestCheckArtifactRoundTrip:
    def test_a_working_store_passes(self, tmp_path: pathlib.Path) -> None:
        store = LocalArtifacts(tmp_path / "artifacts")
        preflight.check_artifact_round_trip(
            store, tmp_path / "scratch", tmp_path / "artifacts", token="r1"
        )

    def test_it_leaves_nothing_behind_in_the_output_directory(self, tmp_path: pathlib.Path) -> None:
        """A check that litters makes the run's own output harder to read
        every time it passes -- which is every time. The first version left a
        300-byte probe tarball beside two 462 MB models."""
        artifacts = tmp_path / "artifacts"
        preflight.check_artifact_round_trip(
            LocalArtifacts(artifacts), tmp_path / "scratch", artifacts, token="r1"
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
            LocalArtifacts(artifacts), tmp_path / "scratch", artifacts, token="r1"
        )
        assert real.read_bytes() == b"a real trained model\n"
        assert [p.name for p in artifacts.iterdir()] == [real.name]

    def test_a_sibling_arm_s_probe_artifact_survives_this_run_s_sweep(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Same collision, one layer down. The sweep used to glob the bare
        probe name, so a concurrent arm's probe artifact was removed out from
        under its round trip. Scoping the glob to this run's token fixes it."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        sibling = artifacts / f"{preflight.PROBE_ARTIFACT}-armA-0000.tar.gz"
        sibling.write_bytes(b"arm A is mid-round-trip\n")

        preflight.check_artifact_round_trip(
            LocalArtifacts(artifacts), tmp_path / "scratch", artifacts, token="armB"
        )

        assert sibling.read_bytes() == b"arm A is mid-round-trip\n"
        assert [p.name for p in artifacts.iterdir()] == [sibling.name]

    def test_a_store_that_returns_the_wrong_bytes_is_refused(self, tmp_path: pathlib.Path) -> None:
        """A configuration check cannot see this: the store answers, it just
        answers wrong. A finished run would be saved incorrectly rather than
        not at all, which is the worse of the two."""
        with pytest.raises(AppError) as excinfo:
            preflight.check_artifact_round_trip(
                _RefusingStore(tmp_path / "artifacts"),
                tmp_path / "scratch",
                tmp_path / "artifacts",
                token="r1",
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
        preflight.check_artifact_round_trip(
            store, tmp_path / "scratch", tmp_path / "artifacts", token="r1"
        )
        assert (tmp_path / "artifacts").is_dir()


def _stage(corpus_dir: pathlib.Path, body: bytes, *, certified: bool) -> str:
    """Place a corpus keyed by its true digest, optionally certified.

    Args:
        corpus_dir: Directory to place it in.
        body: Corpus bytes.
        certified: Whether to write a record admitting the digest.

    Returns:
        The digest the corpus is stored under.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(body).hexdigest()
    (corpus_dir / digest).write_bytes(body)
    if certified:
        (corpus_dir / f"turkic-mi-v3{preflight.CERTIFICATION_SUFFIX}").write_text(
            f"# certified by the audit gate\n{digest}  oscar_kk_ipa.txt\n",
            encoding="utf-8",
        )
    return digest


class TestCheckCorpusCertified:
    def test_a_staged_certified_corpus_passes(self, tmp_path: pathlib.Path) -> None:
        digest = _stage(tmp_path / "corpora", b"kynI ospw bojenSa\n", certified=True)
        preflight.check_corpus_certified(tmp_path / "corpora", digest)

    def test_an_absent_corpus_is_refused(self, tmp_path: pathlib.Path) -> None:
        corpora = tmp_path / "corpora"
        corpora.mkdir()
        with pytest.raises(AppError) as excinfo:
            preflight.check_corpus_certified(corpora, "0" * 64)
        assert excinfo.value.code is ModelTrainerErrorCode.CORPUS_EMPTY
        assert "stage it first" in excinfo.value.message

    def test_a_corpus_whose_bytes_do_not_match_its_name_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The corpus is addressed BY digest, and nothing was checking the
        bytes at that name actually hash to it. A file named after a digest it
        does not have is indistinguishable from the real one until the
        results are wrong."""
        corpora = tmp_path / "corpora"
        digest = _stage(corpora, b"the real corpus\n", certified=True)
        (corpora / digest).write_bytes(b"something else entirely\n")

        with pytest.raises(AppError) as excinfo:
            preflight.check_corpus_certified(corpora, digest)

        assert excinfo.value.code is ModelTrainerErrorCode.CORPUS_EMPTY
        assert "hash to" in excinfo.value.message

    def test_an_uncertified_corpus_is_refused(self, tmp_path: pathlib.Path) -> None:
        """The failure this exists for. The corpus hashed correctly to its own
        name and was still the wrong thing -- raw OSCAR English concatenated
        with a wiki export, assembled by hand and copied up. It trained to
        completion twice and reported perplexities for cookie banners."""
        digest = _stage(tmp_path / "corpora", b"raw scrape, no provenance\n", certified=False)

        with pytest.raises(AppError) as excinfo:
            preflight.check_corpus_certified(tmp_path / "corpora", digest)

        assert excinfo.value.code is ModelTrainerErrorCode.CORPUS_EMPTY
        assert "0 record(s) found" in excinfo.value.message

    def test_a_record_naming_other_corpora_does_not_admit_this_one(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A record present is not a record that admits YOU. Staging seven
        languages certifies seven digests, not the eighth file someone
        dropped in beside them."""
        corpora = tmp_path / "corpora"
        digest = _stage(corpora, b"uncertified newcomer\n", certified=False)
        (corpora / f"other{preflight.CERTIFICATION_SUFFIX}").write_text(
            f"{'a' * 64}  oscar_tr_ipa.txt\n{'b' * 64}  oscar_uz_ipa.txt\n",
            encoding="utf-8",
        )

        with pytest.raises(AppError) as excinfo:
            preflight.check_corpus_certified(corpora, digest)

        assert "1 record(s) found, 2 digest(s)" in excinfo.value.message

    def test_digests_are_read_from_every_record_present(self, tmp_path: pathlib.Path) -> None:
        """The suffix is a suffix, not a fixed name, so a second certification
        run adds a record rather than overwriting the first."""
        corpora = tmp_path / "corpora"
        digest = _stage(corpora, b"certified by the later run\n", certified=False)
        (corpora / f"first{preflight.CERTIFICATION_SUFFIX}").write_text(
            f"{'c' * 64}\n", encoding="utf-8"
        )
        (corpora / f"second{preflight.CERTIFICATION_SUFFIX}").write_text(
            f"{digest}\n", encoding="utf-8"
        )

        preflight.check_corpus_certified(corpora, digest)

    def test_a_corpus_larger_than_one_read_block_hashes_correctly(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Real corpora are ~15 MB and are read in blocks; a digest computed
        from only the first block would admit a truncated file."""
        body = b"".join(f"line {i}\n".encode() for i in range(200_000))
        assert len(body) > 1 << 20
        digest = _stage(tmp_path / "corpora", body, certified=True)
        preflight.check_corpus_certified(tmp_path / "corpora", digest)


class TestTheBaseModelIsResolvableBeforeAGpuIsSpent:
    """The other input a run cannot recover from, checked the same way.

    This half did not exist until 2026-09-04, and the asymmetry cost a real
    job: 55744648 was handed an A30, checked every output root, round-tripped
    the artifact store, certified the corpus -- and then died nine seconds in
    because the base model could not be resolved from a staged cache.

    The cause is worth carrying, because the mistake is the CAREFUL one.
    The model had been fetched with a pinned commit hash, which is the right
    instinct. But huggingface_hub writes its ``refs/<revision>`` pointer only
    when the requested revision differs from the resolved commit hash, so
    pinning by hash writes no ref at all -- and ``from_pretrained("<repo>")``
    asks for ``main``, which offline resolves only through ``refs/main``.
    Every byte was present and none of it was reachable.
    """

    def _resolvable_model(self, tmp_path: pathlib.Path) -> str:
        """Write the smallest directory ``AutoConfig`` will resolve.

        A local path rather than a repo id, so the check exercises its real
        resolution call without reaching the network or depending on a cache
        this test did not build.

        Args:
            tmp_path: The test's temporary directory.

        Returns:
            The directory, as a string.
        """
        directory = tmp_path / "tiny-model"
        directory.mkdir()
        (directory / "config.json").write_text('{"model_type": "gpt2"}', encoding="utf-8")
        return str(directory)

    def test_a_resolvable_model_passes(self, tmp_path: pathlib.Path) -> None:
        preflight.check_model_available(self._resolvable_model(tmp_path))

    def test_a_model_that_cannot_be_resolved_is_refused(self, tmp_path: pathlib.Path) -> None:
        """The failure that cost job 55744648, as a test rather than a story."""
        missing = str(tmp_path / "nothing-here")

        with pytest.raises(AppError) as excinfo:
            preflight.check_model_available(missing)

        assert excinfo.value.code is ModelTrainerErrorCode.MODEL_NOT_FOUND

    def test_the_refusal_explains_the_ref_that_pinning_omits(self, tmp_path: pathlib.Path) -> None:
        """The message has to carry the fix, because the cause is not guessable.

        An operator reading "cannot resolve" against a cache they can SEE the
        model sitting in has no reason to suspect a missing pointer file.
        """
        with pytest.raises(AppError) as excinfo:
            preflight.check_model_available(str(tmp_path / "nothing-here"))

        assert "refs/" in excinfo.value.message
