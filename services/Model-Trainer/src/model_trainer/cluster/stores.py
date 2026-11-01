"""The filesystem stores every run reads corpora from and writes artifacts to.

They were written for a cluster run, since a compute node has no data-bank API
and no artifact service, only a filesystem and a corpus someone put there on
purpose. When data-bank-api was archived on 2026-11-01 (MCPs board task
a9566810) the service deployment's HTTP-backed stores went with it, and these
became the production default behind both hooks for every run; a cluster run
binds them to the directories its flags name.

The corpus is the interesting one. A file id names a file that ``hpc3-stage``
(or whoever staged it) already placed and verified by SHA-256. That is a
stronger guarantee than a fetch gives -- but only if the id really is the
digest, so a mismatch is refused rather than downloaded around, because there
is nothing to download from.
"""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
from pathlib import Path

from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.logging import get_logger

from model_trainer.core.contracts.stored_artifact import StoredArtifact

_log = get_logger(__name__)

_TARBALL_SUFFIX = ".tar.gz"
"""What the service artifact store produces, matched so a run's output is
readable by the same tooling wherever it was produced."""


class StagedCorpus:
    """Resolves a corpus file id to a file already present on the cluster.

    Attributes:
        root: Directory holding staged corpora.
    """

    __slots__ = ("root",)

    def __init__(self, root: Path) -> None:
        """Bind the fetcher to a staging directory.

        Args:
            root: Directory ``hpc3-stage`` placed the corpora in.
        """
        self.root = root

    def fetch(self, file_id: str) -> Path:
        """Resolve a file id to its staged path.

        Args:
            file_id: The corpus's identity, which is its SHA-256 digest.

        Returns:
            Path to the staged file.

        Raises:
            AppError: With ``CORPUS_NOT_FOUND`` when no staged file carries
                that id. The message names the directory and what is in it,
                because the fix is a staging step the operator forgot rather
                than anything this process can do.
        """
        candidate = self.root / file_id
        if candidate.is_file():
            return candidate

        present = sorted(p.name for p in self.root.iterdir()) if self.root.is_dir() else []
        raise AppError(
            ModelTrainerErrorCode.CORPUS_NOT_FOUND,
            f"No staged corpus {file_id!r} under {self.root}. "
            f"Staged there: {present}. "
            "Nothing fetches a corpus; stage it there first (hpc3-stage on a compute node).",
        )


class LocalArtifacts:
    """Writes run artifacts to the filesystem as digest-named tarballs.

    Attributes:
        root: Directory receiving tarballs.
    """

    __slots__ = ("root",)

    def __init__(self, root: Path) -> None:
        """Bind the store to an output directory.

        Args:
            root: Directory to write tarballs into. Created on first write.
        """
        self.root = root

    def upload_artifact(
        self,
        dir_path: Path,
        *,
        artifact_name: str,
        request_id: str,
    ) -> StoredArtifact:
        """Pack a directory into a tarball beside the run's other output.

        The digest is computed from the bytes actually written rather than
        reported from memory, so the value a run records is a fact about the
        file on disk. That is the same thing ``hpc3-stage`` checks, which
        makes an artifact produced here verifiable by the same command that
        verifies an input.

        Args:
            dir_path: Directory to pack.
            artifact_name: Name for the artifact, used as the filename.
            request_id: Correlation id, logged rather than sent anywhere.

        Returns:
            The stored record, whose ``file_id`` is the tarball's digest.

        Raises:
            FileNotFoundError: When ``dir_path`` does not exist or a file
                under it vanishes while it is packed. The staging directory
                is removed either way, so a failed save leaves nothing in the
                store.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        # A fresh staging directory per upload, not a name built from
        # artifact_name. The staging file is transient and nobody reads it by
        # name, so its only requirement is to be unique -- and deriving it from
        # the caller's name makes uniqueness the CALLER's property. Two runs
        # handed the same artifact_name would write the same staging file and
        # one would tar into the other's.
        #
        # That is not hypothetical here: the preflight round-trip called this
        # with a constant name, and the sibling collision it created killed an
        # arm of the Kazakh A/B 19 seconds in. Fixing the caller fixed that
        # instance; taking the name out of the caller's hands fixes the class.
        #
        # The directory's own context removes it however the block ends, so a
        # pack that fails (the directory is gone, a file vanishes mid-walk)
        # leaves no staging file behind and the failure still propagates.
        # Staging inside the root keeps the final rename on one filesystem,
        # and _by_digest reads only the root's own tarballs, never a staging
        # directory's.
        with tempfile.TemporaryDirectory(dir=self.root, prefix=".upload-") as staging_dir:
            staging = Path(staging_dir) / f"staged{_TARBALL_SUFFIX}"
            with tarfile.open(staging, "w:gz") as tar:
                tar.add(dir_path, arcname=dir_path.name)

            digest = hashlib.sha256(staging.read_bytes()).hexdigest()
            # Named for the artifact AND its digest. The name alone collides:
            # every run of one project picks the same one, so a second run
            # would overwrite the first and the first's recorded file_id would
            # name bytes no longer on disk. The digest alone is unreadable to
            # whoever has to find a run's output in a directory listing.
            target = self.root / f"{artifact_name}-{digest[:12]}{_TARBALL_SUFFIX}"
            staging.replace(target)
        size = target.stat().st_size
        _log.info(
            "artifact written",
            extra={"path": str(target), "sha256": digest, "request_id": request_id},
        )
        return StoredArtifact(file_id=digest, size=size)

    def download_artifact(
        self,
        file_id: str,
        *,
        dest_dir: Path,
        request_id: str,
        expected_root: str,
    ) -> Path:
        """Extract a tarball this store previously wrote.

        Args:
            file_id: Digest of the tarball, as returned by
                :meth:`upload_artifact`.
            dest_dir: Directory to extract into.
            request_id: Correlation id, logged rather than sent anywhere.
            expected_root: Directory name the tarball must contain.

        Returns:
            Path to the extracted root.

        Raises:
            AppError: With ``ARTIFACT_DOWNLOAD_FAILED`` when no tarball on
                disk carries that digest, or when the archive does not hold
                the expected root.
        """
        source = self._by_digest(file_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source, "r:gz") as tar:
            tar.extractall(dest_dir, filter="data")

        extracted = dest_dir / expected_root
        if not extracted.is_dir():
            raise AppError(
                ModelTrainerErrorCode.ARTIFACT_DOWNLOAD_FAILED,
                f"{source.name} did not contain {expected_root!r} (request {request_id}).",
            )
        return extracted

    def _by_digest(self, file_id: str) -> Path:
        """Find the tarball whose contents hash to a digest.

        Searched by content rather than by filename: the filename is the
        artifact NAME the caller chose, and two runs of one project choose
        the same one. The digest is what identifies the bytes.

        Args:
            file_id: Digest to look for.

        Returns:
            Path to the matching tarball.

        Raises:
            AppError: With ``ARTIFACT_DOWNLOAD_FAILED`` when nothing matches.
        """
        candidates = sorted(self.root.glob(f"*{_TARBALL_SUFFIX}")) if self.root.is_dir() else []
        for candidate in candidates:
            if hashlib.sha256(candidate.read_bytes()).hexdigest() == file_id:
                return candidate
        raise AppError(
            ModelTrainerErrorCode.ARTIFACT_DOWNLOAD_FAILED,
            f"No artifact under {self.root} hashes to {file_id!r}; "
            f"found {[c.name for c in candidates]}.",
        )


__all__ = ["LocalArtifacts", "StagedCorpus"]
