"""The three services a cluster run supplies for itself.

A compute node has no Redis, no data-bank API and no artifact service. It does
have a filesystem and a corpus someone put there on purpose. Each of those
three dependencies is already a Protocol behind a hook, so a cluster run
implements them rather than doing without the trainer.

The corpus is the interesting one. In the service deployment a file id is
fetched over HTTP and cached; here the same id names a file that ``hpc3-stage``
already placed and verified by SHA-256 on both sides of the transfer. That is
a stronger guarantee than the fetch gives, not a weaker one -- but only if the
id really is the digest, so a mismatch is refused rather than downloaded
around, because there is nothing here to download from.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
from pathlib import Path

from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.logging import get_logger

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
            "A compute node cannot fetch one; stage it with hpc3-stage first.",
        )


class LocalArtifacts:
    """Writes run artifacts to the filesystem instead of an upload service.

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
    ) -> FileUploadResponse:
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
            A response shaped like the upload service's, whose ``file_id`` is
            the tarball's digest.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        # mkstemp, not a name built from artifact_name. The staging file is
        # transient and nobody reads it by name, so its only requirement is to
        # be unique -- and deriving it from the caller's name makes uniqueness
        # the CALLER's property. Two runs handed the same artifact_name would
        # write the same .partial and one would tar into the other's file.
        #
        # That is not hypothetical here: the preflight round-trip called this
        # with a constant name, and the sibling collision it created killed an
        # arm of the Kazakh A/B 19 seconds in. Fixing the caller fixed that
        # instance; taking the name out of the caller's hands fixes the class.
        handle, staged_path = tempfile.mkstemp(
            dir=self.root, prefix=".upload-", suffix=f"{_TARBALL_SUFFIX}.partial"
        )
        os.close(handle)
        staging = Path(staged_path)
        with tarfile.open(staging, "w:gz") as tar:
            tar.add(dir_path, arcname=dir_path.name)

        digest = hashlib.sha256(staging.read_bytes()).hexdigest()
        # Named for the artifact AND its digest. The name alone collides:
        # every run of one project picks the same one, so a second run would
        # overwrite the first and the first's recorded file_id would name
        # bytes no longer on disk. The digest alone is unreadable to whoever
        # has to find a run's output in a directory listing.
        target = self.root / f"{artifact_name}-{digest[:12]}{_TARBALL_SUFFIX}"
        staging.replace(target)
        size = target.stat().st_size
        _log.info(
            "artifact written",
            extra={"path": str(target), "sha256": digest, "request_id": request_id},
        )
        return FileUploadResponse(
            file_id=digest,
            size=size,
            sha256=digest,
            content_type="application/gzip",
            created_at=None,
        )

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
