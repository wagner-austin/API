from __future__ import annotations

from pathlib import Path

from platform_core.job_types import JobStatusLiteral
from platform_core.logging import get_logger
from platform_core.trainer_keys import artifact_file_id_key
from platform_workers.redis import RedisStrProto

from ....worker.trainer_job_store import TrainerJobStatus, TrainerJobStore
from ...config.settings import Settings


class CleanupError(Exception):
    """Raised when artifact cleanup fails."""


class CleanupResult:
    """Result of cleanup operation."""

    run_id: str
    deleted: bool
    bytes_freed: int
    files_deleted: int
    reason: str | None

    def __init__(
        self: CleanupResult,
        run_id: str,
        deleted: bool,
        bytes_freed: int,
        files_deleted: int,
        reason: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.deleted = deleted
        self.bytes_freed = bytes_freed
        self.files_deleted = files_deleted
        self.reason = reason


class ArtifactCleanupService:
    """Service for cleaning up local artifacts after successful upload.

    Safety checks before deletion:
    1. Verify cleanup is enabled (feature flag).
    2. Verify the artifact directory exists.
    3. Verify upload succeeded (file_id exists in Redis).
    4. Verify run is terminal (completed/failed/canceled) based on Redis status.
    5. Apply grace period if configured.
    6. Dry-run mode respects all checks but does not delete.

    All exceptions are logged and re-raised per guard rules; no silent failure.
    """

    settings: Settings
    redis_client: RedisStrProto

    def __init__(
        self: ArtifactCleanupService, settings: Settings, redis_client: RedisStrProto
    ) -> None:
        self.settings = settings
        self.redis_client = redis_client

    def cleanup_run_artifacts(
        self: ArtifactCleanupService, run_id: str, artifact_dir: str | Path
    ) -> CleanupResult:
        """Clean up local artifacts for a completed training run."""
        logger = get_logger(__name__)
        artifact_path = Path(artifact_dir)

        if not self.settings["app"]["cleanup"]["enabled"]:
            return CleanupResult(
                run_id=run_id,
                deleted=False,
                bytes_freed=0,
                files_deleted=0,
                reason="cleanup_disabled",
            )

        if not artifact_path.exists():
            return CleanupResult(
                run_id=run_id,
                deleted=False,
                bytes_freed=0,
                files_deleted=0,
                reason="directory_not_found",
            )

        if self.settings["app"]["cleanup"]["verify_upload"]:
            file_id = self.redis_client.get(artifact_file_id_key(run_id))
            if not isinstance(file_id, str) or file_id.strip() == "":
                logger.warning(
                    "Cleanup skipped: upload not verified",
                    extra={"event": "cleanup_skipped", "run_id": run_id, "reason": "no_file_id"},
                )
                return CleanupResult(
                    run_id=run_id,
                    deleted=False,
                    bytes_freed=0,
                    files_deleted=0,
                    reason="upload_not_verified",
                )

        status_store: TrainerJobStatus | None = TrainerJobStore(self.redis_client).load(run_id)
        status: JobStatusLiteral | None = None if status_store is None else status_store["status"]
        if status not in ("completed", "failed"):
            logger.info(
                "Cleanup skipped: run not terminal",
                extra={
                    "event": "cleanup_skipped",
                    "run_id": run_id,
                    "reason": "run_not_terminal",
                    "status": status,
                },
            )
            return CleanupResult(
                run_id=run_id,
                deleted=False,
                bytes_freed=0,
                files_deleted=0,
                reason="run_not_terminal",
            )

        from model_trainer.core import _test_hooks

        grace = self.settings["app"]["cleanup"]["grace_period_seconds"]
        if grace > 0:
            _test_hooks.time_sleep(grace)

        bytes_freed = self._calculate_directory_size(artifact_path)
        files_deleted = self._count_files(artifact_path)

        if self.settings["app"]["cleanup"]["dry_run"]:
            logger.info(
                "Cleanup dry-run: would delete",
                extra={
                    "event": "cleanup_dry_run",
                    "run_id": run_id,
                    "path": str(artifact_path),
                    "bytes": bytes_freed,
                    "files": files_deleted,
                },
            )
            return CleanupResult(
                run_id=run_id,
                deleted=False,
                bytes_freed=0,
                files_deleted=0,
                reason="dry_run",
            )

        try:
            _test_hooks.shutil_rmtree(str(artifact_path))
        except OSError as exc:
            logger.error(
                "Cleanup failed",
                extra={
                    "event": "cleanup_failed",
                    "run_id": run_id,
                    "path": str(artifact_path),
                    "error": str(exc),
                },
            )
            raise CleanupError(f"Failed to delete {artifact_path}: {exc}") from exc

        logger.info(
            "Cleanup completed",
            extra={
                "event": "cleanup_completed",
                "run_id": run_id,
                "path": str(artifact_path),
                "bytes_freed": bytes_freed,
                "files_deleted": files_deleted,
            },
        )

        return CleanupResult(
            run_id=run_id,
            deleted=True,
            bytes_freed=bytes_freed,
            files_deleted=files_deleted,
            reason=None,
        )

    def _calculate_directory_size(self: ArtifactCleanupService, path: Path) -> int:
        """Calculate total size of all files in directory recursively."""
        total = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    def _count_files(self: ArtifactCleanupService, path: Path) -> int:
        """Count total number of files in directory recursively."""
        count = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                count += 1
        return count
