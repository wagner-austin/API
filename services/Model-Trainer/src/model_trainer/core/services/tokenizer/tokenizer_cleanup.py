from __future__ import annotations

import time
from pathlib import Path

from platform_core.json_utils import JSONValue
from platform_core.logging import get_logger

from ...config.settings import Settings
from ...infra.paths import models_dir, tokenizers_dir


class TokenizerCleanupError(Exception):
    """Raised when tokenizer cleanup fails."""


class TokenizerCleanupResult:
    deleted_tokenizers: int
    bytes_freed: int

    def __init__(self: TokenizerCleanupResult, deleted_tokenizers: int, bytes_freed: int) -> None:
        self.deleted_tokenizers = deleted_tokenizers
        self.bytes_freed = bytes_freed


class TokenizerCleanupService:
    """Service for cleaning up tokenizer artifacts that are no longer referenced.

    Semantics:
    - Never delete tokenizers that appear in any existing model manifest.
    - Only delete tokenizers that are unreferenced and older than the configured
      minimum unused age.
    - All IO or JSON parsing errors during manifest scanning or deletion raise
      TokenizerCleanupError; there is no best-effort cleanup.
    """

    settings: Settings

    def __init__(self: TokenizerCleanupService, settings: Settings) -> None:
        self.settings = settings

    def clean(self: TokenizerCleanupService) -> TokenizerCleanupResult:
        logger = get_logger(__name__)
        cfg = self.settings["app"]["tokenizer_cleanup"]
        if not cfg["enabled"]:
            logger.info(
                "Tokenizer cleanup skipped: disabled",
                extra={"event": "tokenizer_cleanup_skipped", "reason": "disabled"},
            )
            return TokenizerCleanupResult(deleted_tokenizers=0, bytes_freed=0)

        t_root = tokenizers_dir(self.settings)
        if not t_root.exists():
            logger.info(
                "Tokenizer cleanup: tokenizers directory missing",
                extra={
                    "event": "tokenizer_cleanup_completed",
                    "deleted_tokenizers": 0,
                    "bytes_freed": 0,
                    "reason": "directory_missing",
                },
            )
            return TokenizerCleanupResult(deleted_tokenizers=0, bytes_freed=0)

        if not t_root.is_dir():
            raise TokenizerCleanupError(f"tokenizers path is not a directory: {t_root}")

        in_use = self._collect_tokenizers_in_use()
        now = time.time()
        min_age_seconds = float(cfg["min_unused_days"]) * 24.0 * 60.0 * 60.0

        logger.info(
            "Tokenizer cleanup started",
            extra={
                "event": "tokenizer_cleanup_started",
                "tokenizers_root": str(t_root),
                "min_unused_days": cfg["min_unused_days"],
                "in_use_count": len(in_use),
            },
        )

        deleted = 0
        freed = 0

        from model_trainer.core import _test_hooks

        try:
            for entry in _test_hooks.path_iterdir(t_root):
                if not entry.is_dir():
                    continue
                tokenizer_id = entry.name
                if tokenizer_id in in_use:
                    continue
                stat = entry.stat()
                age_seconds = now - float(stat.st_mtime)
                if age_seconds < min_age_seconds:
                    continue
                size = _directory_size(entry)
                try:
                    _test_hooks.shutil_rmtree(entry)
                except OSError as exc:
                    logger.error(
                        "Failed to delete tokenizer directory",
                        extra={
                            "event": "tokenizer_cleanup_failed",
                            "tokenizer_id": tokenizer_id,
                            "path": str(entry),
                            "error": str(exc),
                        },
                    )
                    raise TokenizerCleanupError(
                        f"failed to delete tokenizer {tokenizer_id}: {exc}"
                    ) from exc
                deleted += 1
                freed += size
        except OSError as exc:
            raise TokenizerCleanupError(f"failed to scan tokenizers directory: {exc}") from exc

        logger.info(
            "Tokenizer cleanup completed",
            extra={
                "event": "tokenizer_cleanup_completed",
                "deleted_tokenizers": deleted,
                "bytes_freed": freed,
            },
        )
        return TokenizerCleanupResult(deleted_tokenizers=deleted, bytes_freed=freed)

    def _collect_tokenizers_in_use(self: TokenizerCleanupService) -> set[str]:
        root = models_dir(self.settings)
        in_use: set[str] = set()
        if not root.exists() or not root.is_dir():
            return in_use
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                from platform_core.json_utils import load_json_str

                text = manifest_path.read_text(encoding="utf-8")
                obj = load_json_str(text)
                if not isinstance(obj, dict):
                    raise ValueError("invalid manifest format")
                tok_id_val: JSONValue = obj.get("tokenizer_id")
                if not isinstance(tok_id_val, str):
                    raise ValueError("invalid tokenizer_id in manifest")
                tid = tok_id_val.strip()
            except Exception as exc:
                get_logger(__name__).error(
                    "Failed to read tokenizer manifest",
                    extra={
                        "event": "tokenizer_cleanup_manifest_error",
                        "path": str(manifest_path),
                        "error": str(exc),
                    },
                )
                raise TokenizerCleanupError(
                    f"failed to read manifest {manifest_path}: {exc}"
                ) from exc

            if tid != "":
                in_use.add(tid)
        return in_use


def _directory_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += int(p.stat().st_size)
    return total
