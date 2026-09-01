"""The scanner-riding rename: bounded, observable, and loud at the end.

Exists because of a measured failure, not a hypothetical: on 2026-08-31
``test_eval_job_missing_manifest`` died at the materializing rename with
``PermissionError`` (WinError 5) once under the full 16-worker suite and
passed five of five runs in isolation -- Windows real-time scanning holds a
handle on freshly written files, and a directory holding an open file cannot
be renamed. The retry must clear that case and must NOT absorb a real
permission problem, and both directions are driven here through the hooks,
because the only thing that produces the real denial is a virus scanner's
timing.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from model_trainer.core import _test_hooks
from model_trainer.core._hook_defaults import _default_rename_path, _default_retry_sleep
from model_trainer.worker.job_utils import (
    RENAME_ATTEMPTS,
    RENAME_RETRY_SECONDS,
    rename_with_scan_retry,
)


def _populated_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Build a source directory with content and return it with a target."""
    source = tmp_path / "model-run-1"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"x")
    return source, tmp_path / "run-1"


class TestTheDefaults:
    def test_the_default_rename_moves_the_directory_and_its_content(self, tmp_path: Path) -> None:
        source, target = _populated_dir(tmp_path)

        _default_rename_path(source, target)

        assert (target / "weights.bin").read_bytes() == b"x"
        assert not source.exists()

    def test_the_default_sleep_actually_waits(self) -> None:
        started = time.monotonic()

        _default_retry_sleep(0.05)

        # Slightly under the requested wait, because Windows timers round.
        assert time.monotonic() - started >= 0.04

    def test_the_production_hooks_are_the_defaults(self) -> None:
        assert _test_hooks.rename_path is _default_rename_path
        assert _test_hooks.retry_sleep is _default_retry_sleep


class TestTheRetry:
    def test_a_clean_rename_neither_retries_nor_sleeps(self, tmp_path: Path) -> None:
        source, target = _populated_dir(tmp_path)
        slept: list[float] = []

        def record_sleep(seconds: float) -> None:
            slept.append(seconds)

        _test_hooks.retry_sleep = record_sleep
        try:
            rename_with_scan_retry(source, target)
        finally:
            _test_hooks.retry_sleep = _default_retry_sleep

        assert (target / "weights.bin").read_bytes() == b"x"
        assert slept == []

    def test_a_denial_that_clears_is_waited_out(self, tmp_path: Path) -> None:
        source, target = _populated_dir(tmp_path)
        attempts: list[int] = []
        slept: list[float] = []

        def denies_twice(source: Path, target: Path) -> None:
            attempts.append(len(attempts))
            if len(attempts) <= 2:
                raise PermissionError(13, "Access is denied", str(source))
            _default_rename_path(source, target)

        def record_sleep(seconds: float) -> None:
            slept.append(seconds)

        _test_hooks.rename_path = denies_twice
        _test_hooks.retry_sleep = record_sleep
        try:
            rename_with_scan_retry(source, target)
        finally:
            _test_hooks.rename_path = _default_rename_path
            _test_hooks.retry_sleep = _default_retry_sleep

        assert (target / "weights.bin").read_bytes() == b"x"
        assert len(attempts) == 3
        assert slept == [RENAME_RETRY_SECONDS, RENAME_RETRY_SECONDS]

    def test_a_denial_that_persists_escapes_with_the_real_error(self, tmp_path: Path) -> None:
        source, target = _populated_dir(tmp_path)
        attempts: list[int] = []
        slept: list[float] = []

        def always_denies(source: Path, target: Path) -> None:
            attempts.append(len(attempts))
            raise PermissionError(13, "Access is denied", str(source))

        def record_sleep(seconds: float) -> None:
            slept.append(seconds)

        _test_hooks.rename_path = always_denies
        _test_hooks.retry_sleep = record_sleep
        try:
            with pytest.raises(PermissionError, match="Access is denied"):
                rename_with_scan_retry(source, target)
        finally:
            _test_hooks.rename_path = _default_rename_path
            _test_hooks.retry_sleep = _default_retry_sleep

        # The last attempt runs OUTSIDE the catch: every attempt was made,
        # only the inter-attempt waits were spent, and the source is intact.
        assert len(attempts) == RENAME_ATTEMPTS
        assert len(slept) == RENAME_ATTEMPTS - 1
        assert (source / "weights.bin").read_bytes() == b"x"
        assert not target.exists()

    def test_any_other_failure_is_not_retried(self, tmp_path: Path) -> None:
        # Only the scanner's specific denial is worth waiting out. A missing
        # source is a caller bug, and retrying it would spend a second
        # hiding the stack trace's cause.
        source, target = _populated_dir(tmp_path)
        slept: list[float] = []

        def missing(source: Path, target: Path) -> None:
            raise FileNotFoundError(2, "No such file or directory", str(source))

        def record_sleep(seconds: float) -> None:
            slept.append(seconds)

        _test_hooks.rename_path = missing
        _test_hooks.retry_sleep = record_sleep
        try:
            with pytest.raises(FileNotFoundError, match="No such file"):
                rename_with_scan_retry(source, target)
        finally:
            _test_hooks.rename_path = _default_rename_path
            _test_hooks.retry_sleep = _default_retry_sleep

        assert slept == []

    def test_the_total_patience_stays_under_a_second(self) -> None:
        # The docstring's bargain: a scanner pass is covered, a real
        # permission problem is not silently absorbed for long.
        assert (RENAME_ATTEMPTS - 1) * RENAME_RETRY_SECONDS <= 1.0
