"""Tests for scripts/optimize/logging_config.py - logging configuration."""

from __future__ import annotations

from platform_core.logging import stdlib_logging
from scripts.optimize.logging_config import (
    set_verbose_mode,
    suppress_verbose_logging,
)


class TestSetVerboseMode:
    """Tests for set_verbose_mode function."""

    def test_sets_verbose_mode_true(self) -> None:
        """Test setting verbose mode to True."""
        # Reset to False first
        set_verbose_mode(False)

        set_verbose_mode(True)

        # Verify by checking suppress_verbose_logging behavior
        # When verbose, loggers should NOT be suppressed
        optuna_logger = stdlib_logging.getLogger("optuna")
        original_level = optuna_logger.level

        suppress_verbose_logging()

        # In verbose mode, level should remain unchanged
        assert optuna_logger.level == original_level

        # Cleanup
        set_verbose_mode(False)

    def test_sets_verbose_mode_false(self) -> None:
        """Test setting verbose mode to False."""
        set_verbose_mode(False)

        # Get optuna logger and ensure it has a level that would change
        optuna_logger = stdlib_logging.getLogger("optuna")
        optuna_logger.setLevel(stdlib_logging.INFO)

        suppress_verbose_logging()

        # In non-verbose mode, optuna should be suppressed to WARNING
        assert optuna_logger.level == stdlib_logging.WARNING


class TestSuppressVerboseLogging:
    """Tests for suppress_verbose_logging function."""

    def test_suppresses_optuna_loggers_when_not_verbose(self) -> None:
        """Test that optuna loggers are suppressed when verbose mode is off."""
        set_verbose_mode(False)

        # Set optuna loggers to INFO first
        loggers_to_check = [
            "optuna",
            "optuna.trial",
            "optuna.study",
            "optuna._optimize",
            "covenant_radar_api.worker.optimize_job",
            "covenant_ml.optimizer.optuna_backend",
            "covenant_radar_api.seeding.real_data",
        ]
        for name in loggers_to_check:
            stdlib_logging.getLogger(name).setLevel(stdlib_logging.INFO)

        suppress_verbose_logging()

        # All should be set to WARNING now
        for name in loggers_to_check:
            logger = stdlib_logging.getLogger(name)
            assert logger.level == stdlib_logging.WARNING

    def test_does_not_suppress_when_verbose(self) -> None:
        """Test that loggers are NOT suppressed when verbose mode is on."""
        set_verbose_mode(True)

        # Set optuna logger to DEBUG
        optuna_logger = stdlib_logging.getLogger("optuna")
        optuna_logger.setLevel(stdlib_logging.DEBUG)

        suppress_verbose_logging()

        # Should still be DEBUG (not changed to WARNING)
        assert optuna_logger.level == stdlib_logging.DEBUG

        # Cleanup
        set_verbose_mode(False)
