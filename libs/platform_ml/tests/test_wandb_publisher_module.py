"""Tests for wandb publisher: WandbUnavailableError."""

from __future__ import annotations

import pytest

from platform_ml.testing import (
    WandbModuleProtocol,
)
from platform_ml.testing import (
    hooks as wandb_hooks,
)
from platform_ml.wandb_publisher import (
    WandbPublisher,
    WandbUnavailableError,
    _load_wandb_module,
)


class TestWandbUnavailableError:
    """Tests for WandbUnavailableError exception."""

    def test_error_message(self) -> None:
        """Error should contain descriptive message."""
        err = WandbUnavailableError("test message")
        assert str(err) == "test message"


class TestLoadWandbModule:
    """Tests for _load_wandb_module function."""

    def test_raises_when_wandb_not_installed(self) -> None:
        """Should raise WandbUnavailableError when wandb is not installed."""

        def _load_wandb_unavailable() -> WandbModuleProtocol:
            raise WandbUnavailableError("wandb package is not installed")

        wandb_hooks.load_wandb_module = _load_wandb_unavailable

        with pytest.raises(WandbUnavailableError, match="wandb package is not installed"):
            _load_wandb_module()


class TestWandbPublisherDisabled:
    """Tests for WandbPublisher when disabled."""

    def test_disabled_publisher_init(self) -> None:
        """Disabled publisher should initialize without loading wandb."""
        publisher = WandbPublisher(project="test", run_name="run-1", enabled=False)
        assert publisher.is_enabled is False

    def test_disabled_publisher_get_init_result(self) -> None:
        """Disabled publisher should return disabled status."""
        publisher = WandbPublisher(project="test", run_name="run-1", enabled=False)
        result = publisher.get_init_result()
        assert result["status"] == "disabled"
        assert result["run_id"] is None

    def test_disabled_publisher_log_config_noop(self) -> None:
        """log_config should be a no-op when disabled."""
        publisher = WandbPublisher(project="test", run_name="run-1", enabled=False)
        publisher.log_config({"key": "value"})

    def test_disabled_publisher_log_step_noop(self) -> None:
        """log_step should be a no-op when disabled."""
        publisher = WandbPublisher(project="test", run_name="run-1", enabled=False)
        publisher.log_step({"train_loss": 0.5})

    def test_disabled_publisher_log_epoch_noop(self) -> None:
        """log_epoch should be a no-op when disabled."""
        publisher = WandbPublisher(project="test", run_name="run-1", enabled=False)
        publisher.log_epoch({"val_loss": 0.3})

    def test_disabled_publisher_log_final_noop(self) -> None:
        """log_final should be a no-op when disabled."""
        publisher = WandbPublisher(project="test", run_name="run-1", enabled=False)
        publisher.log_final({"test_loss": 0.25, "early_stopped": False})

    def test_disabled_publisher_log_table_noop(self) -> None:
        """log_table should be a no-op when disabled."""
        publisher = WandbPublisher(project="test", run_name="run-1", enabled=False)
        publisher.log_table("epoch_summary", ["col1"], [[1.0]])

    def test_disabled_publisher_finish_noop(self) -> None:
        """finish should be a no-op when disabled."""
        publisher = WandbPublisher(project="test", run_name="run-1", enabled=False)
        publisher.finish()
