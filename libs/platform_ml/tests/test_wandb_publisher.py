"""Tests for wandb_publisher module.

These tests use Protocol-typed fake classes and monkeypatching to avoid
MagicMock and Any types while maintaining strict mypy compliance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol

import pytest
from platform_core.json_utils import JSONValue

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

    def test_raises_when_wandb_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should raise WandbUnavailableError when wandb is not installed."""
        import importlib.util

        def _find_spec_none(name: str, package: str | None = None) -> None:
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_none)

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


class _FakeRunProto(Protocol):
    """Protocol for fake wandb run."""

    @property
    def id(self) -> str: ...


class _FakeRun:
    """Fake wandb.Run for testing."""

    def __init__(self, run_id: str) -> None:
        self._id = run_id

    @property
    def id(self) -> str:
        return self._id


class _FakeConfig:
    """Fake wandb.config for testing."""

    def __init__(self) -> None:
        self.updates: list[dict[str, JSONValue]] = []

    def update(self, d: Mapping[str, JSONValue]) -> None:
        self.updates.append(dict(d))


class _FakeTable:
    """Fake wandb.Table for testing."""

    def __init__(self, columns: list[str], data: list[list[float | int | str | bool]]) -> None:
        self.columns = columns
        self.data = data


class _FakeWandbModule:
    """Fake wandb module for testing."""

    def __init__(self, *, run_active: bool = True) -> None:
        self._run: _FakeRun | None = None
        self._run_active = run_active
        self._config = _FakeConfig()
        self._logs: list[dict[str, float | int | str | bool | _FakeTable]] = []
        self._tables: list[_FakeTable] = []
        self._finished = False
        self._init_calls: list[tuple[str, str]] = []
        self._table_class = _FakeTable

    @property
    def run(self) -> _FakeRun | None:
        if not self._run_active:
            return None
        return self._run

    @property
    def config(self) -> _FakeConfig:
        return self._config

    def init(self, *, project: str, name: str) -> _FakeRun:
        self._init_calls.append((project, name))
        self._run = _FakeRun(run_id="fake-run-123")
        return self._run

    def log(self, data: Mapping[str, float | int | str | bool | _FakeTable]) -> None:
        self._logs.append(dict(data))
        # Store tables separately for typed access
        for value in data.values():
            if hasattr(value, "columns") and hasattr(value, "data"):
                table: _FakeTable = value
                self._tables.append(table)

    def finish(self) -> None:
        self._finished = True

    @property
    def Table(self) -> type[_FakeTable]:
        return self._table_class

    def get_last_table(self) -> _FakeTable:
        """Get the last logged table, properly typed.

        Raises:
            AssertionError: If no tables have been logged.
        """
        assert self._tables, "No tables logged"
        return self._tables[-1]


def _create_fake_wandb(*, run_active: bool = True) -> _FakeWandbModule:
    """Create a fake wandb module for testing."""
    return _FakeWandbModule(run_active=run_active)


class _FakeModuleSpec:
    """Fake module spec for importlib.util.find_spec."""

    def __init__(self, name: str) -> None:
        self.name = name


class TestWandbPublisherEnabled:
    """Tests for WandbPublisher when enabled with fake wandb."""

    @pytest.fixture
    def fake_wandb(self) -> _FakeWandbModule:
        """Create a fake wandb module."""
        return _create_fake_wandb(run_active=True)

    def test_enabled_publisher_init(
        self, fake_wandb: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enabled publisher should initialize wandb run."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        publisher = WandbPublisher(project="test-proj", run_name="run-1", enabled=True)

        assert len(fake_wandb._init_calls) == 1
        assert fake_wandb._init_calls[0] == ("test-proj", "run-1")
        assert publisher.is_enabled is True

    def test_enabled_publisher_get_init_result(
        self, fake_wandb: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enabled publisher should return enabled status with run_id."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        result = publisher.get_init_result()

        assert result["status"] == "enabled"
        assert result["run_id"] == "fake-run-123"

    def test_enabled_publisher_log_config(
        self, fake_wandb: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_config should update wandb config."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.log_config({"model_family": "gpt2", "batch_size": 32})

        assert len(fake_wandb._config.updates) == 1
        assert fake_wandb._config.updates[0] == {"model_family": "gpt2", "batch_size": 32}

    def test_enabled_publisher_log_step(
        self, fake_wandb: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_step should log metrics to wandb."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.log_step({"train_loss": 0.5, "train_ppl": 1.65})

        assert len(fake_wandb._logs) == 1
        assert fake_wandb._logs[0] == {"train_loss": 0.5, "train_ppl": 1.65}

    def test_enabled_publisher_log_epoch(
        self, fake_wandb: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_epoch should log epoch metrics to wandb."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.log_epoch({"val_loss": 0.3, "val_ppl": 1.35})

        assert len(fake_wandb._logs) == 1
        assert fake_wandb._logs[0] == {"val_loss": 0.3, "val_ppl": 1.35}

    def test_enabled_publisher_log_final_true(
        self, fake_wandb: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_final should log final metrics with bool=True converted to 1."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.log_final({"test_loss": 0.25, "early_stopped": True})

        assert len(fake_wandb._logs) == 1
        assert fake_wandb._logs[0] == {"test_loss": 0.25, "early_stopped": 1}

    def test_enabled_publisher_log_final_false(
        self, fake_wandb: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_final should convert False to 0."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.log_final({"test_loss": 0.25, "early_stopped": False})

        assert len(fake_wandb._logs) == 1
        assert fake_wandb._logs[0] == {"test_loss": 0.25, "early_stopped": 0}

    def test_enabled_publisher_log_table(
        self, fake_wandb: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_table should create and log wandb Table."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.log_table(
            "epoch_summary",
            ["epoch", "loss"],
            [[1, 0.5], [2, 0.3]],
        )

        assert len(fake_wandb._logs) == 1
        assert "epoch_summary" in fake_wandb._logs[0]
        logged_table = fake_wandb.get_last_table()
        assert logged_table.columns == ["epoch", "loss"]
        assert logged_table.data == [[1, 0.5], [2, 0.3]]

    def test_enabled_publisher_finish(
        self, fake_wandb: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """finish should call wandb.finish()."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.finish()

        assert fake_wandb._finished is True


class TestWandbPublisherNoActiveRun:
    """Tests for WandbPublisher when run is None (edge case)."""

    @pytest.fixture
    def fake_wandb_no_run(self) -> _FakeWandbModule:
        """Create a fake wandb module with no active run."""
        return _create_fake_wandb(run_active=False)

    def test_log_config_noop_when_run_none(
        self, fake_wandb_no_run: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_config should be no-op when wandb.run is None."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb_no_run)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.log_config({"key": "value"})

        assert len(fake_wandb_no_run._config.updates) == 0

    def test_log_step_noop_when_run_none(
        self, fake_wandb_no_run: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_step should be no-op when wandb.run is None."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb_no_run)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.log_step({"train_loss": 0.5})

        assert len(fake_wandb_no_run._logs) == 0

    def test_log_epoch_noop_when_run_none(
        self, fake_wandb_no_run: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_epoch should be no-op when wandb.run is None."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb_no_run)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.log_epoch({"val_loss": 0.3})

        assert len(fake_wandb_no_run._logs) == 0

    def test_log_final_noop_when_run_none(
        self, fake_wandb_no_run: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_final should be no-op when wandb.run is None."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb_no_run)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.log_final({"test_loss": 0.25})

        assert len(fake_wandb_no_run._logs) == 0

    def test_log_table_noop_when_run_none(
        self, fake_wandb_no_run: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_table should be no-op when wandb.run is None."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb_no_run)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.log_table("test", ["col"], [[1]])

        assert len(fake_wandb_no_run._logs) == 0

    def test_finish_noop_when_run_none(
        self, fake_wandb_no_run: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """finish should be no-op when wandb.run is None."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb_no_run)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        publisher.finish()

        assert fake_wandb_no_run._finished is False

    def test_is_enabled_false_when_run_none(
        self, fake_wandb_no_run: _FakeWandbModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_enabled should be False when wandb.run is None."""
        import importlib.util
        import sys

        def _find_spec_ok(name: str, package: str | None = None) -> _FakeModuleSpec | None:
            if name == "wandb":
                return _FakeModuleSpec("wandb")
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_ok)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb_no_run)

        publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
        assert publisher.is_enabled is False


class TestWandbPublisherUnavailable:
    """Tests for WandbPublisher when wandb is not installed."""

    def test_raises_when_enabled_but_wandb_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should raise WandbUnavailableError when enabled but wandb not installed."""
        import importlib.util

        def _find_spec_none(name: str, package: str | None = None) -> None:
            return None

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_none)

        with pytest.raises(WandbUnavailableError, match="wandb package is not installed"):
            WandbPublisher(project="test", run_name="run-1", enabled=True)


_EXPECTED_PROJECT: Final[str] = "test-proj"
_EXPECTED_RUN_NAME: Final[str] = "run-1"
