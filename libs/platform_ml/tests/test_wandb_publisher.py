"""Tests for wandb_publisher module using test hooks.

These tests use Protocol-typed fake classes and the testing hooks to avoid
MagicMock and Any types while maintaining strict mypy compliance.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from typing import Final

import pytest
from platform_core.json_utils import JSONValue
from typing_extensions import TypeIs

from platform_ml.testing import (
    WandbModuleProtocol,
    WandbTableCtorProtocol,
    WandbTableProtocol,
    hooks,
    reset_hooks,
)
from platform_ml.wandb_publisher import (
    WandbPublisher,
    WandbUnavailableError,
    _load_wandb_module,
)


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test to prevent state leakage."""
    yield
    reset_hooks()


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

        hooks.load_wandb_module = _load_wandb_unavailable

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


class FakeRun:
    """Fake wandb.Run for testing."""

    def __init__(self, run_id: str) -> None:
        self._id = run_id

    @property
    def id(self) -> str:
        return self._id


class FakeConfig:
    """Fake wandb.config for testing."""

    def __init__(self) -> None:
        self.updates: list[dict[str, JSONValue]] = []

    def update(self, d: Mapping[str, JSONValue]) -> None:
        self.updates.append(dict(d))


class FakeTable:
    """Fake wandb.Table for testing."""

    def __init__(self, columns: list[str], data: list[list[float | int | str | bool]]) -> None:
        self.columns = columns
        self.data = data


def _is_table(val: float | int | str | bool | WandbTableProtocol) -> TypeIs[WandbTableProtocol]:
    """Type narrowing function to identify table values."""
    return hasattr(val, "columns") and hasattr(val, "data")


class FakeWandbModule:
    """Fake wandb module for testing."""

    def __init__(self, *, run_active: bool = True) -> None:
        self._run: FakeRun | None = None
        self._run_active = run_active
        self._config = FakeConfig()
        self._logged_tables: list[WandbTableProtocol] = []
        self._logs: list[dict[str, float | int | str | bool]] = []
        self._finished = False
        self._init_calls: list[tuple[str, str]] = []

    @property
    def run(self) -> FakeRun | None:
        if not self._run_active:
            return None
        return self._run

    @property
    def config(self) -> FakeConfig:
        return self._config

    @property
    def table_ctor(self) -> WandbTableCtorProtocol:
        return FakeTable

    def init(self, *, project: str, name: str) -> FakeRun:
        self._init_calls.append((project, name))
        self._run = FakeRun(run_id="fake-run-123")
        return self._run

    def log(self, data: Mapping[str, float | int | str | bool | WandbTableProtocol]) -> None:
        scalar_data: dict[str, float | int | str | bool] = {}
        for key, val in data.items():
            if _is_table(val):
                # val is WandbTableProtocol after TypeIs narrowing
                self._logged_tables.append(val)
                scalar_data[key] = f"<table:{len(self._logged_tables) - 1}>"
            else:
                scalar_data[key] = val
        self._logs.append(scalar_data)

    def finish(self) -> None:
        self._finished = True

    def get_last_table(self) -> WandbTableProtocol:
        """Get the last logged table."""
        assert self._logged_tables, "No tables logged"
        return self._logged_tables[-1]


def _setup_fake_wandb(fake_wandb: FakeWandbModule) -> None:
    """Set up fake wandb module via hooks."""

    def _load_fake_wandb() -> WandbModuleProtocol:
        return fake_wandb

    hooks.load_wandb_module = _load_fake_wandb


def test_enabled_publisher_init() -> None:
    """Enabled publisher should initialize wandb run."""
    fake_wandb = FakeWandbModule(run_active=True)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test-proj", run_name="run-1", enabled=True)

    assert len(fake_wandb._init_calls) == 1
    assert fake_wandb._init_calls[0] == ("test-proj", "run-1")
    assert publisher.is_enabled is True


def test_enabled_publisher_get_init_result() -> None:
    """Enabled publisher should return enabled status with run_id."""
    fake_wandb = FakeWandbModule(run_active=True)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    result = publisher.get_init_result()

    assert result["status"] == "enabled"
    assert result["run_id"] == "fake-run-123"


def test_enabled_publisher_log_config() -> None:
    """log_config should update wandb config."""
    fake_wandb = FakeWandbModule(run_active=True)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.log_config({"model_family": "gpt2", "batch_size": 32})

    assert len(fake_wandb._config.updates) == 1
    assert fake_wandb._config.updates[0] == {"model_family": "gpt2", "batch_size": 32}


def test_enabled_publisher_log_step() -> None:
    """log_step should log metrics to wandb."""
    fake_wandb = FakeWandbModule(run_active=True)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.log_step({"train_loss": 0.5, "train_ppl": 1.65})

    assert len(fake_wandb._logs) == 1
    assert fake_wandb._logs[0] == {"train_loss": 0.5, "train_ppl": 1.65}


def test_enabled_publisher_log_epoch() -> None:
    """log_epoch should log epoch metrics to wandb."""
    fake_wandb = FakeWandbModule(run_active=True)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.log_epoch({"val_loss": 0.3, "val_ppl": 1.35})

    assert len(fake_wandb._logs) == 1
    assert fake_wandb._logs[0] == {"val_loss": 0.3, "val_ppl": 1.35}


def test_enabled_publisher_log_final_true() -> None:
    """log_final should log final metrics with bool=True converted to 1."""
    fake_wandb = FakeWandbModule(run_active=True)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.log_final({"test_loss": 0.25, "early_stopped": True})

    assert len(fake_wandb._logs) == 1
    assert fake_wandb._logs[0] == {"test_loss": 0.25, "early_stopped": 1}


def test_enabled_publisher_log_final_false() -> None:
    """log_final should convert False to 0."""
    fake_wandb = FakeWandbModule(run_active=True)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.log_final({"test_loss": 0.25, "early_stopped": False})

    assert len(fake_wandb._logs) == 1
    assert fake_wandb._logs[0] == {"test_loss": 0.25, "early_stopped": 0}


def test_enabled_publisher_log_table() -> None:
    """log_table should create and log wandb Table."""
    fake_wandb = FakeWandbModule(run_active=True)
    _setup_fake_wandb(fake_wandb)

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


def test_enabled_publisher_finish() -> None:
    """finish should call wandb.finish()."""
    fake_wandb = FakeWandbModule(run_active=True)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.finish()

    assert fake_wandb._finished is True


# Tests for when wandb.run is None (edge case)


def test_log_config_noop_when_run_none() -> None:
    """log_config should be no-op when wandb.run is None."""
    fake_wandb = FakeWandbModule(run_active=False)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.log_config({"key": "value"})

    assert len(fake_wandb._config.updates) == 0


def test_log_step_noop_when_run_none() -> None:
    """log_step should be no-op when wandb.run is None."""
    fake_wandb = FakeWandbModule(run_active=False)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.log_step({"train_loss": 0.5})

    assert len(fake_wandb._logs) == 0


def test_log_epoch_noop_when_run_none() -> None:
    """log_epoch should be no-op when wandb.run is None."""
    fake_wandb = FakeWandbModule(run_active=False)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.log_epoch({"val_loss": 0.3})

    assert len(fake_wandb._logs) == 0


def test_log_final_noop_when_run_none() -> None:
    """log_final should be no-op when wandb.run is None."""
    fake_wandb = FakeWandbModule(run_active=False)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.log_final({"test_loss": 0.25})

    assert len(fake_wandb._logs) == 0


def test_log_table_noop_when_run_none() -> None:
    """log_table should be no-op when wandb.run is None."""
    fake_wandb = FakeWandbModule(run_active=False)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.log_table("test", ["col"], [[1]])

    assert len(fake_wandb._logs) == 0


def test_finish_noop_when_run_none() -> None:
    """finish should be no-op when wandb.run is None."""
    fake_wandb = FakeWandbModule(run_active=False)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    publisher.finish()

    assert fake_wandb._finished is False


def test_is_enabled_false_when_run_none() -> None:
    """is_enabled should be False when wandb.run is None."""
    fake_wandb = FakeWandbModule(run_active=False)
    _setup_fake_wandb(fake_wandb)

    publisher = WandbPublisher(project="test", run_name="run-1", enabled=True)
    assert publisher.is_enabled is False


def test_raises_when_enabled_but_wandb_not_installed() -> None:
    """Should raise WandbUnavailableError when enabled but wandb not installed."""

    def _load_wandb_unavailable() -> WandbModuleProtocol:
        raise WandbUnavailableError("wandb package is not installed")

    hooks.load_wandb_module = _load_wandb_unavailable

    with pytest.raises(WandbUnavailableError, match="wandb package is not installed"):
        WandbPublisher(project="test", run_name="run-1", enabled=True)


_EXPECTED_PROJECT: Final[str] = "test-proj"
_EXPECTED_RUN_NAME: Final[str] = "run-1"


# ---------------------------------------------------------------------------
# Tests for _WandbModuleAdapter (production adapter for real wandb)
# ---------------------------------------------------------------------------


class RawWandbRun:
    """Fake raw wandb.Run (as returned by real wandb, not adapter)."""

    @property
    def id(self) -> str:
        return "raw-run-id"


class RawWandbConfig:
    """Fake raw wandb.config (as returned by real wandb, not adapter)."""

    def __init__(self) -> None:
        self.updates: list[dict[str, JSONValue]] = []

    def update(self, d: Mapping[str, JSONValue]) -> None:
        self.updates.append(dict(d))


class RawWandbTable:
    """Fake raw wandb.Table (as returned by real wandb, not adapter)."""

    def __init__(self, columns: list[str], data: list[list[float | int | str | bool]]) -> None:
        self.columns = columns
        self.data = data


class RawWandbModule:
    """Fake raw wandb module that simulates the real wandb interface.

    This simulates what `import wandb` returns, which has:
    - wandb.run -> Run object or None
    - wandb.config -> Config object
    - wandb.Table -> Table class (PascalCase!)
    - wandb.init() -> Run
    - wandb.log() -> None
    - wandb.finish() -> None

    Note: This intentionally does NOT implement WandbModuleProtocol because
    the production adapter wraps raw wandb modules that have PascalCase Table.
    """

    _TABLE_ATTR: Final[str] = "Table"

    def __init__(self) -> None:
        self._run: RawWandbRun | None = None
        self._config = RawWandbConfig()
        self._logs: list[dict[str, float | int | str | bool | WandbTableProtocol]] = []
        self._finished = False
        # Set Table attribute on instance to simulate wandb module interface
        # This allows getattr(self, "Table") to work as expected
        self.Table = RawWandbTable

    @property
    def run(self) -> RawWandbRun | None:
        return self._run

    @property
    def config(self) -> RawWandbConfig:
        return self._config

    @property
    def table_ctor(self) -> WandbTableCtorProtocol:
        """Return Table constructor - adapter accesses via getattr('Table')."""
        result: WandbTableCtorProtocol = getattr(self, self._TABLE_ATTR)
        return result

    def init(self, *, project: str, name: str) -> RawWandbRun:
        self._run = RawWandbRun()
        return self._run

    def log(self, data: Mapping[str, float | int | str | bool | WandbTableProtocol]) -> None:
        log_entry: dict[str, float | int | str | bool | WandbTableProtocol] = {}
        for key, val in data.items():
            log_entry[key] = val
        self._logs.append(log_entry)

    def finish(self) -> None:
        self._finished = True


class TestWandbModuleAdapter:
    """Tests for _WandbModuleAdapter that wraps real wandb module."""

    def test_adapter_run_property_returns_run(self) -> None:
        """Adapter should return run from wrapped module."""
        from platform_ml.testing import _WandbModuleAdapter

        raw_wandb: WandbModuleProtocol = RawWandbModule()
        raw_wandb.init(project="test", name="run")
        adapter = _WandbModuleAdapter(raw_wandb)

        # Check run exists and has correct id (strong assertion on actual value)
        run = adapter.run
        # Using guard pattern: narrow type, then assert on value
        if run is None:
            raise AssertionError("run should exist after init")
        assert run.id == "raw-run-id"

    def test_adapter_run_property_returns_none_when_no_run(self) -> None:
        """Adapter should return None when wrapped module has no run."""
        from platform_ml.testing import _WandbModuleAdapter

        raw_wandb: WandbModuleProtocol = RawWandbModule()
        adapter = _WandbModuleAdapter(raw_wandb)

        assert adapter.run is None

    def test_adapter_config_property(self) -> None:
        """Adapter should return config from wrapped module."""
        from platform_ml.testing import _WandbModuleAdapter

        raw_wandb = RawWandbModule()
        adapter = _WandbModuleAdapter(raw_wandb)

        adapter.config.update({"key": "value"})
        assert raw_wandb._config.updates == [{"key": "value"}]

    def test_adapter_table_ctor_property(self) -> None:
        """Adapter should return Table constructor from wrapped module."""
        from platform_ml.testing import _WandbModuleAdapter

        raw_wandb: WandbModuleProtocol = RawWandbModule()
        adapter = _WandbModuleAdapter(raw_wandb)

        table = adapter.table_ctor(columns=["a", "b"], data=[[1, 2]])
        assert table.columns == ["a", "b"]
        assert table.data == [[1, 2]]

    def test_adapter_init_method(self) -> None:
        """Adapter init should call wrapped module's init."""
        from platform_ml.testing import _WandbModuleAdapter

        raw_wandb = RawWandbModule()
        adapter = _WandbModuleAdapter(raw_wandb)

        run = adapter.init(project="my-proj", name="my-run")
        assert run.id == "raw-run-id"
        # Verify run was created by checking its id through the raw module
        if raw_wandb._run is None:
            raise AssertionError("_run should be set after init")
        assert raw_wandb._run.id == "raw-run-id"

    def test_adapter_log_method(self) -> None:
        """Adapter log should call wrapped module's log."""
        from platform_ml.testing import _WandbModuleAdapter

        raw_wandb = RawWandbModule()
        adapter = _WandbModuleAdapter(raw_wandb)

        adapter.log({"loss": 0.5, "acc": 0.9})
        assert raw_wandb._logs == [{"loss": 0.5, "acc": 0.9}]

    def test_adapter_finish_method(self) -> None:
        """Adapter finish should call wrapped module's finish."""
        from platform_ml.testing import _WandbModuleAdapter

        raw_wandb = RawWandbModule()
        adapter = _WandbModuleAdapter(raw_wandb)

        adapter.finish()
        assert raw_wandb._finished is True


class TestProductionLoadWandbModule:
    """Tests for _production_load_wandb_module function."""

    def test_production_loader_raises_when_wandb_not_available(self) -> None:
        """Production loader raises WandbUnavailableError when check returns False."""
        from platform_ml.testing import _production_load_wandb_module

        def _fake_check_unavailable() -> bool:
            return False  # Pretend wandb is not available

        # Inject the fake check
        hooks.check_wandb_available = _fake_check_unavailable

        # Should raise WandbUnavailableError
        with pytest.raises(WandbUnavailableError, match="wandb package is not installed"):
            _production_load_wandb_module()

    def test_production_loader_succeeds_when_wandb_available(self) -> None:
        """Production loader succeeds when wandb is available."""
        from platform_ml.testing import _production_load_wandb_module

        # Call the production loader (wandb is installed as test dependency)
        result = _production_load_wandb_module()
        # Verify it conforms to WandbModuleProtocol by calling methods
        _ = result.init  # Access method (would raise if not present)
        _ = result.log
        _ = result.finish

    def test_production_loader_with_fake_wandb_import(self) -> None:
        """Test production loader path using fake wandb import hook.

        This test injects a fake wandb module via the import_wandb hook
        and fakes the availability check to exercise the full load path.
        """
        from platform_ml.testing import _production_load_wandb_module

        # Create a fake wandb module that conforms to protocol
        fake_wandb: WandbModuleProtocol = RawWandbModule()

        def _fake_import_wandb() -> WandbModuleProtocol:
            return fake_wandb

        def _fake_check_wandb_available() -> bool:
            return True  # Pretend wandb is available

        # Inject the fake hooks
        hooks.check_wandb_available = _fake_check_wandb_available
        hooks.import_wandb = _fake_import_wandb

        # Now call the production loader with our fakes
        result = _production_load_wandb_module()

        # Result should be a _WandbModuleAdapter wrapping our fake
        # Verify by checking that it forwards methods correctly
        assert result.run is None  # RawWandbModule starts with no run
        run = result.init(project="test", name="run")
        assert run.id == "raw-run-id"


class TestProductionImportWandb:
    """Tests for _production_import_wandb function."""

    def test_production_import_wandb_raises_when_not_installed(self) -> None:
        """Production import raises ImportError when wandb not found."""
        import importlib.util

        from platform_ml.testing import _production_import_wandb

        spec = importlib.util.find_spec("wandb")
        if spec is None:
            # wandb not installed - __import__ will raise ModuleNotFoundError
            with pytest.raises(ModuleNotFoundError):
                _production_import_wandb()
        else:
            # wandb is installed - import should succeed
            result = _production_import_wandb()
            # Verify it's a module-like object
            _ = result.init
            _ = result.log
            _ = result.finish
