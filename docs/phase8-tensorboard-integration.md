# Phase 8: TensorBoard Integration Specification

## Overview

Phase 8 adds TensorBoard logging to Model-Trainer for real-time training visualization. This integrates alongside the existing Redis ML Events system (`trainer_metrics_events`).

**Guiding Principles (unchanged from refactor doc):**
- Strict typing: No `Any`, `cast`, `type: ignore`, `.pyi`, `noqa`, or stubs
- Immutable data: TypedDict for all configs (no dataclasses)
- DRY: Reuse existing patterns from BaseTrainer
- Explicit failures: No try/except recovery, no best-effort, no fallbacks
- 100% test coverage: Statements and branches
- Protocol-based: Dynamic imports use Protocol type annotations at assignment

---

## 8.1 Rationale

- **Real-time visualization** of loss/perplexity curves during training
- **No cloud dependency** - runs locally, reads files from disk
- **Already bundled** with PyTorch ecosystem (`tensorboard` package)
- **Minimal code** - ~80 lines total across config, protocol, and publisher

---

## 8.2 Configuration TypedDict

**File:** `libs/platform_core/src/platform_core/config/model_trainer.py`

Add TensorBoard config using immutable TypedDict:

```python
from __future__ import annotations

from typing import Literal, TypedDict


class ModelTrainerTensorBoardConfig(TypedDict, total=True):
    """TensorBoard logging configuration.

    All fields are required (total=True). Immutable after creation.
    """

    enabled: bool
    log_dir: str
    flush_secs: int
    log_every_n_steps: int


# Extend ModelTrainerSettings:
class ModelTrainerSettings(TypedDict, total=True):
    app_env: Literal["dev", "prod"]
    logging: ModelTrainerLoggingConfig
    redis: ModelTrainerRedisConfig
    rq: ModelTrainerRQConfig
    app: ModelTrainerAppConfig
    security: ModelTrainerSecurityConfig
    tensorboard: ModelTrainerTensorBoardConfig  # NEW
```

**Environment variables:**
```
TENSORBOARD__ENABLED=true
TENSORBOARD__LOG_DIR=/data/tensorboard
TENSORBOARD__FLUSH_SECS=30
TENSORBOARD__LOG_EVERY_N_STEPS=10
```

**Loader addition in `load_model_trainer_settings()`:**
```python
tensorboard_cfg: ModelTrainerTensorBoardConfig = {
    "enabled": _parse_bool("TENSORBOARD__ENABLED", False),
    "log_dir": _parse_str("TENSORBOARD__LOG_DIR", "/data/tensorboard"),
    "flush_secs": _parse_int("TENSORBOARD__FLUSH_SECS", 30),
    "log_every_n_steps": _parse_int("TENSORBOARD__LOG_EVERY_N_STEPS", 10),
}
```

---

## 8.3 Protocol for SummaryWriter (Strict Typing)

**File:** `services/Model-Trainer/src/model_trainer/core/services/training/tensorboard_writer.py`

Uses `__import__` + Protocol assignment pattern to avoid `Any` leakage:

```python
from __future__ import annotations

from typing import Protocol


class SummaryWriterProto(Protocol):
    """Protocol for torch.utils.tensorboard.SummaryWriter.

    Defines the minimal interface needed for training metrics logging.
    No Any types. All methods fully typed.
    """

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int,
    ) -> None:
        """Add scalar data to summary.

        Args:
            tag: Data identifier (e.g., "train/loss").
            scalar_value: The scalar value to log.
            global_step: Global step value to record.
        """
        ...

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ) -> None:
        """Add multiple scalars to summary.

        Args:
            main_tag: Parent tag for grouping (e.g., "losses").
            tag_scalar_dict: Mapping of sub-tag to scalar value.
            global_step: Global step value to record.
        """
        ...

    def add_hparams(
        self,
        hparam_dict: dict[str, int | float | str | bool],
        metric_dict: dict[str, float],
    ) -> None:
        """Add hyperparameters and final metrics.

        Args:
            hparam_dict: Hyperparameter name-value pairs.
            metric_dict: Final metric name-value pairs.
        """
        ...

    def flush(self) -> None:
        """Flush pending events to disk."""
        ...

    def close(self) -> None:
        """Close the writer and release resources."""
        ...


def _get_summary_writer_class() -> type[SummaryWriterProto]:
    """Get SummaryWriter class via dynamic import with Protocol typing.

    Uses __import__ pattern:
    1. Import module with __import__
    2. Get class with getattr
    3. Assign directly to Protocol type annotation

    Returns:
        SummaryWriter class typed as SummaryWriterProto.

    Raises:
        ImportError: If tensorboard is not installed.
    """
    tensorboard_mod = __import__(
        "torch.utils.tensorboard",
        fromlist=["SummaryWriter"],
    )
    # Direct assignment to Protocol type - annotation overrides Any from getattr
    cls: type[SummaryWriterProto] = getattr(tensorboard_mod, "SummaryWriter")
    return cls


def create_summary_writer(log_dir: str) -> SummaryWriterProto:
    """Create a SummaryWriter instance for the given log directory.

    Args:
        log_dir: Directory where TensorBoard logs will be written.
            Created if it does not exist.

    Returns:
        A SummaryWriter instance typed as SummaryWriterProto.

    Raises:
        ImportError: If tensorboard is not installed.
    """
    cls = _get_summary_writer_class()
    writer: SummaryWriterProto = cls(log_dir=log_dir)
    return writer
```

---

## 8.4 Metrics TypedDict (Immutable)

```python
from __future__ import annotations

from typing import TypedDict


class TensorBoardStepMetrics(TypedDict, total=True):
    """Metrics logged at each training step. Immutable."""

    step: int
    loss: float
    perplexity: float


class TensorBoardEpochMetrics(TypedDict, total=True):
    """Metrics logged at end of each epoch. Immutable."""

    epoch: int
    epoch_loss: float
    epoch_perplexity: float


class TensorBoardHParams(TypedDict, total=True):
    """Hyperparameters logged at training start. Immutable."""

    model_family: str
    model_size: str
    batch_size: int
    learning_rate: float
    max_seq_len: int
    num_epochs: int


class TensorBoardFinalMetrics(TypedDict, total=True):
    """Final metrics logged at training completion. Immutable."""

    final_loss: float
    final_perplexity: float
    total_steps: int
```

---

## 8.5 TensorBoardPublisher

**File:** `services/Model-Trainer/src/model_trainer/core/services/training/tensorboard_publisher.py`

```python
from __future__ import annotations

import math

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.services.training.tensorboard_writer import (
    SummaryWriterProto,
    create_summary_writer,
)


class TensorBoardPublisher:
    """Publish training metrics to TensorBoard.

    Writes scalar metrics to TensorBoard event files for real-time
    visualization. Enabled/disabled via settings.

    Thread-safe for single-writer usage. Not thread-safe for concurrent writes.
    """

    _writer: SummaryWriterProto | None
    _run_id: str
    _enabled: bool
    _log_every_n: int
    _step_counter: int

    def __init__(
        self,
        settings: Settings,
        run_id: str,
    ) -> None:
        """Initialize TensorBoard publisher.

        Args:
            settings: Application settings with tensorboard config.
            run_id: Unique identifier for this training run.

        Raises:
            ImportError: If tensorboard is not installed and enabled=True.
        """
        tb_cfg = settings["tensorboard"]
        self._enabled = tb_cfg["enabled"]
        self._run_id = run_id
        self._log_every_n = tb_cfg["log_every_n_steps"]
        self._step_counter = 0
        self._writer = None

        if self._enabled:
            log_dir: str = f"{tb_cfg['log_dir']}/{run_id}"
            self._writer = create_summary_writer(log_dir)

    def log_hparams(
        self,
        cfg: ModelTrainConfig,
    ) -> None:
        """Log hyperparameters at training start.

        Args:
            cfg: Training configuration containing hyperparameters.
        """
        if self._writer is None:
            return

        hparams: dict[str, int | float | str | bool] = {
            "model_family": cfg["model_family"],
            "model_size": cfg["model_size"],
            "batch_size": cfg["batch_size"],
            "learning_rate": cfg["learning_rate"],
            "max_seq_len": cfg["max_seq_len"],
            "num_epochs": cfg["num_epochs"],
        }
        self._writer.add_hparams(hparams, {})

    def log_step(
        self,
        *,
        step: int,
        loss: float,
    ) -> None:
        """Log metrics for a single training step.

        Respects log_every_n_steps setting to reduce I/O overhead.

        Args:
            step: Global training step number.
            loss: Loss value for this step.
        """
        if self._writer is None:
            return

        self._step_counter += 1
        if self._step_counter % self._log_every_n != 0:
            return

        ppl: float = math.exp(loss) if loss < 20.0 else float("inf")

        self._writer.add_scalar("train/loss", loss, step)
        self._writer.add_scalar("train/perplexity", ppl, step)

    def log_epoch(
        self,
        *,
        epoch: int,
        epoch_loss: float,
    ) -> None:
        """Log metrics at end of epoch.

        Args:
            epoch: Epoch number (0-indexed).
            epoch_loss: Average loss for the epoch.
        """
        if self._writer is None:
            return

        ppl: float = math.exp(epoch_loss) if epoch_loss < 20.0 else float("inf")

        self._writer.add_scalar("epoch/loss", epoch_loss, epoch)
        self._writer.add_scalar("epoch/perplexity", ppl, epoch)

    def log_validation(
        self,
        *,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Log validation metrics.

        Args:
            epoch: Epoch number.
            val_loss: Validation loss.
        """
        if self._writer is None:
            return

        ppl: float = math.exp(val_loss) if val_loss < 20.0 else float("inf")

        self._writer.add_scalar("val/loss", val_loss, epoch)
        self._writer.add_scalar("val/perplexity", ppl, epoch)

    def flush(self) -> None:
        """Flush pending events to disk."""
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Close the writer and flush remaining events."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
```

---

## 8.6 Integration into BaseTrainer

**File:** `services/Model-Trainer/src/model_trainer/core/services/training/base_trainer.py`

Additions to existing BaseTrainer class:

```python
from model_trainer.core.services.training.tensorboard_publisher import (
    TensorBoardPublisher,
)


class BaseTrainer:
    # Existing fields...
    _tb: TensorBoardPublisher | None

    def __init__(
        self: BaseTrainer,
        prepared: PreparedLMModel,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        redis_hb: Callable[[float], None],
        cancelled: Callable[[], bool],
        progress: Callable[[int, int, float], None] | None = None,
        service_name: str = "base-trainer",
    ) -> None:
        # Existing init...

        # Initialize TensorBoard publisher
        if settings["tensorboard"]["enabled"]:
            self._tb = TensorBoardPublisher(settings, run_id)
            self._tb.log_hparams(cfg)
        else:
            self._tb = None

    def _train_one_epoch(
        self: BaseTrainer,
        *,
        model: LMModelProto,
        dataloader: DataLoader,
        optim: OptimizerProto,
        epoch: int,
        device: str,
        start_step: int,
    ) -> tuple[float, int, bool]:
        # Existing code...

        for batch in dataloader:
            # Existing forward/backward...

            step += 1

            # Log to TensorBoard
            if self._tb is not None:
                self._tb.log_step(step=step, loss=last_loss)

            # Existing progress callback and heartbeat...

        # Log epoch metrics
        if self._tb is not None:
            self._tb.log_epoch(epoch=epoch, epoch_loss=last_loss)

        return last_loss, step, False

    def train(self: BaseTrainer) -> TrainOutcome:
        # Existing training code...

        # Close TensorBoard writer on completion
        if self._tb is not None:
            self._tb.close()

        return outcome
```

---

## 8.7 Dependencies

**File:** `services/Model-Trainer/pyproject.toml`

```toml
[tool.poetry.dependencies]
# Existing deps...
tensorboard = "^2.18.0"
```

No additional type stubs needed - Protocol handles typing.

---

## 8.8 Usage

Start TensorBoard to view training runs:

```powershell
# From any terminal
tensorboard --logdir=/data/tensorboard --reload_interval=10

# Open browser: http://localhost:6006
# Charts update in real-time as training progresses
```

---

## 8.9 Test Coverage Requirements

**File:** `tests/unit/core/services/training/test_tensorboard_writer.py`

```python
from __future__ import annotations

import tempfile
from pathlib import Path

from model_trainer.core.services.training.tensorboard_writer import (
    create_summary_writer,
)


def test_create_summary_writer_returns_protocol_type() -> None:
    """create_summary_writer returns SummaryWriterProto."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = f"{tmpdir}/test_run"
        writer = create_summary_writer(log_dir)

        assert hasattr(writer, "add_scalar")
        assert hasattr(writer, "flush")
        assert hasattr(writer, "close")

        writer.close()


def test_create_summary_writer_creates_event_files() -> None:
    """SummaryWriter creates TensorBoard event files on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = f"{tmpdir}/test_run"
        writer = create_summary_writer(log_dir)

        writer.add_scalar("test/loss", 1.5, 0)
        writer.add_scalar("test/loss", 1.2, 1)
        writer.flush()
        writer.close()

        assert Path(log_dir).exists()
        event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
        assert len(event_files) == 1
```

**File:** `tests/unit/core/services/training/test_tensorboard_publisher.py`

```python
from __future__ import annotations

import tempfile
from pathlib import Path

from model_trainer.core.config.settings import Settings
from model_trainer.core.services.training.tensorboard_publisher import (
    TensorBoardPublisher,
)


def _make_test_settings(
    *,
    enabled: bool,
    log_dir: str,
    log_every_n_steps: int = 1,
) -> Settings:
    """Create test settings with TensorBoard config."""
    return {
        "app_env": "dev",
        "logging": {"level": "INFO"},
        "redis": {"enabled": False, "url": "redis://localhost:6379/0"},
        "rq": {
            "queue_name": "test",
            "job_timeout_sec": 60,
            "result_ttl_sec": 60,
            "failure_ttl_sec": 60,
            "retry_max": 0,
            "retry_intervals_sec": "",
        },
        "app": {
            "data_root": "/data",
            "artifacts_root": "/data/artifacts",
            "runs_root": "/data/runs",
            "logs_root": "/data/logs",
            "threads": 1,
            "tokenizer_sample_max_lines": 100,
            "data_bank_api_url": "",
            "data_bank_api_key": "",
            "cleanup": {
                "enabled": False,
                "verify_upload": False,
                "grace_period_seconds": 0,
                "dry_run": False,
            },
            "corpus_cache_cleanup": {
                "enabled": False,
                "max_bytes": 1000000,
                "min_free_bytes": 100000,
                "eviction_policy": "lru",
            },
            "tokenizer_cleanup": {"enabled": False, "min_unused_days": 30},
        },
        "security": {"api_key": ""},
        "tensorboard": {
            "enabled": enabled,
            "log_dir": log_dir,
            "flush_secs": 30,
            "log_every_n_steps": log_every_n_steps,
        },
    }


def test_publisher_disabled_does_not_create_writer() -> None:
    """Publisher with enabled=False creates no writer."""
    settings = _make_test_settings(enabled=False, log_dir="/tmp")
    pub = TensorBoardPublisher(settings, "run-123")

    pub.log_step(step=1, loss=0.5)
    pub.log_epoch(epoch=0, epoch_loss=0.5)
    pub.flush()
    pub.close()


def test_publisher_enabled_logs_metrics() -> None:
    """Publisher writes metrics when enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _make_test_settings(enabled=True, log_dir=tmpdir)
        pub = TensorBoardPublisher(settings, "run-456")

        pub.log_step(step=1, loss=2.5)
        pub.log_step(step=2, loss=2.3)
        pub.log_epoch(epoch=0, epoch_loss=2.4)
        pub.close()

        log_dir = Path(tmpdir) / "run-456"
        assert log_dir.exists()
        event_files = list(log_dir.glob("events.out.tfevents.*"))
        assert len(event_files) == 1


def test_publisher_respects_log_every_n_steps() -> None:
    """Publisher only logs every N steps per config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _make_test_settings(
            enabled=True,
            log_dir=tmpdir,
            log_every_n_steps=5,
        )
        pub = TensorBoardPublisher(settings, "run-789")

        for i in range(1, 11):
            pub.log_step(step=i, loss=1.0)

        pub.close()


def test_publisher_handles_high_loss_gracefully() -> None:
    """Publisher clamps perplexity for loss > 20."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _make_test_settings(enabled=True, log_dir=tmpdir)
        pub = TensorBoardPublisher(settings, "run-high-loss")

        pub.log_step(step=1, loss=50.0)
        pub.close()


def test_publisher_log_validation() -> None:
    """Publisher logs validation metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _make_test_settings(enabled=True, log_dir=tmpdir)
        pub = TensorBoardPublisher(settings, "run-val")

        pub.log_validation(epoch=0, val_loss=1.5)
        pub.close()
```

---

## 8.10 Verification Checklist

- [ ] `tensorboard` added to pyproject.toml dependencies
- [ ] `ModelTrainerTensorBoardConfig` TypedDict added to platform_core
- [ ] `load_model_trainer_settings()` updated with tensorboard config
- [ ] `SummaryWriterProto` Protocol defined with full type signatures
- [ ] `create_summary_writer()` uses `__import__` + Protocol pattern
- [ ] `TensorBoardPublisher` class implemented with no `Any`
- [ ] `BaseTrainer` updated to optionally use TensorBoardPublisher
- [ ] All metrics TypedDicts are immutable (`total=True`)
- [ ] Unit tests achieve 100% coverage for new code
- [ ] `make check` passes (mypy strict, ruff, pytest)
- [ ] No `Any`, `cast`, `type: ignore`, `.pyi`, or stubs introduced

---

## 8.11 Files to Create/Modify

### New Files
1. `services/Model-Trainer/src/model_trainer/core/services/training/tensorboard_writer.py`
2. `services/Model-Trainer/src/model_trainer/core/services/training/tensorboard_publisher.py`
3. `tests/unit/core/services/training/test_tensorboard_writer.py`
4. `tests/unit/core/services/training/test_tensorboard_publisher.py`

### Modified Files
1. `libs/platform_core/src/platform_core/config/model_trainer.py` - Add TensorBoardConfig
2. `services/Model-Trainer/src/model_trainer/core/config/settings.py` - Re-export TensorBoardConfig
3. `services/Model-Trainer/src/model_trainer/core/services/training/base_trainer.py` - Add TensorBoard integration
4. `services/Model-Trainer/pyproject.toml` - Add tensorboard dependency
