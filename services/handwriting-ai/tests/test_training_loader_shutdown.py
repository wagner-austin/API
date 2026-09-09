"""train_with_config shuts down its loaders' workers on every exit path.

The mechanism (``_test_hooks.shutdown_loader`` and its production default)
predates these tests and is covered by ``test_calibration_cleanup_processes``;
what was MISSING was the call from the main training path, and that absence
had a measured bill: persistent-worker processes stranded by GC-reliant
cleanup accumulated past 26GB RAM + 8GB swap on the CI host (2026-09-09,
runs 34292306485/34301896566 -- 31 stranded ``pt_data_worker`` processes in
the saturation census) and OOM-killed the hosted 16GB runner at ~98% before
that. These tests pin the wiring: both loaders, every exit path, and nothing
when the loaders were never built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import torch
from PIL import Image
from torch.nn import Module as TorchModule
from torch.optim.optimizer import Optimizer as TorchOptimizer
from torch.utils.data import Dataset

from handwriting_ai import _test_hooks
from handwriting_ai._hook_protocols_ml import ResourceLimitsDict
from handwriting_ai._hook_protocols_training import (
    BatchIterableProtocol,
    BatchLoaderProtocol,
    EffectiveConfig,
)
from handwriting_ai.training.calibration.ds_spec import PreprocessSpec
from handwriting_ai.training.dataset import DataLoaderConfig
from handwriting_ai.training.mnist_train import train_with_config
from handwriting_ai.training.train_config import TrainConfig, default_train_config


@pytest.fixture(autouse=True)
def _quiet_system_info() -> None:
    """Bind the system-info logger to a no-op; it reads container files."""
    _test_hooks.log_system_info = lambda: None


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    """A minimal picklable image dataset.

    Args:
        n: Number of items.
    """

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
        return img, idx % 10


class _RecordingShutdown:
    """A shutdown_loader fake that records exactly what it was handed.

    Satisfies ``ShutdownLoaderProtocol``. A list rather than a count, so a
    test can assert WHICH loaders were shut down, not merely how many times
    something was.

    Attributes:
        loaders: Every loader received, in call order.
    """

    loaders: list[BatchIterableProtocol]

    def __init__(self) -> None:
        self.loaders = []

    def __call__(self, loader: BatchIterableProtocol) -> None:
        self.loaders.append(loader)


def _cfg(tmp: Path) -> TrainConfig:
    """A one-epoch CPU config writing under the test's directory.

    Args:
        tmp: The test's temporary directory.

    Returns:
        The config.
    """
    return default_train_config(
        data_root=tmp / "data",
        out_dir=tmp / "out",
        model_id="mnist_resnet18_v1",
        epochs=1,
        batch_size=2,
        seed=0,
        device="cpu",
        calibrate=True,
        calibration_samples=1,
    )


def _bind_fast_calibration() -> None:
    """Bind the calibration hook to a fixed workerless EffectiveConfig."""
    loader_cfg = DataLoaderConfig(
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )
    fixed: EffectiveConfig = {
        "intra_threads": 1,
        "interop_threads": None,
        "batch_size": 1,
        "loader_cfg": loader_cfg,
    }

    def _fixed_calibrate(
        ds: PreprocessSpec,
        *,
        limits: ResourceLimitsDict,
        requested_batch_size: int,
        samples: int,
        cache_path: Path,
        ttl_seconds: int,
        force: bool,
    ) -> EffectiveConfig:
        return fixed

    _test_hooks.calibrate_input_pipeline = _fixed_calibrate


def _bind_train_epoch(outcome: float | None) -> None:
    """Bind the epoch hook to a fixed loss, or to raising.

    Args:
        outcome: The loss to report, or None to raise ``RuntimeError``.
    """

    def _epoch(
        model: TorchModule,
        train_loader: BatchLoaderProtocol,
        device: torch.device,
        precision: Literal["fp32", "fp16", "bf16"],
        optimizer: TorchOptimizer,
        ep: int,
        ep_total: int,
        total_batches: int,
    ) -> float:
        _ = precision
        if outcome is None:
            raise RuntimeError("epoch deliberately failed")
        return outcome

    _test_hooks.train_epoch = _epoch


def test_a_completed_run_shuts_down_both_loaders(tmp_path: Path) -> None:
    """Success path: exactly the train and test loaders, no more, no fewer.

    The two loaders are told apart by their datasets' sizes, which the two
    bases below deliberately make distinct.
    """
    recorder = _RecordingShutdown()
    _test_hooks.shutdown_loader = recorder
    _bind_fast_calibration()
    _bind_train_epoch(0.0)

    result = train_with_config(_cfg(tmp_path), (_TinyBase(2), _TinyBase(1)))

    assert result["model_id"] == "mnist_resnet18_v1"
    assert len(recorder.loaders) == 2
    # The loaders' identity is proven behaviourally: iterated at batch size
    # one, each yields exactly its base's item count, and the two bases are
    # deliberately distinct sizes -- so {2, 1} can only be the run's actual
    # train and test loaders, in either order.
    batch_counts = {sum(1 for _ in received) for received in recorder.loaders}
    assert batch_counts == {2, 1}


def test_a_failing_training_loop_still_shuts_down_both_loaders(tmp_path: Path) -> None:
    """The finally runs on the raising path, and the failure still propagates."""
    recorder = _RecordingShutdown()
    _test_hooks.shutdown_loader = recorder
    _bind_fast_calibration()
    _bind_train_epoch(None)

    with pytest.raises(RuntimeError, match="epoch deliberately failed"):
        train_with_config(_cfg(tmp_path), (_TinyBase(2), _TinyBase(1)))

    assert len(recorder.loaders) == 2


def test_a_failure_before_the_loaders_exist_shuts_down_nothing(tmp_path: Path) -> None:
    """The None branches: a fault before loader construction reaches the
    finally with no loaders, and the shutdown hook must not be handed None."""
    recorder = _RecordingShutdown()
    _test_hooks.shutdown_loader = recorder
    _bind_fast_calibration()

    def _has_interop() -> bool:
        return True

    def _raising_interop() -> int:
        raise RuntimeError("interop probe deliberately failed")

    _test_hooks.torch_has_get_num_interop_threads = _has_interop
    _test_hooks.torch_get_num_interop_threads = _raising_interop

    with pytest.raises(RuntimeError, match="interop probe deliberately failed"):
        train_with_config(_cfg(tmp_path), (_TinyBase(2), _TinyBase(1)))

    assert recorder.loaders == []
