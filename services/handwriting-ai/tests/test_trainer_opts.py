from __future__ import annotations

import torch
import torch.autograd
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.mnist_train import (
    _build_model,
    _build_optimizer_and_scheduler,
    _configure_threads,
    _train_epoch,
)
from handwriting_ai.training.train_config import TrainConfig, default_train_config


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        # Simple white canvas with a small black square
        img = Image.new("L", (32, 32), 255)
        for y in range(12, 20):
            for x in range(14, 18):
                img.putpixel((x, y), 0)
        return img, idx % 10


def test_optimizer_scheduler_variants() -> None:
    for opt in ("sgd", "adam", "adamw"):
        for sched in ("none", "cosine", "step"):
            # Fresh model each iteration to ensure loss can decrease
            model = _build_model()
            cfg = default_train_config(optim=opt, scheduler=sched, epochs=2, step_size=1)
            optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg)

            # Capture weights before optimizer step
            first_param = next(model.parameters())
            weights_before = first_param.data.clone()

            # Create dummy gradients to ensure step updates weights
            dummy_input: Tensor = torch.randn((1, 1, 28, 28))
            dummy_target: Tensor = torch.zeros(1, dtype=torch.long)
            loss_fn = torch.nn.CrossEntropyLoss()

            # Compute initial loss before training
            with torch.no_grad():
                pred_before: Tensor = model(dummy_input)
                loss_t_before: Tensor = loss_fn(pred_before, dummy_target)
                loss_before = float(loss_t_before.item())

            # Train for a few steps
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                output: Tensor = model(dummy_input)
                loss_t: Tensor = loss_fn(output, dummy_target)
                torch.autograd.backward(loss_t)
                optimizer.step()

            # Verify loss decreases after training
            with torch.no_grad():
                pred_after: Tensor = model(dummy_input)
                loss_t_after: Tensor = loss_fn(pred_after, dummy_target)
                loss_after = float(loss_t_after.item())
            assert loss_after < loss_before, f"loss should decrease: {loss_before} -> {loss_after}"

            # Verify weights changed after optimizer step
            weights_after = first_param.data
            assert not torch.allclose(weights_before, weights_after), "weights should change"

            # Scheduler presence by mode
            if sched == "none":
                assert scheduler is None
            else:
                if scheduler is None:
                    raise AssertionError("expected scheduler")


def test_train_epoch_smoke_with_fake_data() -> None:
    base = _TinyBase(4)
    cfg = default_train_config()
    ds = PreprocessDataset(base, cfg)
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        ds, batch_size=4, shuffle=False
    )
    model = _build_model()
    optimizer, _ = _build_optimizer_and_scheduler(model, cfg)
    device = torch.device("cpu")
    loss = _train_epoch(
        model,
        loader,
        device,
        optimizer,
        ep=1,
        ep_total=1,
        total_batches=len(loader),
    )
    assert type(loss) is float
    assert loss >= 0.0


def test_augment_flag_yields_valid_sample() -> None:
    base = _TinyBase(1)
    cfg = default_train_config(augment=True, aug_rotate=5.0, aug_translate=0.1)
    ds = PreprocessDataset(base, cfg)
    x, y = ds[0]
    assert list(x.shape) == [1, 28, 28]
    assert 0 <= int(y) <= 9


class _ThreadCfgAdapter:
    """Adapter to satisfy ThreadConfig Protocol from TrainConfig."""

    def __init__(self, cfg: TrainConfig) -> None:
        self._cfg = cfg

    def __getitem__(self, key: str) -> int:
        return int(self._cfg["threads"])


def test_configure_threads_no_crash() -> None:
    cfg = default_train_config(threads=1)
    _configure_threads(_ThreadCfgAdapter(cfg))
    assert torch.get_num_threads() >= 1
