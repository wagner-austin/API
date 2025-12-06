from __future__ import annotations

from typing import TypedDict

from torch.nn import Module
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD


def build_optimizer_and_scheduler(
    model: Module, cfg: OptimConfig
) -> tuple[Optimizer, LRScheduler | None]:
    if cfg["optim"] == "sgd":
        optimizer: Optimizer = SGD(
            model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"]
        )
    elif cfg["optim"] == "adam":
        optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    if cfg["scheduler"] == "cosine":
        scheduler: LRScheduler | None
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg["epochs"]), eta_min=cfg["min_lr"])
    elif cfg["scheduler"] == "step":
        scheduler = StepLR(optimizer, step_size=max(1, cfg["step_size"]), gamma=cfg["gamma"])
    else:
        scheduler = None
    return optimizer, scheduler


class OptimConfig(TypedDict):
    """Concrete configuration for optimizers and schedulers.

    This TypedDict satisfies the private ``_Cfg`` protocol used by
    ``build_optimizer_and_scheduler`` without dynamic dictionaries.
    It provides a standardized, strongly-typed way to construct
    optimizers across the codebase.
    """

    lr: float
    weight_decay: float
    optim: str
    scheduler: str
    epochs: int
    min_lr: float
    step_size: int
    gamma: float


def default_optim_config() -> OptimConfig:
    """Return the default, lightweight optimizer configuration.

    Defaults are chosen to minimize memory overhead during calibration
    while remaining representative of production training characteristics.
    """
    return {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "optim": "adamw",
        "scheduler": "none",
        "epochs": 1,
        "min_lr": 1e-5,
        "step_size": 10,
        "gamma": 0.5,
    }


__all__ = [
    "OptimConfig",
    "build_optimizer_and_scheduler",
    "default_optim_config",
]
