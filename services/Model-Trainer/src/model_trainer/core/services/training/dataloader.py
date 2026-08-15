"""Strictly-typed adapter over ``torch.utils.data.DataLoader``.

This lives in the training package rather than inside a model backend because
nothing about it is backend-specific, and three consumers outside the gpt2
backend already depended on it: ``base_trainer``, ``char_lstm/evaluate`` and
``gpt2/evaluate``.

Its previous home in ``backends/gpt2/_dl.py`` also closed an import cycle.
``base_trainer`` imported it, which initialised ``backends/gpt2/__init__``,
which imported ``backends/gpt2/train``, which imported ``base_trainer`` while
that module was still executing. The cycle only surfaced when ``base_trainer``
was reached first, which is what the hf_lm backend does through its deferred
``_default_create_trainer`` import: every hf_lm training run died with
``ImportError: cannot import name 'BaseTrainer' from partially initialized
module``. Test import order initialises the gpt2 package first, so the suite
never reproduced it.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Protocol

import torch


class _Indexable(Protocol):
    def __len__(self: _Indexable) -> int: ...
    def __getitem__(self: _Indexable, idx: int) -> torch.Tensor: ...


class _TorchDLInst(Protocol):
    def __iter__(self: _TorchDLInst) -> Generator[torch.Tensor, None, None]: ...


class _TorchDLCtor(Protocol):
    def __call__(
        self: _TorchDLCtor,
        dataset: _Indexable,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
    ) -> _TorchDLInst: ...


class DataLoader:
    """Adapter over torch.utils.data.DataLoader with strict typing.

    Maintains the same usage as previous local loader, while enabling
    num_workers and pin_memory knobs for performance.
    """

    def __init__(
        self: DataLoader,
        ds: _Indexable,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        self._ds = ds
        self._bs = int(batch_size)
        self._shuffle = bool(shuffle)
        self._nw = int(num_workers)
        self._pm = bool(pin_memory)

    def __len__(self: DataLoader) -> int:
        """Return the number of batches in the loader."""
        ds_len = len(self._ds)
        # Ceiling division: (ds_len + batch_size - 1) // batch_size
        return (ds_len + self._bs - 1) // self._bs

    def __iter__(self: DataLoader) -> Generator[torch.Tensor, None, None]:
        tud = __import__("torch.utils.data", fromlist=["DataLoader"])
        torch_dataloader: _TorchDLCtor = tud.DataLoader
        loader = torch_dataloader(
            self._ds,
            batch_size=self._bs,
            shuffle=self._shuffle,
            num_workers=self._nw,
            pin_memory=self._pm,
        )
        yield from loader
