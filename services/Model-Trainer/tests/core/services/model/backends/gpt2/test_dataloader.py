from __future__ import annotations

from collections.abc import Generator

import torch

from model_trainer.core.services.model.backends.gpt2._dl import DataLoader


class _Ds:
    def __init__(self: _Ds, n: int) -> None:
        self._n = n

    def __len__(self: _Ds) -> int:
        return self._n

    def __getitem__(self: _Ds, idx: int) -> torch.Tensor:
        return torch.tensor(idx, dtype=torch.long)


def _collect_sizes(it: Generator[torch.Tensor, None, None]) -> list[int]:
    out: list[int] = []
    for batch in it:
        out.append(int(batch.size(0)))
    return out


def test_dataloader_batches() -> None:
    ds = _Ds(7)
    # shuffle=True exercises random branch; order is not asserted
    dl1 = DataLoader(ds, batch_size=3, shuffle=True)
    sizes1 = _collect_sizes(iter(dl1))
    # Expect 3 batches: sizes 3, 3, 1
    assert sorted(sizes1) == [1, 3, 3]

    # shuffle=False path also works
    dl2 = DataLoader(ds, batch_size=4, shuffle=False)
    sizes2 = _collect_sizes(iter(dl2))
    assert sizes2 == [4, 3]


def test_dataloader_with_loader_knobs() -> None:
    ds = _Ds(5)
    # Explicit knobs: num_workers=0 (single-process), pin_memory=True
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    sizes = _collect_sizes(iter(dl))
    assert sizes == [2, 2, 1]
