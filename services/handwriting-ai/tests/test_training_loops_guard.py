from __future__ import annotations

import pytest
import torch

from handwriting_ai.training import loops


class _StubLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: float | int | str) -> None:
        formatted = message % args if args else message
        self.messages.append(formatted)


class _SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.linear(x)
        return out


class _SingleBatchLoader:
    class _Iter:
        def __init__(self) -> None:
            self._emitted = False

        def __iter__(self) -> _SingleBatchLoader._Iter:
            return self

        def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
            if self._emitted:
                raise StopIteration
            self._emitted = True
            data = torch.zeros((1, 1), dtype=torch.float32)
            labels = torch.zeros((1,), dtype=torch.long)
            return data, labels

    def __iter__(self) -> _Iter:
        return self._Iter()

    def __len__(self) -> int:
        return 1


def test_train_epoch_raises_on_memory_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _SimpleModel()
    from torch.optim.sgd import SGD

    opt = SGD(model.parameters(), lr=0.1)
    loader = _SingleBatchLoader()

    logger = _StubLogger()

    def _get_logger(_: str) -> _StubLogger:
        return logger

    monkeypatch.setattr(loops, "get_logger", _get_logger)
    monkeypatch.setattr(loops, "on_batch_check", lambda: True)

    with pytest.raises(RuntimeError):
        _ = loops.train_epoch(
            model,
            loader,
            torch.device("cpu"),
            opt,
            ep=1,
            ep_total=1,
            total_batches=1,
        )

    assert any("mem_guard_abort" in msg for msg in logger.messages)
