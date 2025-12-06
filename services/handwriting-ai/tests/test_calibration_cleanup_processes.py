from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from handwriting_ai.training.calibration import measure as m


def test_shutdown_loader_uses_iterator() -> None:
    class _Iter:
        def __init__(self) -> None:
            self.closed = False

        def _shutdown_workers(self) -> None:
            self.closed = True

    class _DL(DataLoader[tuple[torch.Tensor, torch.Tensor]]):
        def __init__(self) -> None:  # do not call super
            pass

    loader = _DL()
    name = "_iterator"
    setattr(loader, name, _Iter())

    m._shutdown_loader(loader)
    # After shutdown, the iterator should be cleared
    assert loader._iterator is None


def test_join_active_children_handles_alive_and_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _P:
        def __init__(self, alive: bool) -> None:
            self._alive = alive

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None) -> None:
            events.append("join")

        def terminate(self) -> None:
            self._alive = False
            events.append("term")

    children: list[_P] = [_P(True), _P(False)]

    def _active_children() -> list[_P]:
        nonlocal children
        current = children
        children = []
        return current

    class _MP:
        def __init__(self) -> None:
            self._fn = _active_children

        def active_children(self) -> list[_P]:
            return self._fn()

    monkeypatch.setattr(m, "_mp", _MP(), raising=False)

    m._join_active_children()
    # We should have attempted joins/terminations on the alive child
    assert "join" in events


def test_join_active_children_breaks_when_no_alive_after_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Dead:
        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float | None = None) -> None:
            return None

        def terminate(self) -> None:
            return None

    calls = {"n": 0}

    def _children_once() -> list[_Dead]:
        calls["n"] += 1
        # First pass: one dead child so alive_after_pass stays False
        # Second call: no children to trigger the final break branch
        return [_Dead()] if calls["n"] == 1 else []

    class _MP2:
        def active_children(self) -> list[_Dead]:
            return _children_once()

    monkeypatch.setattr(m, "_mp", _MP2(), raising=False)
    m._join_active_children()


def test_join_active_children_marks_alive_after_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Child:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            return None

        def terminate(self) -> None:
            return None

    class _MP:
        def active_children(self) -> list[_Child]:
            # Always report an alive child so the inner branch sets alive_after_pass
            return [_Child()]

    monkeypatch.setattr(m, "_mp", _MP(), raising=False)

    # Should iterate bounded times and exercise alive_after_pass=True path
    m._join_active_children()


def test_join_active_children_join_makes_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Flip:
        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None) -> None:
            self._alive = False

        def terminate(self) -> None:
            return None

    calls = {"n": 0}

    def _children_once() -> list[_Flip]:
        calls["n"] += 1
        return [_Flip()] if calls["n"] == 1 else []

    class _MP3:
        def active_children(self) -> list[_Flip]:
            return _children_once()

    monkeypatch.setattr(m, "_mp", _MP3(), raising=False)
    m._join_active_children()
