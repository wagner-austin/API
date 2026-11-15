"""This package's lock installs one NCCL distribution on each Python.

covenant_nn takes torch from the cu126 index on Python 3.11 and the cu132
index from 3.12, so that torch's NCCL matches the one xgboost (through
covenant_ml) brings on the same Python. platform_ml's tests prove
:func:`~platform_ml.nccl_lock.nccl_collisions` names a widened marker; this
case reads the committed ``poetry.lock`` beside this package.
"""

from __future__ import annotations

from pathlib import Path

from platform_ml.nccl_lock import nccl_collisions

_LOCK = Path(__file__).resolve().parents[1] / "poetry.lock"


def test_this_packages_lock_keeps_one_nccl_per_python() -> None:
    assert nccl_collisions(_LOCK.read_text(encoding="utf-8")) == {}
