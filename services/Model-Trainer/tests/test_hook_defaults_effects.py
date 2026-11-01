"""The tree remover, run for real through failures.

Every service test that needs a removal installs a fake, so none shows the
production remover refusing a tree that is already gone. These call the
implementation itself (``effect-seam-twin``, board task cc7222ca). The HTTP
client factory beside it left with data-bank-api on 2026-11-01.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model_trainer.core._hook_defaults import _default_shutil_rmtree


def test_the_remover_clears_a_populated_tree(tmp_path: Path) -> None:
    run = tmp_path / "run-1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "step-10.pt").write_bytes(b"weights")

    _default_shutil_rmtree(run)

    assert not run.exists()


def test_the_remover_raises_on_a_tree_that_is_already_gone(tmp_path: Path) -> None:
    """A second removal means two owners thought they held the run."""
    with pytest.raises(FileNotFoundError):
        _default_shutil_rmtree(tmp_path / "run-gone")
