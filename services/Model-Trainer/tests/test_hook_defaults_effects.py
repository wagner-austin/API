"""The HTTP client factory and the tree remover, run for real through failures.

Every service test that needs a client or a removal installs a fake, so none
shows the production client failing to reach a host or the production remover
refusing a tree that is already gone. These call the implementations
themselves (``effect-seam-twin``, board task cc7222ca).
"""

from __future__ import annotations

import socket
from pathlib import Path

import httpx
import pytest

from model_trainer.core._hook_defaults import (
    _default_httpx_client_factory,
    _default_shutil_rmtree,
)


def test_a_client_from_the_factory_raises_when_nothing_answers() -> None:
    """The corpus fetch must fail loudly on a dead host, not return nothing.

    The port is one a listener held and released, so nothing is bound to it.
    """
    released = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    released.bind(("127.0.0.1", 0))
    bound: tuple[str, int] = released.getsockname()
    released.close()
    client = _default_httpx_client_factory(timeout_seconds=10.0)
    with client, pytest.raises(httpx.ConnectError):
        client.get(f"http://127.0.0.1:{bound[1]}/corpus.txt")


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
