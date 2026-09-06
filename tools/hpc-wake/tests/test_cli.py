"""The command: one flag, one cycle, loud failure."""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.mcp_testing import FakeHttpPost, posted_ok

from hpc_wake import _test_hooks
from hpc_wake.cli import wake as wake_cli
from tests.conftest import CONFIGURED_ENV, FakeRun, pin_env


def _write_workspace(tmp_path: pathlib.Path) -> str:
    """Write a minimal workspace document.

    Args:
        tmp_path: Directory the ledger resolves into.

    Returns:
        The document's path, ready to pass as ``--config``.
    """
    document: dict[str, JSONValue] = {
        "cluster": "hpc3",
        "host": "hpc3",
        "root": "/pub/w",
        "ledger": "ledger.jsonl",
        "quiet_seconds": 1800,
    }
    path = tmp_path / "hpc3.json"
    path.write_text(dump_json_str(document), encoding="utf-8")
    return str(path)


class TestMain:
    def test_a_cycle_runs_against_the_named_workspace(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([posted_ok()])
        config = _write_workspace(tmp_path)

        assert wake_cli.main(["--config", config]) == 0

        assert emitted == ["ledger is empty; nothing has been submitted from this machine"]

    def test_a_missing_config_flag_refuses(self) -> None:
        with pytest.raises(ValueError, match="--config"):
            wake_cli.main([])


class TestEntrypoint:
    def test_it_exits_with_mains_status(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([])
        config = _write_workspace(tmp_path)
        original = list(sys.argv)
        sys.argv[:] = ["hpc-wake", "--config", config]
        try:
            with pytest.raises(SystemExit) as caught:
                wake_cli.entrypoint()
        finally:
            sys.argv[:] = original
        assert caught.value.code == 0

    def test_running_as_a_module_actually_runs(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The half that silently goes missing without an ``if __name__``
        block: ``python -m hpc_wake.cli.wake`` would import, run nothing and
        exit 0 -- which from a scheduler's side reads as a quiet cycle."""
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([])
        config = _write_workspace(tmp_path)
        module_name = "hpc_wake.cli.wake"
        saved_argv = list(sys.argv)
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["hpc-wake", "--config", config]
        try:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv[:] = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module
        assert caught.value.code == 0
        assert emitted == ["ledger is empty; nothing has been submitted from this machine"]
