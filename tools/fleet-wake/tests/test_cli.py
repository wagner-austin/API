"""The command: one flag, one cycle, loud failure."""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.json_utils import JSONObject, dump_json_str
from platform_core.mcp_testing import FakeHttpPost, posted_ok

from fleet_wake import _test_hooks
from fleet_wake.cli import wake as wake_cli
from tests.conftest import CONFIGURED_ENV, pin_env


def _write_workspace(tmp_path: pathlib.Path) -> str:
    """Write a minimal fleet workspace document.

    Args:
        tmp_path: Directory the records resolve into.

    Returns:
        The document's path, ready to pass as ``--config``.
    """
    document: JSONObject = {
        "nodes": {
            "lavender": {
                "host": "lavender",
                "stage_root": "C:/fleet/stage",
                "logical_cores": 16,
                "ram_gb": 32.0,
                "gpu": None,
                "enabled": True,
                "budget": {
                    "reserved_cores": 2,
                    "reserved_ram_gb": 4.0,
                    "worker_ram_gb": 1.1,
                    "max_concurrent_runs": 2,
                    "max_disk_gb": 20.0,
                },
            }
        },
        "not_dispatchable": {},
        "projects": {
            "tools/fleet": {
                "worker_ram_gb": 1.1,
                "minimum_workers": 2,
                "expected_minutes": 5,
            }
        },
        "ledger": "runs/ledger.jsonl",
        "feed": "runs/feed.jsonl",
        "leases": "runs/leases.json",
    }
    path = tmp_path / "fleet.json"
    path.write_text(dump_json_str(document), encoding="utf-8")
    return str(path)


class TestMain:
    def test_a_cycle_runs_against_the_named_workspace(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([posted_ok()])
        config = _write_workspace(tmp_path)

        assert wake_cli.main([wake_cli.CONFIG_FLAG, config]) == 0

        assert emitted == ["ledger is empty; nothing has been dispatched from this machine"]

    def test_a_missing_config_flag_refuses(self) -> None:
        """There is no default workspace. A bridge that guessed one would
        announce a different machine's dispatches, or none, silently."""
        with pytest.raises(ValueError, match="--config"):
            wake_cli.main([])

    def test_an_unknown_flag_refuses_rather_than_being_ignored(self) -> None:
        with pytest.raises(ValueError, match="unknown argument"):
            wake_cli.main(["--follow"])


class TestEntrypoint:
    def test_it_exits_with_mains_status(self, tmp_path: pathlib.Path, emitted: list[str]) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([posted_ok()])
        config = _write_workspace(tmp_path)
        original = sys.argv
        sys.argv = ["fleet-wake", wake_cli.CONFIG_FLAG, config]

        try:
            with pytest.raises(SystemExit) as caught:
                wake_cli.entrypoint()
        finally:
            sys.argv = original

        assert caught.value.code == 0

    def test_running_as_a_module_actually_runs(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        """Without the ``__main__`` guard, ``python -m fleet_wake.cli.wake``
        imports the module, runs nothing and exits 0 -- which looks exactly
        like a cycle that had nothing to say."""
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([posted_ok()])
        config = _write_workspace(tmp_path)
        original = sys.argv
        sys.argv = ["fleet-wake", wake_cli.CONFIG_FLAG, config]

        try:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("fleet_wake.cli.wake", run_name="__main__")
        finally:
            sys.argv = original

        assert caught.value.code == 0
        assert emitted == ["ledger is empty; nothing has been dispatched from this machine"]
