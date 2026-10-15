"""Exercise every batch CLI action through real services, sockets and subprocesses."""

from __future__ import annotations

import runpy
import subprocess
import sys
from collections.abc import Callable, Generator
from contextlib import closing
from pathlib import Path
from uuid import UUID

import pytest
from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.http_client import HttpxClient, build_client
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from scripts.kalshi_demo_batch import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.demo_batch import DemoBatchService, build_demo_batch
from covenant_ml.markets.execution_storage import persist_evidence
from covenant_ml.markets.kalshi_auth import KalshiSigner
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._batch_fixtures import (
    account_document,
    batch_intent,
    serve_batch_quote,
    serve_snapshot,
    settled_predecessor,
)
from ._kalshi_fixtures import KEY_ID, account_clock, private_pem, signing_key
from ._server import ApiServer, api_server
from .test_demo_settlement import serve_credit

__all__ = ["api_server"]


@pytest.fixture()
def batch_cli(api_server: ApiServer, tmp_path: Path) -> Generator[ApiServer, None, None]:
    """Inject the real service factory targeting loopback and temporary account storage."""
    original_factory, original_clock = _test_hooks.demo_batch_factory, _test_hooks.clock

    def factory(
        client: HttpxClient, signer: KalshiSigner, clock: Callable[[], str], guard_root: Path
    ) -> DemoBatchService:
        assert guard_root == Path.home() / ".local" / "share" / "covenant-ml" / "demo-batches"
        return build_demo_batch(
            client,
            signer,
            clock,
            tmp_path / ".local" / "share" / "covenant-ml" / "demo-batches",
            origin=api_server.url,
            identity=lambda: UUID(int=2),
        )

    _test_hooks.demo_batch_factory, _test_hooks.clock = factory, account_clock
    yield api_server
    _test_hooks.demo_batch_factory, _test_hooks.clock = original_factory, original_clock


def test_cli_actions_and_actual_subprocess_replay(
    batch_cli: ApiServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key, signer = signing_key()
    previous, root = tmp_path / "trial", tmp_path / "batch"
    with closing(build_client(5)) as client:
        baseline = settled_predecessor(batch_cli, client, signer, previous)
    identity, pem, tickers = tmp_path / "key-id", tmp_path / "key.pem", tmp_path / "tickers.json"
    identity.write_text(KEY_ID, encoding="utf-8")
    pem.write_bytes(private_pem(key))
    persist_evidence(tickers, {"tickers": ["MARKET-2"]})
    base = ["--key-id-file", str(identity), "--private-key-file", str(pem), "--root", str(root)]
    serve_credit(batch_cli, previous)
    assert (
        main(
            [
                *base,
                "prepare",
                "--predecessor",
                str(previous),
                "--tickers-file",
                str(tickers),
                "--limit-price",
                "500",
            ],
            stand_in_pin,
        )
        == 0
    )
    assert narrow_json_to_dict(load_json_str(capsys.readouterr().out))["tickers"] == ["MARKET-2"]
    row = batch_intent(2)
    serve_snapshot(batch_cli, account_document((baseline,), (100,)))
    serve_batch_quote(batch_cli, row, 100)
    serve_snapshot(batch_cli, account_document((baseline, row), (100, 100)))
    assert main([*base, "advance"], stand_in_pin) == 0
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert report["confirmed_batch_orders"] == 1
    assert report["exposure"] == 600
    assert main([*base, "status"], stand_in_pin) == 0
    assert narrow_json_to_dict(load_json_str(capsys.readouterr().out)) == report
    serve_snapshot(batch_cli, account_document((baseline, row), (100, 100)))
    assert main([*base, "reconcile"], stand_in_pin) == 0
    assert narrow_json_to_dict(load_json_str(capsys.readouterr().out)) == report
    environment = config_test_hooks.get_environment()
    environment.update(USERPROFILE=str(tmp_path), HOME=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "scripts.kalshi_demo_batch", *base, "status"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    report["expired"] = True
    assert narrow_json_to_dict(load_json_str(result.stdout)) == report
    assert len(batch_cli.bodies) == 2
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.kalshi_demo_batch", run_name="__main__")
