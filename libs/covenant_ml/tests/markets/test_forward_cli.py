"""The forward report CLI: pinned plan bytes, persisted provenance, a real subprocess."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import narrow_json_to_dict, require_list
from scripts.forward_weather_report import main

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.validation import read_document
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._forward_fixtures import (
    fixed_clock,
    forward_campaign,
    plan_document,
    restore_clock,
    write_plan,
)
from ._server import ApiServer, api_server

__all__ = ["api_server", "restore_clock"]


def test_cli_scores_settled_evidence_under_the_pinned_plan(
    api_server: ApiServer,
    tmp_path: Path,
    restore_clock: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    campaign, root, digest, observed = forward_campaign(
        api_server, tmp_path, [("binary", 1), ("binary", 0), ("binary", 1)]
    )
    plan_path = tmp_path / "forward-plan.json"
    plan_sha = write_plan(
        plan_path,
        plan_document(campaign, digest, [plan["event_day"] for plan in campaign["plans"]]),
    )
    _test_hooks.clock = fixed_clock(observed)
    output = tmp_path / "forward.json"
    args = [
        "--plan",
        str(plan_path),
        "--plan-sha256",
        plan_sha,
        "--root",
        str(root),
        "--out",
        str(output),
        "--market-url",
        api_server.url,
        "--forecast-url",
        api_server.url,
    ]
    requests = len(api_server.requests)
    # The fixture's captures print their own report lines; read only the scorer's.
    capsys.readouterr()
    assert main(args, stand_in_pin) == 0
    assert capsys.readouterr().out == (
        f"Forward evaluation: {output}\nRun record: {output}.runrecord.json\n"
    )
    assert len(api_server.requests) == requests
    report = read_document(output)
    assert (report["scored_dates"], report["plan_sha256"], report["observed_at"]) == (
        3,
        plan_sha,
        observed,
    )
    assert str(plan_path.resolve()) in [
        narrow_json_to_dict(value)["path"] for value in require_list(report, "sources")
    ]
    record = read_document(Path(str(output) + ".runrecord.json"))
    assert (record["experiment"], record["label"]) == ("forward-weather-evaluation", plan_sha)
    with pytest.raises(FileExistsError, match="MARKET-027"):
        main(args, stand_in_pin)
    with pytest.raises(ValueError, match="FORWARD-001"):
        main([*args[:3], "e" * 64, *args[4:]], stand_in_pin)


def test_cli_real_subprocess_refuses_while_a_settlement_is_outstanding(
    api_server: ApiServer, tmp_path: Path, restore_clock: None
) -> None:
    # One plan: its window opens when the fixture runs, so the subprocess's real
    # clock reads after its capture and before its settlement window, two days on.
    campaign, root, digest, _ = forward_campaign(api_server, tmp_path, [("unsettled", 1)])
    plan_path = tmp_path / "forward-plan.json"
    plan_sha = write_plan(
        plan_path,
        plan_document(campaign, digest, [plan["event_day"] for plan in campaign["plans"]]),
    )
    output = tmp_path / "subprocess.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.forward_weather_report",
            "--plan",
            str(plan_path),
            "--plan-sha256",
            plan_sha,
            "--root",
            str(root),
            "--out",
            str(output),
            "--market-url",
            api_server.url,
            "--forecast-url",
            api_server.url,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (completed.returncode, completed.stdout) == (1, "")
    assert completed.stderr.splitlines()[-1] == (
        "ValueError: [FORWARD-007] 1 declared acquisitions are still outstanding, first paper-test"
    )
    assert (output.exists(), Path(str(output) + ".runrecord.json").exists()) == (False, False)
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.forward_weather_report", run_name="__main__")
