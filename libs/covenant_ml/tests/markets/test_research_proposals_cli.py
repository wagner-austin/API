"""Real proposal CLI persistence, refusal exit codes and numerical startup boundary."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.determinism_cpu import NativeLibrariesAlreadyLoadedError
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_list,
)
from scripts.research_proposals import main

from covenant_ml.markets.research_records import encode_research_policy
from tests.benchmarking._cli_fixtures import stand_in_pin

from ._research_final_fixtures import completed_study, write_trial
from ._research_fixtures import policy, proposal


def _files(root: Path) -> list[str]:
    policy_path = root / "policy.json"
    policy_path.write_text(dump_json_str(encode_research_policy(policy())), encoding="utf-8")
    return ["--policy", str(policy_path), "--ledger", str(root / "attempts.sqlite")]


def test_init_admit_refuse_and_inspect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _files(tmp_path)
    assert main([*args, "init"], stand_in_pin) == 0
    assert narrow_json_to_dict(load_json_str(capsys.readouterr().out))["attempts"] == []
    request = tmp_path / "proposal.json"
    request.write_text(dump_json_str(proposal()), encoding="utf-8")
    call = [*args, "propose", "--proposal", str(request)]
    assert main(call, stand_in_pin) == 0
    admitted = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert narrow_json_to_dict(require_list(admitted, "attempts")[0])["status"] == "admitted"
    assert main(call, stand_in_pin) == 2
    refused = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert narrow_json_to_dict(require_list(refused, "attempts")[0])["reason"] == (
        "candidate_seed_consumed"
    )
    assert main([*args, "status"], stand_in_pin) == 0
    status = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert require_list(status, "attempts") == [
        *require_list(admitted, "attempts"),
        *require_list(refused, "attempts"),
    ]


def test_real_process_refusal_and_restart(tmp_path: Path) -> None:
    args = _files(tmp_path)
    command = [sys.executable, "-m", "scripts.research_proposals", *args]
    initialized = subprocess.run(
        [*command, "init"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert narrow_json_to_dict(load_json_str(initialized.stdout))["attempts"] == []
    request = proposal()
    request["partition"] = "final"
    request_path = tmp_path / "proposal.json"
    request_path.write_text(dump_json_str(request), encoding="utf-8")
    refused = subprocess.run(
        [*command, "propose", "--proposal", str(request_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert refused.returncode == 2
    report = narrow_json_to_dict(load_json_str(refused.stdout))
    row = narrow_json_to_dict(require_list(report, "attempts")[0])
    assert row["reason"] == "holdout_forbidden"
    restarted = subprocess.run(
        [*command, "status"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert narrow_json_to_dict(load_json_str(restarted.stdout)) == report


def test_evaluator_selection_seals_the_journal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger, trial, manifest = completed_study(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(dump_json_str(encode_research_policy(trial["policy"])), "utf-8")
    request = write_trial(trial, tmp_path / "request.json")
    args = ["--policy", str(policy_path), "--ledger", str(ledger)]
    selection = ["select", "--trial", str(request), "--final-manifest", str(manifest)]
    assert main([*args, *selection], stand_in_pin) == 0
    report = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    row = narrow_json_to_dict(require_list(report, "attempts")[0])
    assert (row["status"], row["selection"]) == ("complete", "selected")
    with pytest.raises(ValueError, match="RESEARCH-008"):
        main([*args, *selection], stand_in_pin)
    release = [
        "release",
        "--trial",
        str(request),
        "--final-manifest",
        str(manifest),
        "--access",
        str(tmp_path / "validation/access.json"),
        "--directory",
        str(tmp_path / "grant"),
    ]
    assert main([*args, *release], stand_in_pin) == 0
    released = narrow_json_to_dict(load_json_str(capsys.readouterr().out))
    assert narrow_json_to_dict(require_list(released, "attempts")[0])["selection"] == "released"
    assert (tmp_path / "grant/final.agekey").is_file()
    assert set(released) == {"policy_digest", "attempts"}
    with pytest.raises(ValueError, match="RESEARCH-009"):
        main([*args, *release], stand_in_pin)


def test_entry_refuses_late_numeric_pin() -> None:
    with pytest.raises(NativeLibrariesAlreadyLoadedError):
        runpy.run_module("scripts.research_proposals", run_name="__main__")
