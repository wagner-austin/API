"""Execute the actual rendered controller script and native CLI with real child failures."""

from __future__ import annotations

import hashlib
import pathlib
import sys

import pytest
from platform_core.json_utils import dump_json_str, load_json_str

from fleet.contracts.dispatch import decode_job
from fleet.contracts.research_worker import decode_worker, encode_worker
from fleet.core import _test_hooks, research_queue
from tests._research_fixtures import research_job, worker
from tests.test_research_hpc import _document, _row


@pytest.mark.parametrize("exit_code", [0, 7])
def test_rendered_shell_and_native_cli_run_real_controller_children(
    tmp_path: pathlib.Path, exit_code: int
) -> None:
    """The rendered shell is executed, not merely searched for command text."""
    document = _document(tmp_path)
    controller = tmp_path / "script"
    controller.write_text(
        "print("
        + repr(dump_json_str({"attempts": [_row("running")]}))
        + ")\n"
        + f"raise SystemExit({exit_code})\n"
    )
    document["python"] = sys.executable
    document["script"] = {
        "path": str(controller),
        "sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
    }
    env = tmp_path / "env.ps1"
    env.write_text("# intentionally no external credentials in this process twin\n")
    payload = dump_json_str(document).encode()
    digest = hashlib.sha256(payload).hexdigest()
    directory = tmp_path / digest
    directory.mkdir()
    (directory / "request.json").write_bytes(payload)
    placed = encode_worker(worker())
    placed.update(
        {"python": sys.executable, "root": str(tmp_path), "env_file": str(env), "tenant": "ours"}
    )
    row = research_job(
        project="research/hpc3",
        researchRequest={"backend": "hpc3", "specSha256": digest, "attempt": "trial-1"},
    )
    script = tmp_path / "poll.sh"
    script.write_text(research_queue._script(decode_worker(placed), decode_job(row, answer="")))
    result = _test_hooks._default_run(("/bin/sh", str(script)), timeout_seconds=30)
    if exit_code == 0:
        assert result["returncode"] == 0
        assert load_json_str(result["stdout"]) == {
            "state": "running",
            "native_job_id": "12345",
            "record_sha256": None,
        }
    else:
        assert result["returncode"] != 0
        assert "RESEARCH_CONTROLLER_PROCESS: exit 7" in result["stderr"]
    assert not (directory / "poll.lock").exists()
    transcript = (directory / "controller.jsonl").read_text()
    assert f'"exit": {exit_code}' in transcript
