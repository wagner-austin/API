"""Real private-file and subprocess boundaries for research controller polling."""

from __future__ import annotations

import hashlib
import pathlib
import sys

import pytest
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.contracts.research import ResearchBackend, ResearchRequest
from fleet.core import _test_hooks, research_control


def _spec(root: pathlib.Path, document: JSONObject) -> ResearchRequest:
    """Write an exact private specification into its content-addressed directory.

    Args:
        root: Test's private store.
        document: Metadata to freeze.

    Returns:
        The queue reference to the written bytes.
    """
    payload = dump_json_str(document).encode()
    digest = hashlib.sha256(payload).hexdigest()
    directory = root / digest
    directory.mkdir()
    (directory / "request.json").write_bytes(payload)
    return ResearchRequest(backend=ResearchBackend.HPC3, spec_sha256=digest, attempt="pilot")


def test_private_bytes_are_bound_to_tenant_backend_attempt_and_digest(
    tmp_path: pathlib.Path,
) -> None:
    document: JSONObject = {"tenant": "ours", "backend": "hpc3", "attempt": "pilot"}
    request = _spec(tmp_path, document)
    assert research_control.read_specification(tmp_path, request, "ours") == document
    for changed in (
        {"tenant": "other", "backend": "hpc3", "attempt": "pilot"},
        {"tenant": "ours", "backend": "wrds", "attempt": "pilot"},
        {"tenant": "ours", "backend": "hpc3", "attempt": "another"},
    ):
        row: JSONObject = dict(changed)
        different = _spec(tmp_path, row)
        with pytest.raises(ValueError, match="RESEARCH_SPEC_OWNER"):
            research_control.read_specification(tmp_path, different, "ours")
    (tmp_path / request["spec_sha256"] / "request.json").write_text("{}")
    with pytest.raises(ValueError, match="RESEARCH_SPEC_DRIFT"):
        research_control.read_specification(tmp_path, request, "ours")


def test_frozen_inputs_refuse_changed_relative_and_missing_files(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "policy.json"
    source.write_bytes(b"frozen")
    document: JSONObject = {
        "policy": {"path": str(source), "sha256": hashlib.sha256(b"frozen").hexdigest()}
    }
    assert research_control.verified_input(document, "policy") == source
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="RESEARCH_INPUT_DRIFT"):
        research_control.verified_input(document, "policy")
    source.unlink()
    with pytest.raises(FileNotFoundError):
        research_control.verified_input(document, "policy")
    with pytest.raises(ValueError, match="RESEARCH_INPUT_FIELDS"):
        research_control.verified_input({"policy": {"path": str(source)}}, "policy")
    with pytest.raises(ValueError, match="RESEARCH_SPEC_PATH"):
        research_control.require_path({"path": "relative"}, "path")


def test_real_controller_process_success_failure_and_timeout(tmp_path: pathlib.Path) -> None:
    """The production process seam really runs and kills a controller child."""
    _test_hooks.run = _test_hooks._default_run
    log = tmp_path / "controller.jsonl"
    result = research_control.invoke((sys.executable, "-c", 'print(\'{"job_id":"123"}\')'), log, 10)
    assert result == {"job_id": "123"}
    with pytest.raises(ValueError, match="RESEARCH_CONTROLLER_PROCESS: exit 7"):
        research_control.invoke((sys.executable, "-c", "raise SystemExit(7)"), log, 10)
    with pytest.raises(ValueError, match="RESEARCH_CONTROLLER_PROCESS"):
        research_control.invoke((sys.executable, "-c", "import time; time.sleep(60)"), log, 1)
    refusal = "print('{\"attempts\":[]}'); raise SystemExit(2)"
    accepted = research_control.invoke((sys.executable, "-c", refusal), log, 10, accepted=(0, 2))
    assert accepted == {"attempts": []}
    with pytest.raises(ValueError, match="RESEARCH_CONTROLLER_PROCESS: exit 2"):
        research_control.invoke((sys.executable, "-c", refusal), log, 10)
    assert len(log.read_text().splitlines()) == 5
