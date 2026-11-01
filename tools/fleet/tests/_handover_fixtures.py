"""Real pinned request files and existing process/report seams for handover tests."""

from __future__ import annotations

import pathlib
import sys

from platform_core.json_utils import JSONObject, dump_json_str, load_json_str, require_str
from platform_core.mcp_testing import FakeHttpPost, posted_ok, stack_endpoint_files
from platform_core.session_label import STACK_ENV_PATH

from fleet.contracts.research import ResearchBackend, ResearchRequest
from fleet.core import _test_hooks, staging
from tests.conftest import FakeRun, ok
from tests.test_research_hpc import _document, _row

TASK = "33333333-cccc-4ccc-8ccc-333333333333"


def pin(path: pathlib.Path) -> JSONObject:
    """Pin actual fixture bytes using the production digest function."""
    return {"path": str(path), "sha256": staging.digest(path.read_bytes())}


def freeze(root: pathlib.Path, document: JSONObject, name: str) -> str:
    """Write exact content-addressed metadata without changing earlier documents."""
    payload = dump_json_str(document)
    digest = staging.digest(payload.encode())
    directory = root / digest
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(payload)
    return digest


def setup(root: pathlib.Path) -> tuple[ResearchRequest, JSONObject, JSONObject, JSONObject]:
    """Create an original request, replacement and completed native result plan."""
    document = _document(root)
    document["python"] = sys.executable
    original = freeze(root, document, "request.json")
    request = ResearchRequest(backend=ResearchBackend.HPC3, spec_sha256=original, attempt="trial-1")
    replacement = root / "replacement-python"
    replacement.write_text("#!/bin/sh\nexit 7\n")
    replacement.chmod(0o700)
    target = {**document, "python": str(replacement)}
    target_digest = freeze(root, target, "request.json")
    provenance = root / "provenance.json"
    provenance.write_text(
        dump_json_str(
            {
                "controller_python": str(replacement),
                "controller_python_sha256": staging.digest(replacement.read_bytes()),
            }
        )
    )
    selfcheck = root / "selfcheck.py"
    selfcheck.write_text("print('image self-check OK')\n")
    (root / "work/checksums.txt").write_text("frozen artifact checksums\n")
    row = _row("complete")
    record = load_json_str(require_str(row, "record"))
    plan: JSONObject = {
        "request_sha256": original,
        "replacement_sha256": target_digest,
        "provenance": pin(provenance),
        "selfcheck": pin(selfcheck),
        "wheels_dir": str(root),
        "native_job_id": "12345",
        "record_sha256": staging.digest(dump_json_str(record).encode()),
        "recipe": pin(root / "work/run.json"),
        "checksums": pin(root / "work/checksums.txt"),
    }
    measured: JSONObject = {
        "controller_python": str(replacement),
        "controller_python_sha256": staging.digest(replacement.read_bytes()),
        "provenance_sha256": staging.digest(provenance.read_bytes()),
    }
    return request, document, plan, measured


def replies(measured: JSONObject) -> FakeRun:
    """Answer inspection, journal status and fresh native result observation."""
    row = _row("complete")
    return FakeRun(
        [
            ok(dump_json_str(measured)),
            ok(dump_json_str({"attempts": [row]})),
            ok(
                dump_json_str(
                    {
                        "ledger": {"job_id": "12345"},
                        "record": load_json_str(require_str(row, "record")),
                    }
                )
            ),
        ]
    )


def reporting() -> FakeHttpPost:
    """Bind only stack configuration; research files are always read from disk."""
    files = stack_endpoint_files()
    files[STACK_ENV_PATH] = "MCP_INTERNAL_KEY=test-key\nOPERATOR_TENANT_ID=ours\n"

    def read(path: pathlib.Path) -> str:
        return files[path] if path in files else path.read_text()

    _test_hooks.read_text = read
    post = FakeHttpPost([posted_ok(), posted_ok()])
    _test_hooks.http_post = post
    return post


__all__ = ["TASK", "freeze", "pin", "replies", "reporting", "setup"]
