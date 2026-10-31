"""Staging reports bind verified transport to its independent digest record."""

import sys
from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject, dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments
from platform_core.stage_manifest import decode_stage_manifest

from hpc3.cli import _test_hooks as cli_hooks
from hpc3.cli import stage
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.digest import sha256_hex
from hpc3.core.stage import certification_text
from tests.conftest import FakeRun, workspace_document, write_workspace
from tests.wrds_board import TASK, accepting_board


def arguments(root: Path, fake_run: FakeRun) -> list[str]:
    """Write real source files and independent identities for the public command."""
    payload = b"encrypted transport fixture"
    source = root / "source"
    source.mkdir()
    (source / "sample.age").write_bytes(payload)
    manifest: JSONObject = {
        "destination": "/private/study",
        "files": [
            {"name": "sample.age", "sha256": sha256_hex(payload), "size_bytes": len(payload)}
        ],
        "provenance": {"partition": "train-validation"},
    }
    text = dump_json_str(manifest)
    (root / "stage.json").write_text(text)
    (root / "expected.txt").write_text(sha256_hex(payload))
    certificate = certification_text(decode_stage_manifest(load_json_str(text))).encode()
    fake_run.add("sha256sum '/private/study/sample.age'", stdout=sha256_hex(payload) + "  file\n")
    fake_run.add(
        "sha256sum '/private/study/stage-digests.txt'", stdout=sha256_hex(certificate) + "  file\n"
    )
    return [
        "--config",
        write_workspace(root / "workspace.json", workspace_document()),
        "--manifest",
        str(root / "stage.json"),
        "--source-dir",
        str(source),
        "--expect-from",
        str(root / "expected.txt"),
        "--report-task",
        TASK,
    ]


def test_report_names_verified_bytes_and_controller(
    tmp_path: Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    board = accepting_board(2)
    assert stage.main(arguments(tmp_path, fake_run)) == 0
    report = narrow_json_to_dict(load_json_str(emitted[-1]))
    assert report["kind"] == "HPC3_STAGE_EVIDENCE"
    assert report["host"] == "hpc3"
    assert report["verified_files"] == 1
    assert report["manifest"] == load_json_str((tmp_path / "stage.json").read_text())
    assert report["manifest_sha256"] == sha256_hex((tmp_path / "stage.json").read_bytes())
    assert report["expected_sha256"] == sha256_hex((tmp_path / "expected.txt").read_bytes())
    assert report["certification_path"] == "/private/study/stage-digests.txt"
    manifest = decode_stage_manifest(report["manifest"])
    assert report["certification_sha256"] == sha256_hex(certification_text(manifest).encode())
    assert report["controller_python"] == sys.executable
    assert report["controller_python_sha256"] == sha256_hex(Path(sys.executable).read_bytes())
    assert sent_arguments(board.bodies[-1])["body"] == "HPC3 STAGE LOAD-REPORT\n" + emitted[-1]
    assert fake_run.commands()[-1] == "sha256sum '/private/study/stage-digests.txt'"


def test_bad_task_refuses_before_transport(tmp_path: Path, fake_run: FakeRun) -> None:
    board = accepting_board(2)
    args = arguments(tmp_path, fake_run)
    args[-1] = "invalid"
    with pytest.raises(ValueError, match="WRDS_REPORT_TASK"):
        stage.main(args)
    assert fake_run.calls == []
    assert board.bodies == []


@pytest.mark.parametrize("failure", ["transfer", "certificate", "changed-expected"])
def test_failed_transport_or_changed_evidence_never_posts(
    tmp_path: Path, fake_run: FakeRun, failure: str
) -> None:
    board = accepting_board(2)
    if failure == "transfer":
        fake_run.add("cat >", returncode=255, stderr="connection lost")
    elif failure == "certificate":
        fake_run.add("sha256sum '/private/study/stage-digests.txt'", stdout="0" * 64 + "  file\n")
    args = arguments(tmp_path, fake_run)
    if failure == "changed-expected":
        original_read = core_hooks.read_bytes
        reads: list[Path] = []

        def read(path: Path) -> bytes:
            if path == tmp_path / "expected.txt":
                reads.append(path)
                if len(reads) == 3:
                    path.write_text("altered independent evidence")
            return original_read(path)

        core_hooks.read_bytes = read
        with pytest.raises(ValueError, match="STAGE_EXPECTED_CHANGED"):
            stage.main(args)
    else:
        with pytest.raises(AppError, match=r"connection lost|on the cluster"):
            stage.main(args)
    assert board.bodies == []


def test_board_failure_propagates_after_verified_transport(
    tmp_path: Path, fake_run: FakeRun
) -> None:
    accepting_board(0)
    board = FakeHttpPost(
        [
            posted_ok(),
            McpHttpResponse(status=503, body="unavailable", content_type=EVENT_STREAM_MEDIA_TYPE),
        ]
    )
    cli_hooks.http_post = board
    with pytest.raises(AppError, match="HTTP 503"):
        stage.main(arguments(tmp_path, fake_run))
    assert len(board.bodies) == 2
    assert fake_run.commands()[-1] == "sha256sum '/private/study/stage-digests.txt'"
