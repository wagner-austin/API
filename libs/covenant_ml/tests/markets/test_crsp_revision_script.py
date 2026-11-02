"""The existing extraction CLI publishes keyless protocol-revision readbacks."""

from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments
from scripts import crsp_extraction

from covenant_ml.markets import _test_hooks

from ._crsp_revision_fixtures import revision_case
from .test_crsp_extraction_script import TASK
from .test_crsp_extraction_script import board as board


def test_existing_cli_revises_and_independently_rereads_without_private_keys(
    tmp_path: Path, board: None, capsys: pytest.CaptureFixture[str]
) -> None:
    case = revision_case(tmp_path)
    for key in (tmp_path / "keys").iterdir():
        key.unlink()
    arguments = [
        "--outputs",
        str(case.outputs),
        "--receipt-sha256",
        case.receipt_digest,
        "--previous-protocol",
        str(case.previous),
        "--protocol",
        str(case.protocol),
        "--access",
        str(tmp_path / "access.json"),
        "--destination",
        str(tmp_path / "revision"),
        "--report-task",
        TASK,
    ]
    reports: list[str] = []
    for command in ("revise-protocol", "verify-revision"):
        post = FakeHttpPost([posted_ok(), posted_ok()])
        _test_hooks.http_post = post
        assert crsp_extraction.main([command, *arguments]) == 0
        printed = capsys.readouterr().out.strip()
        report = narrow_json_to_dict(load_json_str(printed))
        assert report["opened_observations"] is False
        assert report["retained_relation_ciphertexts"] == 16
        assert (
            sent_arguments(post.bodies[-1])["body"]
            == "WRDS PROTOCOL REVISION LOAD-REPORT\n" + printed
        )
        reports.append(printed)
    assert reports[0] == reports[1]
