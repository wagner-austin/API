"""The submitter label: the one resolution every submitting command shares.

Lives in its own module rather than ``test_cli`` for a workspace reason, not
a design one: ``test_cli`` carried another session's uncommitted work when
this surface landed, and an explicit-path commit must not sweep a file two
sessions are editing. The subject is ``hpc3.cli._config``, so the module
name still says where the code under test lives.

The label is resolved against the board now (MCPs board task 3843d29f):
inside a session the shell's export is checked against the label the board
bound to ``CLAUDE_CODE_SESSION_ID``, and a terminal's export is taken as
given. The rule itself is ``platform_core.session_label``'s and is tested
there; what is pinned here is that ``hpc3`` feeds it the two variables, its
own poster and the core's file seam, and that a refusal reaches the
submitting command before anything is submitted.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.config import config_test_hooks
from platform_core.error_codes_tooling import SessionLabelErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, sent_arguments, tool_text_body
from platform_core.session_label import (
    API_KEY_NAME,
    LABEL_VARIABLE,
    SESSION_ID_VARIABLE,
    STACK_ENV_PATH,
    TASKBOARD_URL,
    TENANT_ID_NAME,
)

from hpc3.cli import _test_hooks as cli_hooks
from hpc3.cli import submit as submit_cli
from hpc3.cli._config import submitter_label
from hpc3.core import _test_hooks as core_hooks
from tests.against_hpc3 import read_ledger
from tests.conftest import (
    FakeRun,
    script_healthy_cluster,
    workspace_document,
    write_file,
    write_workspace,
)

SESSION = "4f8a2c1e-9b3d-4e7f-8a6b-1c2d3e4f5a6b"
BOUND = "opus-rebuild-deadlock-0910"
WRONG = "opus-dashboard-0911"
TENANT = "2e137b5f-0000-4000-8000-0000aa"
ENV_BYTES = f"{API_KEY_NAME}=internal-key\n{TENANT_ID_NAME}={TENANT}\n".encode()


def _pin(values: dict[str, str]) -> None:
    """Answer environment reads from ``values`` and nothing else.

    Args:
        values: The variables that are set; every other reads as unset.
    """

    def _env(key: str) -> str | None:
        return values.get(key)

    config_test_hooks.get_env = _env


def _whereis(label: str) -> McpHttpResponse:
    """``task_whereis`` for :data:`SESSION`, bound to ``label`` (or, empty,
    never written to the board), as taskboard-mcp renders it."""
    board = (
        f"board     {label} — 2 posts, last post 6m ago, C:\\x\n"
        if label != ""
        else "board     never written to this board (no label)\n"
    )
    return McpHttpResponse(
        status=200,
        body=tool_text_body(f"WHEREIS session {SESSION}\n\nsession   {SESSION}\n{board}"),
        content_type=EVENT_STREAM_MEDIA_TYPE,
    )


def _in_session(label: str, exported: str | None) -> tuple[FakeHttpPost, list[pathlib.Path]]:
    """Put the command inside a session: the harness's id, the stack's
    ``.env`` through the core's byte seam, and a board answering for it."""
    _pin({SESSION_ID_VARIABLE: SESSION} | ({} if exported is None else {LABEL_VARIABLE: exported}))
    read: list[pathlib.Path] = []
    real_read_bytes = core_hooks.read_bytes

    def _read_bytes(path: pathlib.Path) -> bytes:
        if path != STACK_ENV_PATH:
            return real_read_bytes(path)
        read.append(path)
        return ENV_BYTES

    core_hooks.read_bytes = _read_bytes
    post = FakeHttpPost([_whereis(label)])
    cli_hooks.http_post = post
    return post, read


class TestSubmitterLabel:
    """Unit behaviour of the resolution as hpc3 wires it."""

    def test_a_terminals_export_is_read_back_verbatim(self) -> None:
        _pin({LABEL_VARIABLE: "fable-brain-audit-0903"})
        assert submitter_label() == "fable-brain-audit-0903"

    def test_no_declaration_reads_as_the_positive_empty_string(self) -> None:
        """The ledger's "declared none", never a decode-time default."""
        _pin({})
        assert submitter_label() == ""

    def test_a_whitespace_declaration_names_nobody(self) -> None:
        """An export of spaces cannot become a label nothing can mention."""
        _pin({LABEL_VARIABLE: "   "})
        assert submitter_label() == ""

    def test_inside_a_session_an_unset_label_is_filled_from_the_board(self) -> None:
        post, read = _in_session(BOUND, None)
        assert submitter_label() == BOUND
        assert read == [STACK_ENV_PATH]
        assert post.urls == [TASKBOARD_URL]
        assert post.headers[0]["x-api-key"] == "internal-key"
        assert sent_arguments(post.bodies[0]) == {"session": SESSION}

    def test_inside_a_session_a_label_other_than_the_binding_is_refused(self) -> None:
        _in_session(BOUND, WRONG)
        with pytest.raises(AppError) as caught:
            submitter_label()
        assert caught.value.code is SessionLabelErrorCode.LABEL_MISMATCH
        assert BOUND in caught.value.message and WRONG in caught.value.message

    def test_a_session_the_board_never_saw_keeps_its_export(self) -> None:
        _in_session("", "fable-brain-audit-0903")
        assert submitter_label() == "fable-brain-audit-0903"


class TestSubmitCliRecordsTheSubmitter:
    """The label crosses from the environment into the row, end to end."""

    def test_the_declared_label_lands_in_the_ledger_row(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        declared_label: str,
    ) -> None:
        """Whoever the bridge should tag is whoever exported the label."""
        run = {
            "project": "abl",
            "name": "arm-b-42",
            "command": "python train.py",
            "artifact": None,
            "experiment": {"arm": "B", "seed": "42"},
        }
        write_file(tmp_path / "run.json", dump_json_str(run).encode("utf-8"))
        config = write_workspace(tmp_path / "hpc3.json", workspace_document())
        script_healthy_cluster(fake_run)

        submit_cli.main(["--config", config, "--run", str(tmp_path / "run.json")])

        entries = read_ledger(tmp_path / "ledger.jsonl")
        assert [e["submitter"] for e in entries] == [declared_label]

    def test_inside_a_session_the_boards_binding_lands_in_the_ledger_row(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Acceptance 2: the job is addressed with no export at all."""
        _in_session(BOUND, None)
        run = {
            "project": "abl",
            "name": "arm-b-42",
            "command": "python train.py",
            "artifact": None,
            "experiment": {"arm": "B", "seed": "42"},
        }
        write_file(tmp_path / "run.json", dump_json_str(run).encode("utf-8"))
        config = write_workspace(tmp_path / "hpc3.json", workspace_document())
        script_healthy_cluster(fake_run)

        submit_cli.main(["--config", config, "--run", str(tmp_path / "run.json")])

        entries = read_ledger(tmp_path / "ledger.jsonl")
        assert [e["submitter"] for e in entries] == [BOUND]
        assert not [line for line in emitted if "BOARD_AGENT_LABEL" in line]

    def test_inside_a_session_a_wrong_label_refuses_before_anything_is_submitted(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Acceptance 1, the 20:38Z case at this writer."""
        _in_session(BOUND, WRONG)
        run = {
            "project": "abl",
            "name": "arm-b-42",
            "command": "python train.py",
            "artifact": None,
            "experiment": {"arm": "B", "seed": "42"},
        }
        write_file(tmp_path / "run.json", dump_json_str(run).encode("utf-8"))
        config = write_workspace(tmp_path / "hpc3.json", workspace_document())
        script_healthy_cluster(fake_run)

        with pytest.raises(AppError) as caught:
            submit_cli.main(["--config", config, "--run", str(tmp_path / "run.json")])

        assert caught.value.code is SessionLabelErrorCode.LABEL_MISMATCH
        assert not (tmp_path / "ledger.jsonl").exists()
        assert not [command for command in fake_run.commands() if "sbatch" in command]
