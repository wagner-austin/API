"""Reading the registry, and the connection a new document copies from it.

``declared_projects``'s duplicate refusal is exercised in
``test_research_index``, which held it before the lift; these are the reads
``hpc3-register`` added.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core.registry import (
    shared_connection,
    workspace_documents,
    workspace_filename,
)
from tests.conftest import project_config, workspace_document, write_file, write_workspace


def test_a_project_is_declared_in_a_document_named_after_it() -> None:
    assert workspace_filename("tankpit") == "hpc3-tankpit.json"


class TestWhichDocumentsAreWorkspaces:
    """The ``projects`` table is what makes a document a workspace."""

    def test_run_documents_beside_them_are_skipped(self, tmp_path: pathlib.Path) -> None:
        """A run document lives in the same directory and declares nothing.

        Args:
            tmp_path: Directory holding the documents.
        """
        _ = write_workspace(tmp_path / "hpc3-abl.json")
        write_file(tmp_path / "arm-b.json", b'{"project": "abl", "command": "true"}')

        assert list(workspace_documents(tmp_path)) == ["hpc3-abl.json"]


class TestTheSharedConnection:
    """What a new workspace copies, and the refusals when there is no one answer."""

    def test_the_ledger_comes_back_as_written(self, tmp_path: pathlib.Path) -> None:
        """A committed document must carry the relative form, not this machine's path.

        Args:
            tmp_path: Directory holding the documents.
        """
        _ = write_workspace(tmp_path / "hpc3-abl.json")
        _ = write_workspace(
            tmp_path / "hpc3-other.json",
            workspace_document(quiet_seconds=600, projects={"other": project_config()}),
        )

        connection = shared_connection(tmp_path)

        assert (connection["cluster"], connection["host"], connection["root"]) == (
            "hpc3",
            "hpc3",
            "/pub/w",
        )
        assert connection["ledger"] == "ledger.jsonl"

    def test_workspaces_naming_two_roots_are_refused(self, tmp_path: pathlib.Path) -> None:
        """Picking one would be guessing which machine the new project runs on.

        Args:
            tmp_path: Directory holding the documents.
        """
        _ = write_workspace(tmp_path / "hpc3-abl.json")
        _ = write_workspace(
            tmp_path / "hpc3-other.json",
            workspace_document(root="/pub/elsewhere", projects={"other": project_config()}),
        )

        with pytest.raises(AppError) as refused:
            _ = shared_connection(tmp_path)

        assert refused.value.code is Hpc3ErrorCode.REGISTRATION_INCOMPLETE
        assert "/pub/elsewhere" in refused.value.message
        assert "['hpc3-abl.json', 'hpc3-other.json']" in refused.value.message

    def test_a_directory_with_no_workspace_is_refused(self, tmp_path: pathlib.Path) -> None:
        """No sibling means nothing to copy the cluster's address from.

        Args:
            tmp_path: An empty directory.
        """
        with pytest.raises(AppError) as refused:
            _ = shared_connection(tmp_path)

        assert refused.value.code is Hpc3ErrorCode.REGISTRATION_INCOMPLETE
        assert "declare [] (cluster, host, root, ledger) across []" in refused.value.message
