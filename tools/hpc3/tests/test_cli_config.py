"""The submitter label: the one environment reading every submitting command shares.

Lives in its own module rather than ``test_cli`` for a workspace reason, not
a design one: ``test_cli`` carried another session's uncommitted work when
this surface landed, and an explicit-path commit must not sweep a file two
sessions are editing. The subject is ``hpc3.cli._config``, so the module
name still says where the code under test lives.
"""

from __future__ import annotations

import pathlib

from platform_core.config import config_test_hooks
from platform_core.json_utils import dump_json_str

from hpc3.cli import submit as submit_cli
from hpc3.cli._config import SUBMITTER_ENV, submitter_label
from tests.against_hpc3 import read_ledger
from tests.conftest import (
    FakeRun,
    script_healthy_cluster,
    workspace_document,
    write_file,
    write_workspace,
)


class TestSubmitterLabel:
    """Unit behaviour of the reader itself."""

    def _pin(self, value: str | None) -> None:
        """Answer only :data:`SUBMITTER_ENV`, and that with ``value``.

        Args:
            value: What the environment should say, or None for unset.
        """

        def _env(key: str) -> str | None:
            return value if key == SUBMITTER_ENV else None

        config_test_hooks.get_env = _env

    def _restore(self) -> None:
        """Put the production environment reader back."""
        config_test_hooks.get_env = config_test_hooks._default_get_env

    def test_a_declared_label_is_read_back_verbatim(self) -> None:
        self._pin("fable-brain-audit-0903")
        try:
            assert submitter_label() == "fable-brain-audit-0903"
        finally:
            self._restore()

    def test_no_declaration_reads_as_the_positive_empty_string(self) -> None:
        """The ledger's "declared none", never a decode-time default."""
        self._pin(None)
        try:
            assert submitter_label() == ""
        finally:
            self._restore()

    def test_a_whitespace_declaration_names_nobody(self) -> None:
        """An export of spaces cannot become a label nothing can mention."""
        self._pin("   ")
        try:
            assert submitter_label() == ""
        finally:
            self._restore()


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
