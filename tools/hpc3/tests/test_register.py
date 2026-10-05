"""The registration steps: what is refused, what is read, what is written.

Driven against a copy of the real registry and index (``tests._registry_tree``)
so every check here runs beside every project already registered rather than
beside a fixture's invention. No count of them is written here: a test that
pinned one went red the day a real project was registered, which is the
surprise this command exists to remove.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue, dump_json_str

from hpc3.contracts.budget import Budget
from hpc3.contracts.project import ProjectConfig, encode_project_config
from hpc3.core import register
from hpc3.core.index_sections import projects_without_section
from hpc3.core.registry import declared_projects, shared_connection
from hpc3.core.research_index import extract_projects_block, render_projects_block
from tests._registry_tree import (
    NEWCOMER,
    NEWCOMER_DIGEST,
    NEWCOMER_IMAGE,
    copy_tree,
    index_of,
    repo_of,
    runs_of,
    write_section,
)
from tests.conftest import FakeRun, project_config, workspace_document, write_workspace


def _config(**overrides: str) -> ProjectConfig:
    """Build the newcomer's declaration as ``hpc3-register`` would.

    Args:
        **overrides: ``partition`` to replace.

    Returns:
        A CPU-only, imaged declaration with one pin.
    """
    return ProjectConfig(
        partition=overrides.get("partition", "free"),
        gpu=None,
        cpus=2,
        mem_gb=4,
        minutes=60,
        requeue=True,
        resumes_from_checkpoint=False,
        image={"path": NEWCOMER_IMAGE, "sha256": NEWCOMER_DIGEST, "binds": ["/pub/wagnera3"]},
        env_path="/opt/env",
        pinned_packages={"newcomer": "0.1.0"},
        deterministic=True,
        certified_inputs=False,
        budget=Budget(self_imposed_gpu_hours=0.0, max_service_units=0.0, charge_account=""),
        repo="../../../clients/Newcomer",
    )


class TestThePreconditions:
    """Every local reason, collected, so none is met as a later failure."""

    def test_a_ready_project_has_none(self, tmp_path: pathlib.Path) -> None:
        """Section written, repo present, name free.

        Args:
            tmp_path: Root of the copied tree.
        """
        root = copy_tree(tmp_path)
        write_section(root, NEWCOMER)

        problems = register.registration_problems(
            runs=runs_of(root), index=index_of(root), project=NEWCOMER, repo=repo_of(root)
        )

        assert problems == []

    def test_all_four_are_named_at_once(self, tmp_path: pathlib.Path) -> None:
        """Declared already, document present, no section, no repo.

        Args:
            tmp_path: Root of the copied tree.
        """
        root = copy_tree(tmp_path)
        _ = write_workspace(
            runs_of(root) / "hpc3-newcomer.json",
            workspace_document(root="/pub/wagnera3", projects={NEWCOMER: project_config()}),
        )
        absent = root / "nowhere"

        with pytest.raises(AppError) as refused:
            register.require_registrable(
                runs=runs_of(root), index=index_of(root), project=NEWCOMER, repo=absent
            )

        assert refused.value.code is Hpc3ErrorCode.REGISTRATION_INCOMPLETE
        message = refused.value.message
        assert message.startswith("cannot register 'newcomer', 4 precondition(s) unmet: ")
        assert "project 'newcomer' is already declared by a workspace in" in message
        assert f"{runs_of(root) / 'hpc3-newcomer.json'} already exists" in message
        assert "has no section headed '### `newcomer`' under '## Registered with" in message
        assert f"--repo {absent} is not a directory" in message


class TestReadingTheImage:
    """The digest is the file's own answer, read on the cluster."""

    def test_the_digest_sha256sum_prints_is_the_one_declared(self, fake_run: FakeRun) -> None:
        """Args:
        fake_run: The scripted cluster.
        """
        fake_run.add("sha256sum", stdout=f"{NEWCOMER_DIGEST}  {NEWCOMER_IMAGE}\n")

        image = register.read_image("hpc3", NEWCOMER_IMAGE, binds=["/pub/wagnera3"])

        assert image == {
            "path": NEWCOMER_IMAGE,
            "sha256": NEWCOMER_DIGEST,
            "binds": ["/pub/wagnera3"],
        }
        assert fake_run.commands() == [f"sha256sum '{NEWCOMER_IMAGE}'"]

    def test_an_image_not_yet_built_is_refused(self, fake_run: FakeRun) -> None:
        """Registration cannot run before the build, which is the order it protects.

        Args:
            fake_run: The scripted cluster.
        """
        fake_run.add("sha256sum", stderr="No such file or directory", returncode=1)

        with pytest.raises(AppError) as refused:
            _ = register.read_image("hpc3", NEWCOMER_IMAGE, binds=["/pub/wagnera3"])

        assert refused.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED


class TestWritingTheDocument:
    """The document as committed, and the table as the registry says."""

    def test_the_repo_is_written_relative_to_the_directory(self, tmp_path: pathlib.Path) -> None:
        """Args:
        tmp_path: Root of the copied tree.
        """
        root = copy_tree(tmp_path)

        assert register.relative_repo(runs_of(root), repo_of(root)) == "../../../clients/Newcomer"

    def test_the_payload_copies_the_connection_and_the_default_threshold(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Args:
        tmp_path: Root of the copied tree.
        """
        root = copy_tree(tmp_path)
        connection = shared_connection(runs_of(root))

        payload = register.workspace_payload(connection, NEWCOMER, _config())

        assert {k: v for k, v in payload.items() if k != "projects"} == {
            "cluster": "hpc3",
            "host": "hpc3",
            "root": "/pub/wagnera3",
            "ledger": "ledger.jsonl",
            "quiet_seconds": 1800,
        }
        assert payload["projects"] == {NEWCOMER: encode_project_config(_config())}

    def test_the_table_is_regenerated_from_the_documents(self, tmp_path: pathlib.Path) -> None:
        """Every gate the registry carries holds on the tree it leaves.

        Args:
            tmp_path: Root of the copied tree.
        """
        root = copy_tree(tmp_path)
        write_section(root, NEWCOMER)
        before = set(declared_projects(runs_of(root)))
        payload = register.workspace_payload(shared_connection(runs_of(root)), NEWCOMER, _config())

        registry = register.write_registration(
            runs=runs_of(root), index=index_of(root), project=NEWCOMER, payload=payload
        )

        document = runs_of(root) / "hpc3-newcomer.json"
        assert document.read_bytes() == (dump_json_str(payload, indent=2) + "\n").encode("utf-8")
        assert registry == declared_projects(runs_of(root))
        assert set(registry) == before | {NEWCOMER}
        assert NEWCOMER not in before
        assert registry[NEWCOMER]["image"]["sha256"] == NEWCOMER_DIGEST
        text = index_of(root).read_text(encoding="utf-8")
        assert extract_projects_block(text) == render_projects_block(registry)
        assert projects_without_section(text, registry) == []

    def test_a_declaration_the_contract_refuses_is_never_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Decoded as a workspace first, so the directory never holds it.

        Args:
            tmp_path: Root of the copied tree.
        """
        root = copy_tree(tmp_path)
        before = index_of(root).read_bytes()
        payload: dict[str, JSONValue] = register.workspace_payload(
            shared_connection(runs_of(root)), NEWCOMER, _config(partition="nonesuch")
        )

        with pytest.raises(AppError) as refused:
            _ = register.write_registration(
                runs=runs_of(root), index=index_of(root), project=NEWCOMER, payload=payload
            )

        assert refused.value.code is Hpc3ErrorCode.PARTITION_UNKNOWN
        assert not (runs_of(root) / "hpc3-newcomer.json").exists()
        assert index_of(root).read_bytes() == before
