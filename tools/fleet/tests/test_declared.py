"""The rolled registry's declared projects, and their publication to the queue.

MCPs board tasks 1915b042 and 0be88a36. The read runs real git over a
repository whose ``refs/fleet/rolled`` names a commit, with the working tree
dirtied after it, so a pass shows the projects and their tags come from the
commit the ref names and not from files on disk; one file carries multibyte
UTF-8, so a pass also shows the batch's byte sizes are read as bytes. The
queue speaks the real JSON-RPC-over-SSE shape through
:class:`tests._queue_fakes.FakeQueue`.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, dump_json_str

from fleet.cli import agent
from fleet.contracts.tags import NodeTag
from fleet.core import _test_hooks, declared, queue, rolled
from tests._agent_cli import agent_argv
from tests._git_fixtures import committed_repository
from tests._hub_tick import UNROLLED, unrolled_call
from tests._queue_fakes import FakeQueue, queue_env
from tests.conftest import FakeRun, failed, ok, pin_clock

#: 2026-10-14T23:00:00Z, the instant the tests pin the clock at.
NOW = 1_792_018_800

#: The commit the scripted cases read.
COMMIT: Final = "a" * 40


def project_file(tags: list[JSONValue], *, note: str = "") -> str:
    """A per-project file the project contract decodes.

    Args:
        tags: Its ``required_tags``.
        note: Text for a key the contract does not read, to put multibyte
            UTF-8 in a file without changing what it declares.

    Returns:
        The file's JSON text.
    """
    body: JSONObject = {
        "worker_ram_gb": 0.5,
        "minimum_workers": 1,
        "expected_minutes": 5,
        "required_tags": tags,
        "source": None,
        "note": note,
    }
    return dump_json_str(body) + "\n"


#: The registry a rolled commit declares in these cases.
REGISTRY_FILES = {
    "tools/fleet/registry/nodes.json": "{}\n",
    "tools/fleet/registry/projects/MCPs.json": project_file([]),
    "tools/fleet/registry/projects/MCPs/packages/wiki-check.json": project_file(
        ["cxx", "pdftotext"], note="caf\u00e9 \u2014 r\u00e9sum\u00e9"
    ),
    "tools/fleet/registry/projects/tools/fleet.json": project_file(["linux"]),
    "tools/fleet/registry/projects/README.md": "not a project\n",
}


@pytest.fixture(name="rolled_repo")
def _rolled_repo(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A repository whose rolled ref names a commit, with a project added after it.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The repository and the commit the ref names.
    """
    repo = tmp_path / "api"
    commit = committed_repository(repo, REGISTRY_FILES, rolled.ROLLED_REF)
    unrolled = repo / "tools/fleet/registry/projects/chat.json"
    unrolled.write_text(project_file(["windows"]), encoding="utf-8")
    return repo, commit


def scripted(batch: _test_hooks.CommandResult) -> FakeRun:
    """A runner whose ref and listing answer one project, then ``batch``.

    Args:
        batch: What ``git cat-file --batch`` answers.

    Returns:
        The runner, scripted for the resolve, the listing and the batch.
    """
    return FakeRun([ok(COMMIT), ok("tools/fleet/registry/projects/MCPs.json\n"), batch])


class TestReadDeclared:
    def test_reads_each_project_and_its_tags_at_the_rolled_commit_and_the_instant_before(
        self, rolled_repo: tuple[pathlib.Path, str]
    ) -> None:
        repo, commit = rolled_repo
        pin_clock(NOW)

        assert declared.read_declared(repo) == declared.RegistryRead(
            commit=commit,
            projects=(
                declared.DeclaredProject(project="MCPs", required_tags=()),
                declared.DeclaredProject(
                    project="MCPs/packages/wiki-check",
                    required_tags=(NodeTag.CXX, NodeTag.PDFTOTEXT),
                ),
                declared.DeclaredProject(project="tools/fleet", required_tags=(NodeTag.LINUX,)),
            ),
            checked_at="2026-10-14T23:00:00+00:00",
        )

    def test_a_repository_with_no_rolled_ref_sends_nothing(self, tmp_path: pathlib.Path) -> None:
        run = FakeRun([UNROLLED])
        _test_hooks.run = run

        assert declared.read_declared(tmp_path) == (
            f"{rolled.REF_UNRESOLVED_CODE}: {rolled.ROLLED_REF} in {tmp_path} did not resolve "
            "to a commit (exit 128, stdout '', stderr 'fatal: Needed a single revision'); "
            "nothing ran"
        )
        assert run.calls == [unrolled_call(tmp_path)]

    def test_a_listing_that_fails_is_named_with_its_exit(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([ok(COMMIT), failed(128, "fatal: not a tree object")])

        assert declared.read_declared(tmp_path) == (
            f"{declared.LIST_FAILED_CODE}: git ls-tree {COMMIT} exited 128: "
            "fatal: not a tree object"
        )

    def test_a_commit_declaring_no_project_sends_nothing(self, tmp_path: pathlib.Path) -> None:
        repo = tmp_path / "api"
        commit = committed_repository(
            repo, {"tools/fleet/registry/nodes.json": "{}\n"}, rolled.ROLLED_REF
        )

        assert declared.read_declared(repo) == (
            f"{declared.EMPTY_CODE}: {declared.PROJECTS_PATH} at {commit} declares no project; "
            "nothing was published"
        )

    def test_asks_one_batch_for_every_file_at_the_commit(self, tmp_path: pathlib.Path) -> None:
        blob = project_file([])
        run = scripted(ok(f"{'c' * 40} blob {len(blob.encode('utf-8'))}\n{blob}\n"))
        _test_hooks.run = run
        pin_clock(NOW)

        assert declared.read_declared(tmp_path) == declared.RegistryRead(
            commit=COMMIT,
            projects=(declared.DeclaredProject(project="MCPs", required_tags=()),),
            checked_at="2026-10-14T23:00:00+00:00",
        )
        assert run.calls[2] == ("git", "-C", str(tmp_path), "cat-file", "--batch")
        assert run.stdin[2] == f"{COMMIT}:tools/fleet/registry/projects/MCPs.json\n".encode()

    def test_a_batch_that_fails_is_named_with_its_exit(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = scripted(failed(128, "fatal: bad object"))

        assert declared.read_declared(tmp_path) == (
            f"{declared.READ_FAILED_CODE}: git cat-file --batch {COMMIT} exited 128: "
            "fatal: bad object"
        )

    @pytest.mark.parametrize(
        ("answer", "header"),
        [
            (
                f"{COMMIT}:tools/fleet/registry/projects/MCPs.json missing\n",
                f"{COMMIT}:tools/fleet/registry/projects/MCPs.json missing",
            ),
            ("", ""),
        ],
    )
    def test_a_batch_answer_that_is_not_a_blob_is_named(
        self, tmp_path: pathlib.Path, answer: str, header: str
    ) -> None:
        _test_hooks.run = scripted(ok(answer))

        assert declared.read_declared(tmp_path) == (
            f"{declared.BATCH_UNREADABLE_CODE}: git cat-file --batch answered {header!r} for "
            "tools/fleet/registry/projects/MCPs.json, not a blob header"
        )

    def test_a_project_file_that_does_not_decode_is_raised(self, tmp_path: pathlib.Path) -> None:
        repo = tmp_path / "api"
        committed_repository(
            repo, {"tools/fleet/registry/projects/MCPs.json": "{}\n"}, rolled.ROLLED_REF
        )

        with pytest.raises(JSONTypeError, match="project must declare 'source'"):
            declared.read_declared(repo)


class TestPublishRegistry:
    READ = declared.RegistryRead(
        commit="b" * 40,
        projects=(
            declared.DeclaredProject(project="MCPs", required_tags=()),
            declared.DeclaredProject(
                project="tools/fleet", required_tags=(NodeTag.LINUX, NodeTag.TESTDB)
            ),
        ),
        checked_at="2026-10-14T23:00:00+00:00",
    )
    IDENTITY: Final[JSONObject] = {
        "agent": "fleet-runner-austinpc",
        "sessionId": "s",
        "cwd": "C:/fleet",
    }

    def test_sends_every_project_with_its_tags_and_answers_the_counts(self) -> None:
        _test_hooks.env = queue_env()
        endpoint = FakeQueue(
            [dump_json_str({"published": 2, "removed": 1, "refused": 3, "commit": "b" * 40})]
        )
        _test_hooks.http_post = endpoint

        publication = declared.publish_registry(
            queue.load_credentials(), self.READ, identity=self.IDENTITY
        )

        assert publication == declared.RegistryPublication(published=2, removed=1, refused=3)
        sent: JSONObject = {
            "projects": [
                {"project": "MCPs", "requiredTags": []},
                {"project": "tools/fleet", "requiredTags": ["linux", "testdb"]},
            ],
            "commit": "b" * 40,
            "checkedAt": "2026-10-14T23:00:00+00:00",
            **self.IDENTITY,
        }
        assert endpoint.tools == ["dispatch_registry_publish"]
        assert endpoint.arguments == [sent]

    def test_an_answer_that_is_not_an_object_is_refused(self) -> None:
        _test_hooks.env = queue_env()
        _test_hooks.http_post = FakeQueue([dump_json_str([1])])

        with pytest.raises(JSONTypeError, match="must answer an object, got list"):
            declared.publish_registry(queue.load_credentials(), self.READ, identity=self.IDENTITY)


class TestTheHubTickPublishes:
    def test_a_tick_publishes_the_rolled_registry_after_its_claim(
        self,
        config_path: pathlib.Path,
        rolled_repo: tuple[pathlib.Path, str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        repo, commit = rolled_repo
        _test_hooks.env = queue_env()
        endpoint = FakeQueue(
            [
                dump_json_str({"claimed": None}),
                dump_json_str({"published": 3, "removed": 0, "refused": 1, "commit": commit}),
            ]
        )
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert agent.main(agent_argv(config_path, repo)) == 0

        assert endpoint.tools == ["dispatch_claim", "dispatch_registry_publish"]
        assert endpoint.arguments[1]["projects"] == [
            {"project": "MCPs", "requiredTags": []},
            {"project": "MCPs/packages/wiki-check", "requiredTags": ["cxx", "pdftotext"]},
            {"project": "tools/fleet", "requiredTags": ["linux"]},
        ]
        assert endpoint.arguments[1]["commit"] == commit
        assert (
            f"published 3 project(s) from the registry at {commit} (0 dropped since the last "
            "publication) and refused 1 queued job(s) naming a project it does not declare"
        ) in [record.getMessage() for record in caplog.records]
