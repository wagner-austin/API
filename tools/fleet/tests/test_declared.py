"""The rolled registry's declared projects, and the refusal they drive.

MCPs board task 1915b042. The read runs real git over a repository whose
``refs/fleet/rolled`` names a commit, with the working tree dirtied after it,
so a pass shows the keys come from the commit the ref names and not from
files on disk. The queue speaks the real JSON-RPC-over-SSE shape through
:class:`tests._queue_fakes.FakeQueue`.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str

from fleet.cli import agent
from fleet.core import _test_hooks, declared, queue, rolled
from tests._git_fixtures import committed_repository
from tests._hub_tick import UNROLLED, unrolled_call
from tests._queue_fakes import FakeQueue, queue_env
from tests.conftest import FakeRun, agent_argv, failed, ok, pin_clock

#: 2026-10-14T23:00:00Z, the instant the tests pin the clock at.
NOW = 1_792_018_800

#: The registry a rolled commit declares in these cases.
REGISTRY_FILES = {
    "tools/fleet/registry/nodes.json": "{}\n",
    "tools/fleet/registry/projects/MCPs.json": "{}\n",
    "tools/fleet/registry/projects/MCPs/packages/wiki-check.json": "{}\n",
    "tools/fleet/registry/projects/tools/fleet.json": "{}\n",
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
    unrolled.write_text("{}\n", encoding="utf-8")
    return repo, commit


class TestReadDeclared:
    def test_lists_the_keys_the_rolled_commit_declares_and_the_instant_before(
        self, rolled_repo: tuple[pathlib.Path, str]
    ) -> None:
        repo, commit = rolled_repo
        pin_clock(NOW)

        assert declared.read_declared(repo) == declared.RegistryRead(
            commit=commit,
            declared=("MCPs", "MCPs/packages/wiki-check", "tools/fleet"),
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
        commit = "a" * 40
        _test_hooks.run = FakeRun([ok(commit), failed(128, "fatal: not a tree object")])

        assert declared.read_declared(tmp_path) == (
            f"{declared.LIST_FAILED_CODE}: git ls-tree {commit} exited 128: "
            "fatal: not a tree object"
        )

    def test_a_commit_declaring_no_project_sends_nothing(self, tmp_path: pathlib.Path) -> None:
        repo = tmp_path / "api"
        commit = committed_repository(
            repo, {"tools/fleet/registry/nodes.json": "{}\n"}, rolled.ROLLED_REF
        )

        assert declared.read_declared(repo) == (
            f"{declared.EMPTY_CODE}: {declared.PROJECTS_PATH} at {commit} declares no project; "
            "nothing was refused"
        )


class TestRefuseUndeclared:
    READ = declared.RegistryRead(
        commit="b" * 40, declared=("MCPs", "tools/fleet"), checked_at="2026-10-14T23:00:00+00:00"
    )
    IDENTITY: Final[JSONObject] = {
        "agent": "fleet-runner-austinpc",
        "sessionId": "s",
        "cwd": "C:/fleet",
    }

    def test_sends_the_read_and_answers_the_count(self) -> None:
        _test_hooks.env = queue_env()
        endpoint = FakeQueue([dump_json_str({"refused": 2, "commit": "b" * 40})])
        _test_hooks.http_post = endpoint

        refused = declared.refuse_undeclared(
            queue.load_credentials(), self.READ, identity=self.IDENTITY
        )

        assert refused == 2
        assert endpoint.tools == ["dispatch_refuse_undeclared"]
        assert endpoint.arguments == [
            {
                "declared": ["MCPs", "tools/fleet"],
                "commit": "b" * 40,
                "checkedAt": "2026-10-14T23:00:00+00:00",
                **self.IDENTITY,
            }
        ]

    def test_an_answer_that_is_not_an_object_is_refused(self) -> None:
        _test_hooks.env = queue_env()
        _test_hooks.http_post = FakeQueue([dump_json_str([1])])

        with pytest.raises(JSONTypeError, match="must answer an object, got list"):
            declared.refuse_undeclared(queue.load_credentials(), self.READ, identity=self.IDENTITY)


class TestTheHubTickRefuses:
    def test_a_tick_hands_the_rolled_registry_to_the_queue_after_its_claim(
        self, config_path: pathlib.Path, rolled_repo: tuple[pathlib.Path, str]
    ) -> None:
        repo, commit = rolled_repo
        _test_hooks.env = queue_env()
        endpoint = FakeQueue(
            [
                dump_json_str({"claimed": None}),
                dump_json_str({"refused": 1, "commit": commit}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert endpoint.tools == ["dispatch_claim", "dispatch_refuse_undeclared"]
        assert endpoint.arguments[1]["declared"] == [
            "MCPs",
            "MCPs/packages/wiki-check",
            "tools/fleet",
        ]
        assert endpoint.arguments[1]["commit"] == commit
