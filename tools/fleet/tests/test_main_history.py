"""Each source repository's default-branch history, read and published (MCPs board task 8b3dcb31).

The reads run real git against real repositories used as remotes, so what is
asserted is the history git itself lists: a merge's second parent left out,
the head's depth counted from the root, at most ``MAX_COMMITS`` sent. The
queue is the dispatch-queue fake that speaks the endpoint's own wire shape.
"""

from __future__ import annotations

import logging
import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, dump_json_str

from fleet.cli import hub_passes
from fleet.core import _test_hooks, declared, export, main_history, queue
from tests._git_fixtures import git
from tests._queue_fakes import FakeQueue, queue_env
from tests.conftest import FakeRun, failed, ok, pin_clock

#: The instant every publication here is read at.
READ_AT = "2026-11-13T22:00:00+00:00"

#: This runner's identity arguments.
IDENTITY: JSONObject = {
    "agent": "fleet-hub-lacrymal",
    "sessionId": "s",
    "cwd": "/home/corvis/fleet",
}

#: Any instant: the clock is pinned so a lost fetch's pauses pass at once.
PINNED = 1_794_000_000


def commit(repo: pathlib.Path, name: str, body: str, message: str) -> str:
    """Write one file, commit it and answer the commit.

    Args:
        repo: The repository.
        name: The file.
        body: Its contents.
        message: The commit's message.

    Returns:
        The commit's id.
    """
    (repo / name).write_text(body, encoding="utf-8")
    git(repo, "add", name)
    git(repo, "commit", "--quiet", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def merged_repository(repo: pathlib.Path) -> dict[str, str]:
    """A repository whose main merged a side branch: first, second, then the merge.

    Args:
        repo: Where to build it.

    Returns:
        Each commit by name: ``first``, ``side``, ``second`` and ``merge``.
    """
    repo.mkdir(parents=True)
    git(repo, "init", "--quiet", "--initial-branch=main")
    first = commit(repo, "a.txt", "1", "first")
    git(repo, "checkout", "--quiet", "-b", "side")
    side = commit(repo, "b.txt", "side", "side")
    git(repo, "checkout", "--quiet", "main")
    second = commit(repo, "a.txt", "2", "second")
    git(repo, "merge", "--quiet", "--no-ff", "-m", "merge side", "side")
    merge = git(repo, "rev-parse", "HEAD")
    return {"first": first, "side": side, "second": second, "merge": merge}


def source(remote: str, repository: str) -> declared.DeclaredProject:
    """A declared project sourced from a remote.

    Args:
        remote: The remote.
        repository: The repository its remote names.

    Returns:
        The project.
    """
    return declared.DeclaredProject(
        project=f"{repository}/pkg",
        required_tags=(),
        dispatch_nodes=(),
        source=declared.DeclaredSource(repository=repository, directory="pkg", remote=remote),
    )


def registry(*projects: declared.DeclaredProject) -> declared.RegistryRead:
    """A read of the rolled registry declaring these projects.

    Args:
        projects: The projects.

    Returns:
        The read, at :data:`READ_AT`.
    """
    return declared.RegistryRead(
        commit="c" * 40, projects=projects, runners=("fleet-node-sedona",), checked_at=READ_AT
    )


def messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every line logged.

    Args:
        caplog: The capture.

    Returns:
        Each record's message.
    """
    return [record.getMessage() for record in caplog.records]


class TestReadMain:
    def test_reads_the_default_branch_first_parent_history_and_the_head_depth(
        self, tmp_path: pathlib.Path
    ) -> None:
        remote = tmp_path / "remote" / "MCPs"
        commits = merged_repository(remote)

        read = main_history.read_main(tmp_path / "mirrors", "MCPs", str(remote))

        assert read == main_history.MainHistoryRead(
            repository="MCPs",
            head_depth=3,
            commits=(commits["merge"], commits["second"], commits["first"]),
        )
        assert commits["side"] not in read["commits"]

    def test_sends_at_most_the_newest_thousand_commits_counting_the_head_from_the_root(
        self, tmp_path: pathlib.Path
    ) -> None:
        repo = tmp_path / "remote" / "long"
        repo.mkdir(parents=True)
        git(repo, "init", "--quiet", "--initial-branch=main")
        stream = "".join(
            "commit refs/heads/main\n"
            f"committer t <t@example.invalid> {1_700_000_000 + index} +0000\n"
            "data 2\nc\n\n"
            for index in range(main_history.MAX_COMMITS + 1)
        )
        imported = _test_hooks.run(
            ("git", "-C", str(repo), "fast-import", "--quiet"),
            timeout_seconds=60,
            stdin_bytes=stream.encode("utf-8"),
        )
        assert imported["returncode"] == 0, imported["stderr"]

        read = main_history.read_main(tmp_path / "mirrors", "long", str(repo))

        assert read["head_depth"] == 1001
        assert len(read["commits"]) == 1000
        assert read["commits"][0] == git(repo, "rev-parse", "main")
        assert read["commits"][-1] == git(repo, "rev-parse", "main~999")

    def test_refuses_a_history_git_cannot_list(self, tmp_path: pathlib.Path) -> None:
        remote = "https://github.com/wagner-austin/MCPs.git"
        mirror = export.mirror_path(tmp_path, export.remote_mirror_key(remote))
        mirror.mkdir()
        head = "a" * 40
        _test_hooks.run = FakeRun([ok(""), ok(head), failed(128, f"fatal: bad object {head}")])

        with pytest.raises(AppError) as raised:
            main_history.read_main(tmp_path, "MCPs", remote)

        assert raised.value.code is FleetErrorCode.EXPORT_FAILED
        assert raised.value.message.startswith(
            f"{main_history.HISTORY_UNREADABLE_CODE}: "
            f"git rev-list --first-parent --count {head} in {mirror}: "
        )
        assert f"fatal: bad object {head}" in raised.value.message


class TestReadMains:
    def test_reads_every_repository_it_can_and_names_each_it_cannot(
        self, tmp_path: pathlib.Path
    ) -> None:
        pin_clock(PINNED)
        remote = tmp_path / "remote" / "MCPs"
        commits = merged_repository(remote)
        missing = str(tmp_path / "remote" / "gone")

        reads, unread = main_history.read_mains(
            tmp_path / "mirrors", {"gone": missing, "MCPs": str(remote)}
        )

        assert [read["repository"] for read in reads] == ["MCPs"]
        assert reads[0]["commits"][0] == commits["merge"]
        assert len(unread) == 1
        assert unread[0].startswith(
            f"gone: EXPORT_FAILED: git fetch {missing} {main_history.DEFAULT_BRANCH_REF} into "
        )


class TestSourceRemotes:
    def test_names_each_repository_once_and_leaves_out_one_whose_projects_disagree(
        self,
    ) -> None:
        mcps = "https://github.com/wagner-austin/MCPs.git"
        slime_https = "https://github.com/wagner-austin/slime.git"
        slime_ssh = "ssh://git@github.com/wagner-austin/slime.git"
        projects = (
            source(mcps, "MCPs"),
            source(mcps, "MCPs"),
            source(slime_https, "slime"),
            source(slime_ssh, "slime"),
            declared.DeclaredProject(
                project="loose", required_tags=(), dispatch_nodes=(), source=None
            ),
        )

        remotes, disagreeing = main_history.source_remotes(projects)

        assert remotes == {"MCPs": mcps}
        assert disagreeing == (
            f"FLEET_MAIN_REMOTES_DISAGREE: slime's projects name {slime_https}, {slime_ssh}; "
            "its main was not read",
        )


class TestPublishMains:
    READ = main_history.MainHistoryRead(
        repository="MCPs", head_depth=3, commits=("b" * 40, "a" * 40)
    )

    def test_sends_each_read_and_answers_each_head(self) -> None:
        _test_hooks.env = queue_env()
        head: JSONObject = {"repository": "MCPs", "head": "b" * 40, "depth": 3, "added": 2}
        endpoint = FakeQueue([dump_json_str({"heads": [head]})])
        _test_hooks.http_post = endpoint

        heads = main_history.publish_mains(
            queue.load_credentials(), [self.READ], read_at=READ_AT, identity=IDENTITY
        )

        assert heads == (
            main_history.MainHistoryHead(repository="MCPs", head="b" * 40, depth=3, added=2),
        )
        sent: JSONObject = {
            "repositories": [
                {"repository": "MCPs", "headDepth": 3, "commits": ["b" * 40, "a" * 40]}
            ],
            "readAt": READ_AT,
            **IDENTITY,
        }
        assert endpoint.tools == ["dispatch_main_publish"]
        assert endpoint.arguments == [sent]

    @pytest.mark.parametrize(
        ("answer", "words"),
        [
            ([1], "must answer an object, got list"),
            ({"heads": 3}, "must answer a heads list"),
            ({"heads": [1]}, "heads must be objects, got int"),
            ({"heads": [{"repository": "MCPs", "head": "b", "depth": "3", "added": 2}]}, "depth"),
        ],
    )
    def test_refuses_an_answer_it_cannot_read(self, answer: JSONValue, words: str) -> None:
        _test_hooks.env = queue_env()
        _test_hooks.http_post = FakeQueue([dump_json_str(answer)])

        with pytest.raises(JSONTypeError, match=words):
            main_history.publish_mains(
                queue.load_credentials(), [self.READ], read_at=READ_AT, identity=IDENTITY
            )


class TestMainPass:
    def test_reads_no_main_when_the_registry_was_not_read(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _test_hooks.env = queue_env()
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.WARNING):
            hub_passes.main_pass(
                queue.load_credentials(), IDENTITY, "FLEET_REGISTRY_EMPTY", mirrors_root=tmp_path
            )

        assert endpoint.tools == []
        assert "read no main history: the rolled registry was not read" in messages(caplog)

    def test_publishes_every_repository_read_at_the_registry_read_and_logs_each_left_out(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pin_clock(PINNED)
        commits = merged_repository(tmp_path / "remote" / "MCPs")
        good = str(tmp_path / "remote" / "MCPs")
        missing = str(tmp_path / "remote" / "gone")
        read = registry(
            source(good, "MCPs"),
            source(missing, "gone"),
            source(good, "twin"),
            source(missing, "twin"),
        )
        _test_hooks.env = queue_env()
        head: JSONObject = {"repository": "MCPs", "head": commits["merge"], "depth": 3, "added": 3}
        endpoint = FakeQueue([dump_json_str({"heads": [head]})])
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.INFO):
            hub_passes.main_pass(
                queue.load_credentials(), IDENTITY, read, mirrors_root=tmp_path / "mirrors"
            )

        assert endpoint.arguments[0]["readAt"] == READ_AT
        assert endpoint.arguments[0]["repositories"] == [
            {
                "repository": "MCPs",
                "headDepth": 3,
                "commits": [commits["merge"], commits["second"], commits["first"]],
            }
        ]
        logged = messages(caplog)
        assert (
            f"FLEET_MAIN_REMOTES_DISAGREE: twin's projects name {good}, {missing}; "
            "its main was not read"
        ) in logged
        left_out = "read no main history for gone: EXPORT_FAILED: "
        assert any(line.startswith(left_out) for line in logged)
        assert (
            f"published the main history of 1 repositories: MCPs at {commits['merge'][:12]} "
            "(depth 3, 3 new)"
        ) in logged

    def test_publishes_nothing_when_no_repository_could_be_read(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pin_clock(PINNED)
        read = registry(source(str(tmp_path / "gone"), "gone"))
        _test_hooks.env = queue_env()
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.WARNING):
            hub_passes.main_pass(
                queue.load_credentials(), IDENTITY, read, mirrors_root=tmp_path / "mirrors"
            )

        assert endpoint.tools == []
        assert "published no main history: none of 1 repositories was read" in messages(caplog)
