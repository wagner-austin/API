"""Each source repository's default-branch history, read and published (MCPs board task 8b3dcb31).

The reads run real git against real repositories used as remotes, so what is
asserted is the history git itself lists: a merge's second parent left out,
each commit's generation counted from its parents, a first-parent chain
re-routed through a merge's second parent (MCPs board task ecd088e0) and
at most ``MAX_COMMITS`` sent; whether main descends from a held head is
``test_main_ancestry.py``'s. The queue is the dispatch-queue fake that
speaks the endpoint's own wire shape.
"""

from __future__ import annotations

import logging
import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, dump_json_str

from fleet.cli import hub_passes
from fleet.core import _test_hooks, declared, export, main_history, mirror_fetch, queue
from tests._git_fixtures import git
from tests._main_history_repos import (
    IDENTITY,
    PINNED,
    READ_AT,
    merged_repository,
    messages,
    mirror_of,
    registry,
    rerouted_repository,
    source,
)
from tests._queue_fakes import FakeQueue, queue_env
from tests.conftest import FakeRun, failed, ok, pin_clock


class TestReadMain:
    def test_reads_the_default_branch_first_parent_history_and_the_head_depth(
        self, tmp_path: pathlib.Path
    ) -> None:
        # The merge's parents, second and side, are each of generation 2, so
        # it is of 3 (test claim upheld on MCPs board task ecd088e0).
        remote = tmp_path / "remote" / "MCPs"
        commits = merged_repository(remote)

        read = main_history.read_main(tmp_path / "mirrors", "MCPs", str(remote))

        assert read == main_history.MainRead(
            repository="MCPs",
            mirror=mirror_of(tmp_path / "mirrors", remote),
            head=commits["merge"],
            commits=(
                main_history.MainCommit(sha=commits["merge"], generation=3),
                main_history.MainCommit(sha=commits["second"], generation=2),
                main_history.MainCommit(sha=commits["first"], generation=1),
            ),
        )
        assert commits["side"] not in [listed["sha"] for listed in read["commits"]]

    def test_ranks_a_chain_re_routed_through_a_merge_by_generation_above_the_old_main(
        self, tmp_path: pathlib.Path
    ) -> None:
        remote = tmp_path / "remote" / "osrs-lab"
        commits = rerouted_repository(remote)

        read = main_history.read_main(tmp_path / "mirrors", "osrs-lab", str(remote))

        # The branch's b2 shares third's first-parent depth, 3; the merge
        # contains third and outranks it, 5 against 3.
        assert read["commits"] == (
            main_history.MainCommit(sha=commits["merge"], generation=5),
            main_history.MainCommit(sha=commits["b3"], generation=4),
            main_history.MainCommit(sha=commits["b2"], generation=3),
            main_history.MainCommit(sha=commits["b1"], generation=2),
            main_history.MainCommit(sha=commits["first"], generation=1),
        )
        assert main_history.descends(read["mirror"], commits["third"], read["head"]) is True

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

        # A linear history's head is of generation 1001, its count from the root.
        assert len(read["commits"]) == 1000
        assert read["commits"][0] == main_history.MainCommit(
            sha=git(repo, "rev-parse", "main"), generation=1001
        )
        assert read["commits"][-1] == main_history.MainCommit(
            sha=git(repo, "rev-parse", "main~999"), generation=2
        )

    @pytest.mark.parametrize("returncode", [128, -9])
    def test_refuses_a_history_git_cannot_list(
        self, tmp_path: pathlib.Path, returncode: int
    ) -> None:
        """A git that exits non-zero, or is killed by a signal, lists nothing."""
        remote = "https://github.com/wagner-austin/MCPs.git"
        mirror = export.mirror_path(tmp_path, export.remote_mirror_key(remote))
        mirror.mkdir()
        head = "a" * 40
        stderr = f"fatal: bad object {head}"
        _test_hooks.run = FakeRun([ok(""), ok(head), failed(returncode, stderr)])

        with pytest.raises(AppError) as raised:
            main_history.read_main(tmp_path, "MCPs", remote)

        assert raised.value.code is FleetErrorCode.EXPORT_FAILED
        assert raised.value.message.startswith(
            f"{main_history.HISTORY_UNREADABLE_CODE}: "
            f"git rev-list --topo-order --reverse --parents {head} in {mirror}: "
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
        assert reads[0]["commits"][0]["sha"] == commits["merge"]
        assert len(unread) == 1
        assert unread[0].startswith(
            f"gone: EXPORT_FAILED: git fetch {missing} {export.DEFAULT_BRANCH_REF} into "
        )

    def test_fetches_nothing_more_from_a_host_one_fetch_could_not_resolve(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # RFC 6761 reserves .invalid, so real git meets a real resolver that
        # answers no address, as lacrymal's did for github.com on 2026-11-14.
        pin_clock(PINNED)
        caplog.set_level(logging.INFO)
        remote = tmp_path / "remote" / "MCPs"
        commits = merged_repository(remote)
        first = "https://one.invalid/wagner-austin/API.git"
        second = "https://one.invalid/wagner-austin/LSTM.git"
        elsewhere = "https://two.invalid/wagner-austin/wiki.git"

        reads, unread = main_history.read_mains(
            tmp_path / "mirrors",
            {"API": first, "LSTM": second, "MCPs": str(remote), "wiki": elsewhere},
        )

        assert [read["repository"] for read in reads] == ["MCPs"]
        assert reads[0]["commits"][0]["sha"] == commits["merge"]
        assert len(unread) == 3
        assert unread[0].startswith(f"API: EXPORT_FAILED: git fetch {first} ")
        assert "Could not resolve host: one.invalid" in unread[0]
        assert unread[1] == (
            f"LSTM: {main_history.HOST_UNRESOLVED_CODE}: not fetched from {second}: "
            "this tick's fetch for API could not resolve one.invalid"
        )
        assert unread[2].startswith(f"wiki: EXPORT_FAILED: git fetch {elsewhere} ")
        assert "Could not resolve host: two.invalid" in unread[2]
        said = messages(caplog)
        assert sum(first in line for line in said) == len(mirror_fetch.FETCH_PAUSES_SECONDS)
        assert not any(second in line for line in said)
        assert sum(elsewhere in line for line in said) == len(mirror_fetch.FETCH_PAUSES_SECONDS)


class TestUnresolvedHost:
    def test_names_the_host_git_could_not_resolve_lowercased(self) -> None:
        assert (
            mirror_fetch.unresolved_host(
                "git fetch https://GitHub.com/a.git HEAD into m: attempt 1: fatal: unable to "
                "access 'https://GitHub.com/a.git/': Could not resolve host: GitHub.com; "
                "attempt 2: Could not resolve host: other.invalid"
            )
            == "github.com"
        )

    def test_names_the_host_where_the_refusal_ends(self) -> None:
        assert mirror_fetch.unresolved_host("Could not resolve host: one.invalid") == "one.invalid"

    @pytest.mark.parametrize(
        "refusal",
        [
            pytest.param("fatal: Connection reset by peer", id="another-refusal"),
            pytest.param("fatal: Could not resolve host: ", id="no-host-named"),
            pytest.param("fatal: Could not resolve host: ;attempt 2", id="empty-before-semicolon"),
        ],
    )
    def test_names_none_when_git_named_no_host(self, refusal: str) -> None:
        assert mirror_fetch.unresolved_host(refusal) is None


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
        repository="MCPs",
        commits=(
            main_history.MainCommit(sha="b" * 40, generation=3),
            main_history.MainCommit(sha="a" * 40, generation=2),
        ),
        since="a" * 40,
    )

    def test_sends_each_read_and_answers_each_head(self) -> None:
        _test_hooks.env = queue_env()
        head: JSONObject = {"repository": "MCPs", "head": "b" * 40, "generation": 3, "added": 1}
        refused: JSONObject = {
            "repository": "API",
            "code": "DISPATCH_MAIN_HISTORY_STALE",
            "message": "DISPATCH_MAIN_HISTORY_STALE: API's main read at ...",
        }
        endpoint = FakeQueue([dump_json_str({"heads": [head], "refused": [refused]})])
        _test_hooks.http_post = endpoint

        published = main_history.publish_mains(
            queue.load_credentials(), [self.READ], read_at=READ_AT, identity=IDENTITY
        )

        assert published == main_history.MainHistoryPublication(
            heads=(
                main_history.MainHistoryHead(
                    repository="MCPs", head="b" * 40, generation=3, added=1
                ),
            ),
            refused=(
                main_history.MainHistoryRefused(
                    repository="API",
                    code="DISPATCH_MAIN_HISTORY_STALE",
                    message="DISPATCH_MAIN_HISTORY_STALE: API's main read at ...",
                ),
            ),
        )
        sent: JSONObject = {
            "repositories": [
                {
                    "repository": "MCPs",
                    "commits": [
                        {"sha": "b" * 40, "generation": 3},
                        {"sha": "a" * 40, "generation": 2},
                    ],
                    "since": "a" * 40,
                }
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
            ({"heads": 3, "refused": []}, "must answer a heads list"),
            ({"heads": [1], "refused": []}, "heads must be objects, got int"),
            (
                {
                    "heads": [{"repository": "MCPs", "head": "b", "generation": "3", "added": 2}],
                    "refused": [],
                },
                "generation",
            ),
            ({"heads": []}, "must answer a refused list"),
            ({"heads": [], "refused": [1]}, "refused must be objects, got int"),
            ({"heads": [], "refused": [{"repository": "MCPs", "code": 1, "message": "m"}]}, "code"),
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
        held: JSONObject = {
            "repository": "MCPs",
            "head": commits["second"],
            "generation": 2,
            "readAt": READ_AT,
        }
        head: JSONObject = {
            "repository": "MCPs",
            "head": commits["merge"],
            "generation": 3,
            "added": 1,
        }
        endpoint = FakeQueue(
            [dump_json_str({"heads": [held]}), dump_json_str({"heads": [head], "refused": []})]
        )
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.INFO):
            hub_passes.main_pass(
                queue.load_credentials(), IDENTITY, read, mirrors_root=tmp_path / "mirrors"
            )

        assert endpoint.tools == ["dispatch_main_heads", "dispatch_main_publish"]
        assert endpoint.arguments[1]["readAt"] == READ_AT
        assert endpoint.arguments[1]["repositories"] == [
            {
                "repository": "MCPs",
                "commits": [
                    {"sha": commits["merge"], "generation": 3},
                    {"sha": commits["second"], "generation": 2},
                    {"sha": commits["first"], "generation": 1},
                ],
                "since": commits["second"],
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
            "(generation 3, 1 new)"
        ) in logged

    def test_logs_each_repository_the_queue_refused_and_publishes_the_rest(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pin_clock(PINNED)
        merged = merged_repository(tmp_path / "remote" / "MCPs")
        rerouted = rerouted_repository(tmp_path / "remote" / "osrs-lab")
        read = registry(
            source(str(tmp_path / "remote" / "MCPs"), "MCPs"),
            source(str(tmp_path / "remote" / "osrs-lab"), "osrs-lab"),
        )
        _test_hooks.env = queue_env()
        held: JSONObject = {
            "heads": [
                {"repository": "MCPs", "head": "f" * 40, "generation": 9, "readAt": READ_AT},
                {
                    "repository": "osrs-lab",
                    "head": rerouted["third"],
                    "generation": 3,
                    "readAt": READ_AT,
                },
            ]
        }
        message = (
            "DISPATCH_MAIN_HISTORY_REWRITTEN: MCPs's main as read does not descend from the "
            f"head the queue holds, {'f' * 40} at generation 9, so main was rewritten; nothing "
            "was written for it"
        )
        answer: JSONObject = {
            "heads": [
                {"repository": "osrs-lab", "head": rerouted["merge"], "generation": 5, "added": 4}
            ],
            "refused": [
                {
                    "repository": "MCPs",
                    "code": "DISPATCH_MAIN_HISTORY_REWRITTEN",
                    "message": message,
                }
            ],
        }
        endpoint = FakeQueue([dump_json_str(held), dump_json_str(answer)])
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.INFO):
            hub_passes.main_pass(
                queue.load_credentials(), IDENTITY, read, mirrors_root=tmp_path / "mirrors"
            )

        # MCPs's mirror never held the queue's head, so it is sent with no since.
        assert endpoint.arguments[1]["repositories"] == [
            {
                "repository": "MCPs",
                "commits": [
                    {"sha": merged["merge"], "generation": 3},
                    {"sha": merged["second"], "generation": 2},
                    {"sha": merged["first"], "generation": 1},
                ],
                "since": None,
            },
            {
                "repository": "osrs-lab",
                "commits": [
                    {"sha": rerouted["merge"], "generation": 5},
                    {"sha": rerouted["b3"], "generation": 4},
                    {"sha": rerouted["b2"], "generation": 3},
                    {"sha": rerouted["b1"], "generation": 2},
                    {"sha": rerouted["first"], "generation": 1},
                ],
                "since": rerouted["third"],
            },
        ]
        logged = messages(caplog)
        assert f"the queue wrote no main history for {message}" in logged
        assert (
            f"published the main history of 1 repositories: osrs-lab at {rerouted['merge'][:12]} "
            "(generation 5, 4 new)"
        ) in logged

    def test_publishes_nothing_when_no_read_could_be_proved(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pin_clock(PINNED)
        remote = tmp_path / "remote" / "MCPs"
        merged_repository(remote)
        blob = git(remote, "rev-parse", "main:a.txt")
        read = registry(source(str(remote), "MCPs"))
        _test_hooks.env = queue_env()
        held: JSONObject = {
            "heads": [{"repository": "MCPs", "head": blob, "generation": 1, "readAt": READ_AT}]
        }
        endpoint = FakeQueue([dump_json_str(held)])
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.WARNING):
            hub_passes.main_pass(
                queue.load_credentials(), IDENTITY, read, mirrors_root=tmp_path / "mirrors"
            )

        assert endpoint.tools == ["dispatch_main_heads"]
        logged = messages(caplog)
        unproved = (
            "read no main history for MCPs: EXPORT_FAILED: "
            f"{main_history.ANCESTRY_UNREADABLE_CODE}: "
        )
        assert any(line.startswith(unproved) for line in logged)
        assert "published no main history: none of 1 repositories was proved" in logged

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
