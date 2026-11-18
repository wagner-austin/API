"""Whether main descends from the head the queue holds, as git proves it (MCPs board task ecd088e0).

The queue writes a repository's history only against the held head the hub
proved main contains, because a first-parent chain can re-route through a
merge's second parent on a fast forward, as osrs-lab's e6b6a0a8 did, and only
git can say whether one commit contains another. These cases run real git
against real repositories used as remotes; the held heads come from the
dispatch-queue fake that speaks the endpoint's own wire shape.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str

from fleet.core import _test_hooks, main_history, queue
from tests._git_fixtures import git
from tests._main_history_repos import READ_AT, merged_repository, rerouted_repository
from tests._queue_fakes import FakeQueue, queue_env


class TestDescends:
    def test_answers_whether_main_contains_a_held_commit_and_none_the_mirror_lacks(
        self, tmp_path: pathlib.Path
    ) -> None:
        remote = tmp_path / "remote" / "osrs-lab"
        commits = rerouted_repository(remote)
        read = main_history.read_main(tmp_path / "mirrors", "osrs-lab", str(remote))
        mirror = read["mirror"]

        assert main_history.descends(mirror, commits["merge"], commits["merge"]) is True
        assert main_history.descends(mirror, commits["second"], commits["merge"]) is True
        # Present in the mirror, but the newer commit is not contained in the older.
        assert main_history.descends(mirror, commits["merge"], commits["third"]) is False
        # A commit the mirror never fetched is no ancestor of anything it holds.
        assert main_history.descends(mirror, "f" * 40, commits["merge"]) is False

    def test_refuses_an_ancestry_git_cannot_answer(self, tmp_path: pathlib.Path) -> None:
        # A blob is an object the mirror holds but no commit, and merge-base exits 128.
        remote = tmp_path / "remote" / "MCPs"
        merged_repository(remote)
        read = main_history.read_main(tmp_path / "mirrors", "MCPs", str(remote))
        blob = git(remote, "rev-parse", "main:a.txt")

        with pytest.raises(AppError) as raised:
            main_history.descends(read["mirror"], blob, read["head"])

        assert raised.value.code is FleetErrorCode.EXPORT_FAILED
        assert raised.value.message.startswith(
            f"{main_history.ANCESTRY_UNREADABLE_CODE}: "
            f"git merge-base --is-ancestor {blob} {read['head']} in {read['mirror']}: "
        )
        assert "is a blob, not a commit" in raised.value.message


class TestHeldHeads:
    def test_reads_each_held_head_keyed_by_repository(self) -> None:
        _test_hooks.env = queue_env()
        answer = {
            "heads": [
                {"repository": "API", "head": "a" * 40, "generation": 7, "readAt": READ_AT},
                {"repository": "MCPs", "head": "b" * 40, "generation": 3, "readAt": READ_AT},
            ]
        }
        endpoint = FakeQueue([dump_json_str(answer)])
        _test_hooks.http_post = endpoint

        held = main_history.held_heads(queue.load_credentials())

        assert held == {
            "API": main_history.HeldHead(
                repository="API", head="a" * 40, generation=7, read_at=READ_AT
            ),
            "MCPs": main_history.HeldHead(
                repository="MCPs", head="b" * 40, generation=3, read_at=READ_AT
            ),
        }
        assert endpoint.tools == ["dispatch_main_heads"]
        assert endpoint.arguments == [{}]

    @pytest.mark.parametrize(
        ("answer", "words"),
        [
            ([1], "dispatch_main_heads must answer an object, got list"),
            ({"heads": 3}, "dispatch_main_heads must answer a heads list"),
            ({"heads": [1]}, "dispatch_main_heads heads must be objects, got int"),
            (
                {"heads": [{"repository": "MCPs", "head": "b", "generation": "3", "readAt": "t"}]},
                "generation",
            ),
        ],
    )
    def test_refuses_an_answer_it_cannot_read(self, answer: JSONValue, words: str) -> None:
        _test_hooks.env = queue_env()
        _test_hooks.http_post = FakeQueue([dump_json_str(answer)])

        with pytest.raises(JSONTypeError, match=words):
            main_history.held_heads(queue.load_credentials())


class TestProveReads:
    def test_sends_the_held_head_main_descends_from_and_none_otherwise(
        self, tmp_path: pathlib.Path
    ) -> None:
        remote = tmp_path / "remote" / "osrs-lab"
        commits = rerouted_repository(remote)
        rerouted = main_history.read_main(tmp_path / "mirrors", "osrs-lab", str(remote))
        unheld = main_history.MainRead(
            repository="fresh", mirror=rerouted["mirror"], head=commits["merge"], commits=()
        )
        rewritten = main_history.MainRead(
            repository="rewritten", mirror=rerouted["mirror"], head=commits["third"], commits=()
        )
        held = {
            "osrs-lab": main_history.HeldHead(
                repository="osrs-lab", head=commits["third"], generation=3, read_at=READ_AT
            ),
            "rewritten": main_history.HeldHead(
                repository="rewritten", head=commits["merge"], generation=5, read_at=READ_AT
            ),
        }

        proved, unproved = main_history.prove_reads((rerouted, unheld, rewritten), held)

        assert proved == (
            main_history.MainHistoryRead(
                repository="osrs-lab", commits=rerouted["commits"], since=commits["third"]
            ),
            main_history.MainHistoryRead(repository="fresh", commits=(), since=None),
            main_history.MainHistoryRead(repository="rewritten", commits=(), since=None),
        )
        assert unproved == ()

    def test_leaves_out_and_names_a_read_whose_ancestry_git_cannot_answer(
        self, tmp_path: pathlib.Path
    ) -> None:
        remote = tmp_path / "remote" / "MCPs"
        merged_repository(remote)
        read = main_history.read_main(tmp_path / "mirrors", "MCPs", str(remote))
        unheld = main_history.MainRead(
            repository="fresh", mirror=read["mirror"], head=read["head"], commits=()
        )
        blob = git(remote, "rev-parse", "main:a.txt")
        held = {
            "MCPs": main_history.HeldHead(
                repository="MCPs", head=blob, generation=1, read_at=READ_AT
            )
        }

        proved, unproved = main_history.prove_reads((read, unheld), held)

        assert proved == (main_history.MainHistoryRead(repository="fresh", commits=(), since=None),)
        assert len(unproved) == 1
        assert unproved[0].startswith(
            f"MCPs: EXPORT_FAILED: {main_history.ANCESTRY_UNREADABLE_CODE}: "
            f"git merge-base --is-ancestor {blob} "
        )
