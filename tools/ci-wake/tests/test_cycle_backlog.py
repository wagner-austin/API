"""Bound outage recovery without losing the older enrollments."""

from __future__ import annotations

import pathlib

from platform_core.mcp_testing import announcing_poster

from ci_wake import _test_hooks
from ci_wake.cycle import MAX_PUSHES_PER_CYCLE, run_cycle
from ci_wake.enrolment import attempt_key
from ci_wake.position import position_path, read_announced
from ci_wake.runs import jobs_argv, runs_argv
from tests.conftest import CONFIGURED_ENV, REPO, FakeCompleted, FakeGh, pin_env
from tests.test_cycle import RUN_ID, _enrol, _jobs_reply, _run_entry, _runs_reply


def test_newest_batch_is_bounded_and_the_next_tick_drains_older_work(
    tmp_path: pathlib.Path, frozen_clock: int, emitted: list[str]
) -> None:
    pin_env(CONFIGURED_ENV)
    path = tmp_path / "pushes.jsonl"
    shas = [f"{index:040x}" for index in range(1, MAX_PUSHES_PER_CYCLE + 2)]
    replies: dict[tuple[str, ...], FakeCompleted] = {
        jobs_argv(REPO, RUN_ID): _jobs_reply(),
    }
    for index, sha in enumerate(shas):
        _enrol(path, sha=sha, ago=3600 - index)
        replies[runs_argv(REPO, sha)] = _runs_reply(_run_entry())
    gh = FakeGh(replies)
    _test_hooks.run_process = gh
    _test_hooks.http_post = announcing_poster()

    run_cycle(path)

    assert len(gh.calls) == MAX_PUSHES_PER_CYCLE * 2
    assert gh.calls[0] == runs_argv(REPO, shas[-1])
    assert runs_argv(REPO, shas[0]) not in gh.calls
    assert read_announced(position_path(path)) == frozenset(
        attempt_key(REPO, sha) for sha in shas[1:]
    )
    assert emitted[0] == (
        "backlog: inspecting 20 of 21 outstanding pushes; "
        "newest first, deferred pushes remain unacknowledged"
    )

    remaining = FakeGh(
        {
            runs_argv(REPO, shas[0]): _runs_reply(_run_entry()),
            jobs_argv(REPO, RUN_ID): _jobs_reply(),
        }
    )
    _test_hooks.run_process = remaining
    _test_hooks.http_post = announcing_poster()
    run_cycle(path)
    assert remaining.calls == [runs_argv(REPO, shas[0]), jobs_argv(REPO, RUN_ID)]
    assert read_announced(position_path(path)) == frozenset(attempt_key(REPO, sha) for sha in shas)


__all__: list[str] = []
