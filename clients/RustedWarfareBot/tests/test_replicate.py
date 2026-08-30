"""Reading a replication panel's verdict, end to end against the fake host.

The comparison itself is exercised in ``test_harness_replication``; what is
checked here is the part that decides what gets run afterwards -- which pairs
are found, and whether the exit status can be trusted to gate a sweep.
"""

from __future__ import annotations

import runpy

import pytest
from scripts.replicate import (
    EXIT_BAD_USAGE,
    EXIT_FORKED,
    EXIT_OK,
    LEFT_LABEL,
    RIGHT_LABEL,
    main,
    seeds_in,
)

from rw_bot.harness.results_layout import TRACE_ROOT
from tests.harness_fakes import FakeHost
from tests.test_harness_replication import FORKED, TRACE

_BATCH = "replicate"
_SEEDS = (12345, 777)


def _planted(forked: tuple[int, ...] = ()) -> FakeHost:
    """Build a host holding a panel's traces.

    Args:
        forked: Seeds whose second member should diverge.

    Returns:
        The host.
    """
    host = FakeHost()
    for seed in _SEEDS:
        host.files[f"{TRACE_ROOT}/{_BATCH}/{LEFT_LABEL}-s{seed}.ndjson"] = TRACE
        host.files[f"{TRACE_ROOT}/{_BATCH}/{RIGHT_LABEL}-s{seed}.ndjson"] = (
            FORKED if seed in forked else TRACE
        )
    host.dirs.add(f"{TRACE_ROOT}/{_BATCH}")
    return host


class TestFindingThePairs:
    def test_a_seed_with_both_members_is_compared(self) -> None:
        names = [f"{LEFT_LABEL}-s777.ndjson", f"{RIGHT_LABEL}-s777.ndjson"]
        assert seeds_in(names) == (777,)

    def test_a_seed_with_only_one_member_is_not(self) -> None:
        """A member that never ran is a gap in the panel, not a pass. Compared
        against nothing it would have to be called identical."""
        assert seeds_in([f"{LEFT_LABEL}-s777.ndjson"]) == ()

    def test_the_seeds_come_back_in_order(self) -> None:
        names = [
            f"{label}-s{seed}.ndjson"
            for seed in (99991, 777)
            for label in (LEFT_LABEL, RIGHT_LABEL)
        ]
        assert seeds_in(names) == (777, 99991)

    def test_a_file_that_is_not_a_trace_is_ignored(self) -> None:
        names = [
            f"{LEFT_LABEL}-s777.ndjson",
            f"{RIGHT_LABEL}-s777.ndjson",
            f"{LEFT_LABEL}-s777.txt",
            "notes.md",
            f"{LEFT_LABEL}-sxyz.ndjson",
        ]
        assert seeds_in(names) == (777,)


class TestTheVerdict:
    def test_a_panel_that_replicated_exits_ok(self) -> None:
        with _planted() as host:
            assert main([_BATCH]) == EXIT_OK
        assert any("the regime holds" in line for line in host.printed)

    def test_it_reports_every_pair(self) -> None:
        with _planted() as host:
            main([_BATCH])
        reported = [line for line in host.printed if "identical over" in line]
        assert len(reported) == len(_SEEDS)

    def test_one_forked_pair_fails_the_panel(self) -> None:
        """Exits non-zero so it can gate what runs after it: a sweep launched
        on an uncertified regime produces numbers nobody can place."""
        with _planted(forked=(777,)) as host:
            assert main([_BATCH]) == EXIT_FORKED
        assert any("FORKED at frame 225" in line for line in host.printed)
        assert any("is NOT certified" in line for line in host.printed)

    def test_it_counts_what_replicated(self) -> None:
        with _planted(forked=(777,)) as host:
            main([_BATCH])
        assert any("1/2 pair(s) replicated" in line for line in host.printed)

    def test_an_empty_panel_fails_and_says_it_certified_nothing(self) -> None:
        """Distinguished from a fork on purpose: both exit non-zero and only
        one of them is a determinism finding. Reporting them alike is how "we
        ran it and it failed" becomes "the regime does not hold"."""
        host = FakeHost()
        host.dirs.add(f"{TRACE_ROOT}/{_BATCH}")
        with host:
            assert main([_BATCH]) == EXIT_FORKED
        assert any("certified nothing" in line for line in host.printed)
        assert not any("NOT certified under this runtime" in line for line in host.printed)


class TestTheEntryPoint:
    def test_a_trace_root_may_be_given(self) -> None:
        """The cluster files traces under the project's own directory, so the
        root is an argument rather than a repository-relative assumption."""
        host = FakeHost()
        root = "/pub/wagnera3/rusted/runs/traces"
        for seed in _SEEDS:
            host.files[f"{root}/{_BATCH}/{LEFT_LABEL}-s{seed}.ndjson"] = TRACE
            host.files[f"{root}/{_BATCH}/{RIGHT_LABEL}-s{seed}.ndjson"] = TRACE
        host.dirs.add(f"{root}/{_BATCH}")
        with host:
            assert main([_BATCH, root]) == EXIT_OK

    def test_a_bad_argument_count_prints_usage(self) -> None:
        with FakeHost() as host:
            assert main([]) == EXIT_BAD_USAGE
            assert any(line.startswith("usage: replicate") for line in host.printed)

    def test_it_reads_the_process_arguments_when_given_none(self) -> None:
        host = _planted()
        host.argv = [_BATCH]
        with host:
            assert main(None) == EXIT_OK

    def test_the_module_guard_runs_main(self) -> None:
        host = _planted()
        host.argv = [_BATCH]
        with host:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("scripts.replicate", run_name="__main__")
            assert caught.value.code == EXIT_OK
