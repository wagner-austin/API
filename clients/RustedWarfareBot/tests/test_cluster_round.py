"""The cluster round runner, driven against a scripted cluster conversation.

Every child process goes through the ``run_capture`` hook, so these script
the entire freeze-stage-converge-pull exchange and assert the commands the
runner actually issues -- the same commands an operator issues by hand,
because a second submission path is the parallel write path this workspace
bans.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from rw_bot.harness import _test_hooks
from rw_bot.harness.cluster_round import CONVERGE_PASSES, ClusterRound, ClusterRoundError

_JOBS = (
    "# the round's members",
    "control|101|doctrines/aa-counter-guard.doctrine|4000",
    "close3|101|doctrines/search/close3.doctrine|4000",
)


class _Cluster:
    """Scripts the round's command conversation and records every call.

    Attributes:
        argvs: Every command issued, in order.
        filed: Scorecard counts to serve, consumed per poll.
        queued: Queue counts to serve, consumed per poll.
    """

    def __init__(self, filed: list[int], queued: list[int]) -> None:
        self.argvs: list[tuple[str, ...]] = []
        self.filed = filed
        self.queued = queued

    def run(self, argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
        """Serve one command.

        Args:
            argv: The command issued.

        Returns:
            Exit status and output lines, per the script.
        """
        self.argvs.append(tuple(argv))
        joined = " ".join(argv)
        if joined.endswith("rev-parse HEAD"):
            return 0, ("abc123",)
        if "hpc3.cli.campaign" in joined:
            return 0, ("0 done, 0 in flight, 2 submitted, 2 remaining",)
        if 'grep -c "\\.txt$"' in joined:
            return 0, (str(self.filed.pop(0)),)
        if 'squeue --me -h -o "%j"' in joined:
            # The drain must match the array's QUALIFIED name end-anchored:
            # tasks carry the sweep's own name with no member suffix, and
            # the first drain's `<batch>-` pattern saw zero rows while 136
            # tasks ran (vhsearch2-r0, 2026-09-02).
            assert "\\.probe-r0$" in joined
            return 0, (str(self.queued.pop(0)),)
        return 0, ()


def _runner(tmp_path: Path) -> ClusterRound:
    return ClusterRound(
        config="runs/hpc3-rusted.json",
        host="hpc3",
        cluster_root="/pub/wagnera3/rusted",
        map_path="maps/skirmish/[p2]duel_lake.tmx",
        difficulty=1,
        fast_forward=10,
        scratch=tmp_path / "staging",
        sweeps_root=tmp_path / "sweeps",
        jobs_dir=tmp_path / "jobs",
        poll_seconds=5.0,
    )


def _drive(
    tmp_path: Path, cluster: _Cluster, batch: str = "probe-r0"
) -> tuple[_Cluster, list[float]]:
    slept: list[float] = []

    def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    saved = (_test_hooks.run_capture, _test_hooks.sleep)
    _test_hooks.run_capture = cluster.run
    _test_hooks.sleep = record_sleep
    try:
        _runner(tmp_path).run(batch, _JOBS)
    finally:
        (_test_hooks.run_capture, _test_hooks.sleep) = saved
    return cluster, slept


def test_a_clean_round_issues_the_canonical_chain_in_order(tmp_path: Path) -> None:
    """Freeze, stage, extract, document, converge, poll, pull -- the exact
    tools an operator drives by hand, in the exact order."""
    cluster, slept = _drive(tmp_path, _Cluster(filed=[2], queued=[]))
    joined = [" ".join(argv) for argv in cluster.argvs]
    stages = [
        "rev-parse HEAD",
        "scripts.stage_payload",
        "hpc3.cli.stage",
        "tar -xf /pub/wagnera3/rusted/staging/rw-payload.tar",
        "scripts.campaign_doc",
        "hpc3.cli.campaign",
        'grep -c "\\.txt$"',
        "scp",
    ]
    cursor = 0
    for stage in stages:
        matches = [i for i, line in enumerate(joined) if stage in line]
        assert matches != [], f"no command ran {stage}"
        assert min(matches) >= cursor, f"{stage} ran out of order"
        cursor = min(matches)
    assert slept == []

    # The job file the members will read is the one the driver wrote.
    written = (tmp_path / "jobs" / "probe-r0.txt").read_text(encoding="utf-8")
    assert "close3|101|doctrines/search/close3.doctrine|4000" in written

    # The freeze was told the real commit, and the payload is per-batch.
    freeze = next(line for line in joined if "scripts.stage_payload" in line)
    assert "--commit abc123" in freeze
    doc = next(line for line in joined if "scripts.campaign_doc" in line)
    assert "--payload payload-probe-r0" in doc
    assert "--difficulty 1" in doc

    # The pull lands where the margin scorer reads.
    pull = cluster.argvs[-1]
    assert pull[0] == "scp"
    assert pull[-1] == str(tmp_path / "sweeps" / "probe-r0")


def test_a_drained_but_short_round_reconverges_and_finishes(tmp_path: Path) -> None:
    """The measured casualty shape: the queue empties with members missing,
    and the next convergence pass resubmits exactly the gap."""
    cluster, slept = _drive(
        tmp_path,
        _Cluster(filed=[1, 1, 2], queued=[1, 0]),
    )
    converges = [argv for argv in cluster.argvs if "hpc3.cli.campaign" in " ".join(argv)]
    assert len(converges) == 2
    assert slept == [5.0]


def test_a_round_that_never_fills_is_a_loud_gap_not_a_loop(tmp_path: Path) -> None:
    cluster = _Cluster(filed=[0] * CONVERGE_PASSES, queued=[0] * CONVERGE_PASSES)
    with pytest.raises(ClusterRoundError) as caught:
        _drive(tmp_path, cluster)
    assert caught.value.code == "RW-CROUND-002"
    assert "0 of 2 scorecards" in caught.value.message
    converges = [argv for argv in cluster.argvs if "hpc3.cli.campaign" in " ".join(argv)]
    assert len(converges) == CONVERGE_PASSES
    # Nothing was pulled: a partial batch filed locally would read as a
    # complete measurement.
    assert all(argv[0] != "scp" for argv in cluster.argvs)


def test_a_failed_command_raises_with_its_own_words(tmp_path: Path) -> None:
    class _Refusing(_Cluster):
        def run(self, argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
            self.argvs.append(tuple(argv))
            if "scripts.stage_payload" in " ".join(argv):
                return 3, ("the freeze refused", "a second line")
            return 0, ("abc123",)

    with pytest.raises(ClusterRoundError) as caught:
        _drive(tmp_path, _Refusing(filed=[], queued=[]))
    assert caught.value.code == "RW-CROUND-001"
    assert "command failed (3)" in caught.value.message
    assert "the freeze refused" in caught.value.message
    assert "a second line" in caught.value.message


def test_the_search_entry_point_routes_the_cluster_prefix(tmp_path: Path) -> None:
    """`hpc3:<workspace>` builds the cluster runner with the workspace it
    names; everything else stays a queue DSN. Routed by explicit prefix,
    never sniffed."""
    from scripts.search import CLUSTER_PREFIX, main

    seen: list[tuple[str, ...]] = []

    def refuse(argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
        seen.append(tuple(argv))
        return 9, ("stopped by the test before anything real",)

    saved = _test_hooks.run_capture
    _test_hooks.run_capture = refuse
    try:
        with pytest.raises(ClusterRoundError):
            main(
                [f"{CLUSTER_PREFIX}runs/hpc3-rusted.json", "probe", "3"],
                sweeps_root=tmp_path / "sweeps",
                variant_dir=tmp_path / "variants",
            )
    finally:
        _test_hooks.run_capture = saved
    # The first real act was the runner asking git for the commit --
    # proof the cluster path was taken, not the queue.
    assert seen[0] == ("git", "rev-parse", "HEAD")


def test_python_module_invocations_use_this_interpreter(tmp_path: Path) -> None:
    """The tools run in THIS venv -- hpc3 is a path dependency -- so the
    interpreter is sys.executable, never a name the shell resolves."""
    cluster, _ = _drive(tmp_path, _Cluster(filed=[2], queued=[]))
    modules = [argv for argv in cluster.argvs if "-m" in argv]
    assert modules != []
    assert {argv[0] for argv in modules} == {sys.executable}
