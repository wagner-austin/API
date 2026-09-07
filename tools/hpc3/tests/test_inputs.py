"""Tests for refusing a run whose declared inputs are not on the cluster.

The check exists because job 55806418 was admitted with a clean preflight
and died fourteen seconds later on an absent payload. So these hold it to
the two things that makes it worth having: it refuses an absent input, and
it does NOT refuse an output the job is about to write.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core import _test_hooks as core_hooks
from hpc3.core._test_hooks import CommandResult
from hpc3.core.inputs import (
    declared_inputs,
    missing_inputs,
    present_on_cluster,
    require_inputs_present,
)

_TRAIN = (
    "HF_HOME=/pub/wagnera3/hf modeltrainer-cluster-train "
    "--payload /pub/wagnera3/code-style/payloads/code-style-qlora-v2.json "
    "--corpus-dir /pub/wagnera3/code-style/train-corpora "
    "--artifacts-dir /pub/wagnera3/code-style/artifacts/qlora-v2"
)

_GEN = (
    "modeltrainer-continuations "
    "--spec /pub/wagnera3/code-style/specs/code-style-gen-v2-base.json "
    "--out-dir /pub/wagnera3/code-style/generated/v2-base "
    "--record /pub/wagnera3/code-style/results/gen-v2-base.json"
)


class TestWhichPathsAreInputs:
    """The flag says which, not the directory."""

    def test_a_payload_is_an_input(self) -> None:
        assert declared_inputs(_TRAIN) == (
            "/pub/wagnera3/code-style/payloads/code-style-qlora-v2.json",
        )

    def test_a_spec_is_an_input(self) -> None:
        assert declared_inputs(_GEN) == (
            "/pub/wagnera3/code-style/specs/code-style-gen-v2-base.json",
        )

    def test_what_the_job_writes_is_not_required_to_exist(self) -> None:
        """--record, --out-dir and --artifacts-dir name outputs. Requiring
        them would refuse the FIRST run of everything, which is worse than
        the gap this closes."""
        for command in (_TRAIN, _GEN):
            for path in declared_inputs(command):
                assert "/results/" not in path
                assert "/generated/" not in path
                assert "/artifacts/" not in path

    def test_a_results_path_under_an_input_flag_would_still_be_an_input(self) -> None:
        """Deliberate: the directory is a habit, the flag is a fact. A file
        under results/ passed to --payload IS read, and a checker that
        believed the directory instead would wave it through."""
        command = "trainer --payload /pub/w/results/oddly-placed.json"

        assert declared_inputs(command) == ("/pub/w/results/oddly-placed.json",)

    def test_a_non_cluster_path_is_not_checked(self) -> None:
        """This can only speak for the shared filesystem."""
        assert declared_inputs("trainer --payload ./local.json") == ()

    def test_a_repeated_path_is_named_once(self) -> None:
        command = "t --payload /pub/w/p.json --spec /pub/w/p.json"

        assert declared_inputs(command) == ("/pub/w/p.json",)

    def test_a_flag_with_no_value_after_it_is_not_read_past(self) -> None:
        """A truncated command must not index off the end."""
        assert declared_inputs("trainer --payload") == ()

    def test_a_command_naming_nothing_declares_nothing(self) -> None:
        assert declared_inputs("modeltrainer-score --help") == ()


class TestTheRefusal:
    def test_an_absent_input_is_refused_and_named(self) -> None:
        with pytest.raises(AppError) as caught:
            require_inputs_present(_TRAIN, frozenset())

        error: AppError[Hpc3ErrorCode] = caught.value
        assert error.code == Hpc3ErrorCode.RUN_INPUT_MISSING
        assert "code-style-qlora-v2.json" in str(error)

    def test_a_present_input_passes(self) -> None:
        require_inputs_present(
            _TRAIN, frozenset({"/pub/wagnera3/code-style/payloads/code-style-qlora-v2.json"})
        )

    def test_every_absent_path_is_named_not_only_the_first(self) -> None:
        """Otherwise a reader stages one file and runs into the next."""
        command = "t --payload /pub/w/a.json --spec /pub/w/b.json"

        with pytest.raises(AppError) as caught:
            require_inputs_present(command, frozenset())

        error: AppError[Hpc3ErrorCode] = caught.value
        message = str(error)
        assert "/pub/w/a.json" in message
        assert "/pub/w/b.json" in message

    def test_a_command_declaring_no_inputs_is_not_refused(self) -> None:
        require_inputs_present("modeltrainer-score --help", frozenset())

    def test_missing_inputs_preserves_declaration_order(self) -> None:
        paths = ("/pub/w/a.json", "/pub/w/b.json", "/pub/w/c.json")

        assert missing_inputs(paths, frozenset({"/pub/w/b.json"})) == (
            "/pub/w/a.json",
            "/pub/w/c.json",
        )


class TestAskingTheCluster:
    def test_absence_is_an_answer_rather_than_a_failure(self) -> None:
        """The probe ends `; true` so the loop does not inherit the exit
        status of its last test. Without it an absent file makes ssh exit 1
        and run_remote raises REMOTE_COMMAND_FAILED -- reporting an
        infrastructure fault for the ordinary case this exists to find."""
        seen: list[list[str]] = []

        def _fake_run(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
            _ = stdin_bytes
            seen.append(list(argv))
            return CommandResult(returncode=0, stdout="/pub/w/a.json\n", stderr="")

        original = core_hooks.run
        core_hooks.run = _fake_run
        try:
            found = present_on_cluster("hpc3", ("/pub/w/a.json", "/pub/w/b.json"))
        finally:
            core_hooks.run = original

        assert found == frozenset({"/pub/w/a.json"})
        assert seen[0][-1].endswith("; true")

    def test_no_paths_reaches_no_network(self) -> None:
        """A run declaring nothing must not pay for an ssh round trip."""

        def _explode(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
            _ = stdin_bytes
            raise AssertionError(f"the cluster was contacted for nothing: {list(argv)}")

        original = core_hooks.run
        core_hooks.run = _explode
        try:
            assert present_on_cluster("hpc3", ()) == frozenset()
        finally:
            core_hooks.run = original

    def test_every_path_is_asked_about_in_one_round_trip(self) -> None:
        """One ssh call for the whole set: this sits in front of every
        submission and a per-path probe would make the common case -- all
        present -- the slowest."""
        calls: list[list[str]] = []

        def _fake_run(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
            _ = stdin_bytes
            calls.append(list(argv))
            return CommandResult(returncode=0, stdout="", stderr="")

        original = core_hooks.run
        core_hooks.run = _fake_run
        try:
            present_on_cluster("hpc3", ("/pub/w/a.json", "/pub/w/b.json", "/pub/w/c.json"))
        finally:
            core_hooks.run = original

        assert len(calls) == 1
        for path in ("/pub/w/a.json", "/pub/w/b.json", "/pub/w/c.json"):
            assert path in calls[0][-1]


__all__ = ["TestAskingTheCluster", "TestTheRefusal", "TestWhichPathsAreInputs"]
