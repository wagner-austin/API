"""Tests for tracing a result back to the run that produced it.

The question this answers is asked backwards and months late: an outcome file
turns up and nobody can say which corpus it came from. Before the ledger
carried an experiment record, the only link was a job name somebody typed --
and ``arm-b-43`` mistyped as ``arm-b-42`` gives two jobs claiming one identity
with no error anywhere.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str

from hpc3.cli import trace as trace_cli
from hpc3.contracts.experiment import (
    MAX_COMMENT_LENGTH,
    comment_fragment,
    format_experiment,
    matches,
    require_experiment,
)
from tests.conftest import FakeRun, workspace_document, write_file, write_workspace

_ARM_B = "07ab4976" + "a" * 56
_ARM_C = "4c91fbc1" + "b" * 56


def _entry(job_id: str, name: str, experiment: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Build a ledger record.

    Args:
        job_id: Job id.
        name: Qualified job name.
        experiment: The run's identity record.

    Returns:
        The record.
    """
    return {
        "job_id": job_id,
        "project": "abl",
        "name": name,
        "host": "hpc3",
        "partition": "free-gpu",
        "submitted_at": "2026-08-22T16:00:00+00:00",
        "log_dir": "/pub/w/abl/logs",
        "deterministic": False,
        "experiment": experiment,
    }


def _args(tmp_path: pathlib.Path, *records: dict[str, JSONValue]) -> list[str]:
    """Write a workspace and a ledger, and build the trace arguments.

    Args:
        tmp_path: Directory holding both.
        *records: Ledger records to write, oldest first.

    Returns:
        Arguments excluding the program name and the question flag.
    """
    config = write_workspace(tmp_path / "hpc3.json", workspace_document())
    payload = "".join(dump_json_str(record) + "\n" for record in records)
    write_file(tmp_path / "ledger.jsonl", payload.encode("utf-8"))
    return ["--config", config]


class TestTraceByExperimentValue:
    def test_it_names_the_job_that_trained_a_corpus(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The question the reviewer asked for, answered from the ledger."""
        args = _args(
            tmp_path,
            _entry("101", "abl.armB-s42", {"corpus": _ARM_B, "seed": "42"}),
            _entry("102", "abl.armC-s42", {"corpus": _ARM_C, "seed": "42"}),
        )
        assert trace_cli.main([*args, "--match", _ARM_B]) == 0
        assert emitted[0] == "101 abl.armB-s42 submitted 2026-08-22T16:00:00+00:00"
        assert emitted[1] == f"  corpus={_ARM_B} seed=42"
        assert emitted[-1] == "1 of 2 recorded run(s) match"

    def test_two_seeds_over_one_corpus_both_match(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        args = _args(
            tmp_path,
            _entry("101", "abl.armB-s42", {"corpus": _ARM_B, "seed": "42"}),
            _entry("102", "abl.armB-s43", {"corpus": _ARM_B, "seed": "43"}),
        )
        assert trace_cli.main([*args, "--match", _ARM_B]) == 0
        assert emitted[-1] == "2 of 2 recorded run(s) match"

    def test_a_key_matches_as_well_as_a_value(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        args = _args(tmp_path, _entry("101", "abl.a", {"corpus": _ARM_B}))
        assert trace_cli.main([*args, "--match", "corpus"]) == 0

    def test_no_match_exits_non_zero_without_claiming_nothing_ran(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        args = _args(tmp_path, _entry("101", "abl.a", {"corpus": _ARM_B}))
        assert trace_cli.main([*args, "--match", "4c91fbc1"]) == 1
        assert emitted[0] == "no recorded run matches; the ledger holds 1 entry(s)"


class TestTraceByJobId:
    def test_it_says_what_a_job_trained(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        args = _args(tmp_path, _entry("55519937", "abl.armB-s42", {"corpus": _ARM_B}))
        assert trace_cli.main([*args, "--job", "55519937"]) == 0
        assert emitted[1] == f"  corpus={_ARM_B}"
        assert emitted[2] == "  logs /pub/w/abl/logs"

    def test_an_unrecorded_job_exits_non_zero(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        args = _args(tmp_path, _entry("101", "abl.a", {"corpus": _ARM_B}))
        assert trace_cli.main([*args, "--job", "999"]) == 1


class TestTraceArguments:
    def test_asking_neither_question_is_refused(self, tmp_path: pathlib.Path) -> None:
        args = _args(tmp_path, _entry("101", "abl.a", {"corpus": _ARM_B}))
        with pytest.raises(ValueError, match="exactly one of --match or --job"):
            trace_cli.main(args)

    def test_asking_both_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Answering the one it picked would look authoritative either way."""
        args = _args(tmp_path, _entry("101", "abl.a", {"corpus": _ARM_B}))
        with pytest.raises(ValueError, match="exactly one of --match or --job"):
            trace_cli.main([*args, "--match", _ARM_B, "--job", "101"])

    def test_the_config_flag_is_not_optional(self) -> None:
        with pytest.raises(ValueError, match="--config is required"):
            trace_cli.main(["--match", _ARM_B])

    def test_the_entrypoint_reads_the_process_arguments(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], argv: list[str]
    ) -> None:
        args = _args(tmp_path, _entry("101", "abl.a", {"corpus": _ARM_B}))
        argv[:] = ["prog", *args, "--match", _ARM_B]
        with pytest.raises(SystemExit) as excinfo:
            trace_cli.entrypoint()
        assert excinfo.value.code == 0


class TestExperimentContract:
    def test_it_records_pairs_verbatim(self) -> None:
        decoded = require_experiment({"e": {"corpus": _ARM_B, "seed": "42"}}, "e")
        assert decoded == {"corpus": _ARM_B, "seed": "42"}

    def test_an_empty_record_is_refused(self) -> None:
        """A job id says which queue row it was, not which result it made."""
        with pytest.raises(JSONTypeError, match="at least one thing identifying"):
            require_experiment({"e": {}}, "e")

    def test_a_missing_record_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            require_experiment({}, "e")

    def test_a_non_string_value_is_refused(self) -> None:
        """A seed is written '42', so a record round-trips through JSON."""
        with pytest.raises(JSONTypeError, match="maps to int"):
            require_experiment({"e": {"seed": 42}}, "e")

    def test_an_empty_value_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="empty name or value"):
            require_experiment({"e": {"seed": ""}}, "e")

    def test_an_empty_key_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="empty name or value"):
            require_experiment({"e": {"": "42"}}, "e")

    def test_it_formats_in_a_stable_order(self) -> None:
        assert format_experiment({"seed": "42", "arm": "B"}) == "arm=B seed=42"


class TestWhitespaceIsRefusedBeforeItReachesSlurm:
    """`comment_fragment` always REQUIRED this and nothing ever CHECKED it.

    Its docstring says the pairs are joined with commas "because Slurm takes
    the comment as a single token" -- correct, and stated where no input
    passes through. A run declaring `established_on: "NVIDIA GeForce RTX
    3090 Ti"` rendered

        #SBATCH --comment project=floor;...;exp=established_on=NVIDIA GeForce...

    unquoted, and Slurm refused the whole script with `Invalid directive
    found in batch script: GeForce` -- naming neither the field, nor the run,
    nor the comment, after an SSH round trip. Measured 2026-08-25 preflighting
    the A100 floor run.
    """

    def test_a_space_in_a_value_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="whitespace in a value"):
            require_experiment({"e": {"card": "NVIDIA GeForce RTX 3090 Ti"}}, "e")

    def test_a_space_in_a_name_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="whitespace in a name"):
            require_experiment({"e": {"base model": "gpt2"}}, "e")

    def test_a_tab_is_refused_as_readily_as_a_space(self) -> None:
        """A tab renders identically to a space and breaks identically."""
        with pytest.raises(JSONTypeError, match="whitespace in a value"):
            require_experiment({"e": {"card": "A100\t40GB"}}, "e")

    def test_a_newline_is_refused(self) -> None:
        """A newline would end the directive and start a shell line."""
        with pytest.raises(JSONTypeError, match="whitespace in a value"):
            require_experiment({"e": {"note": "line\nbreak"}}, "e")

    def test_the_message_says_what_to_do_instead(self) -> None:
        """The cluster's own error names a random word; this must not."""
        with pytest.raises(JSONTypeError) as excinfo:
            require_experiment({"e": {"card": "RTX 3090"}}, "e")
        assert "underscore or a hyphen" in str(excinfo.value)

    def test_the_separators_the_record_uses_are_still_allowed(self) -> None:
        """Only whitespace is refused; a value is still free-form otherwise."""
        decoded = require_experiment({"e": {"card": "NVIDIA_GeForce_RTX_3090_Ti"}}, "e")
        assert decoded == {"card": "NVIDIA_GeForce_RTX_3090_Ti"}


class TestMatches:
    def test_an_exact_value_matches(self) -> None:
        assert matches({"corpus": _ARM_B}, _ARM_B) is True

    def test_a_prefix_does_not_match(self) -> None:
        """A prefix matching two corpora would answer with two jobs, silently."""
        assert matches({"corpus": _ARM_B}, "07ab4976") is False

    def test_an_unrelated_value_does_not_match(self) -> None:
        assert matches({"corpus": _ARM_B}, _ARM_C) is False


class TestCommentFragment:
    def test_it_joins_without_spaces(self) -> None:
        """Slurm takes --comment as one token; a space truncates the rest."""
        fragment = comment_fragment({"seed": "42", "arm": "B"})
        assert fragment == "arm=B,seed=42"
        assert " " not in fragment

    def test_a_long_record_is_truncated_rather_than_breaking_the_comment(self) -> None:
        long_record = {f"k{index}": "v" * 20 for index in range(20)}
        fragment = comment_fragment(long_record)
        assert len(fragment) == MAX_COMMENT_LENGTH
        assert fragment.endswith("…")

    def test_a_record_at_the_limit_is_not_truncated(self) -> None:
        exact = {"k": "v" * (MAX_COMMENT_LENGTH - 2)}
        assert comment_fragment(exact).endswith("v")
