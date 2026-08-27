"""The trace report, over records written to a real directory.

The records here are synthesised, and that is the right call for the same
reason it is in the ladder report's tests: what is under test is the READING
-- which tensor is named as the first difference, in which order, and whether
a tensor one run skipped is reported instead of counted as agreement. A CPU
cannot produce two cards, so the interesting cases are unreachable from real
measurement on a test runner, and computing them would exercise the trace,
which has its own tests.

The one thing NOT synthesised is the naming scheme. Names are built by
``trace_plan``, so a change there breaks this file rather than quietly
producing a report in which no observation parses and every rung is empty.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import TRUE, determinism_record
from platform_core.json_utils import dump_json_str
from platform_core.run_record import (
    Observation,
    RunAgreement,
    RunRecord,
    agree_across_runs,
    encode_run_record,
    run_record,
)

from model_trainer.cli import probe_trace_report as report_cli
from model_trainer.cli.record_reports import agreement_groups
from model_trainer.core.services.model.trace_plan import (
    DIGEST_SUFFIX,
    SUM_SUFFIX,
    TRACE_EXPERIMENT,
    TRACE_RUNGS,
    WORKSPACE_NAME,
    WORKSPACE_UNSET,
    TraceName,
    trace_label,
    trace_loss_name,
    trace_tensor_name,
)

PINNED = determinism_record("torch", {"cudnn_deterministic": TRUE})

#: A short stand-in graph: the boundaries a real trace crosses, in the order
#: it crosses them, with the module classes the tiny rung actually produces.
GRAPH: tuple[tuple[str, str, str], ...] = (
    ("Embedding", "transformer.wte", "in"),
    ("Embedding", "transformer.wte", "out"),
    ("LayerNorm", "transformer.h.0.ln_1", "out"),
    ("Conv1D", "transformer.h.0.attn.c_attn", "out"),
    ("Conv1D", "transformer.h.0.attn.c_proj", "in"),
    ("Conv1D", "transformer.h.0.attn.c_proj", "out"),
    ("Linear", "lm_head", "out"),
)


def _name(rung: str, step: int, suffix: str) -> str:
    """Build one observation name from the stand-in graph."""
    module_class, path, kind = GRAPH[step]
    return trace_tensor_name(
        TraceName(
            rung=rung,
            step=step,
            kind=kind,
            index=0,
            module_class=module_class,
            path=path,
            suffix=suffix,
        )
    )


def trace_record(
    gpu: str,
    *,
    rungs: tuple[str, ...] = ("tiny",),
    diverge_from: int | None = None,
    loss: float = 6.25,
    steps: int = len(GRAPH),
    workspace: float | None = WORKSPACE_UNSET,
) -> RunRecord:
    """Build a trace record for one card.

    Args:
        gpu: The card the run reports.
        rungs: The rungs the record covers.
        diverge_from: First step whose digest is nudged, and every step after
            it -- which is how a real divergence behaves, since a transformer
            carries a difference forward through the residual stream. None
            leaves every digest shared.
        loss: The loss every rung reports.
        steps: How many of the stand-in graph's boundaries to record.
        workspace: The split-K condition to record, or None to omit it --
            which is what a record written before the trace recorded the
            condition looks like.

    Returns:
        The record.
    """
    observations: list[Observation] = []
    if workspace is not None:
        observations.append(Observation(name=WORKSPACE_NAME, value=workspace))
    for rung in rungs:
        for step in range(steps):
            nudge = 1.0 if diverge_from is not None and step >= diverge_from else 0.0
            observations.append(
                Observation(name=_name(rung, step, DIGEST_SUFFIX), value=100.0 + step + nudge)
            )
            observations.append(
                Observation(name=_name(rung, step, SUM_SUFFIX), value=float(step) + nudge)
            )
        observations.append(Observation(name=trace_loss_name(rung), value=loss))

    return run_record(
        experiment=TRACE_EXPERIMENT,
        label=trace_label(rungs),
        fingerprint=RunFingerprint(
            image_digest="sha256:b002cffc",
            gpu_model=gpu,
            driver_version="580.82.07",
            determinism=PINNED,
        ),
        observations=tuple(observations),
        payload_digest="",
    )


def write_records(directory: pathlib.Path, records: dict[str, RunRecord]) -> pathlib.Path:
    """Write records into a directory under the given filenames."""
    directory.mkdir(parents=True, exist_ok=True)
    for stem, record in records.items():
        (directory / f"{stem}.json").write_text(
            dump_json_str(encode_run_record(record)), encoding="utf-8"
        )
    return directory


def agreement_of(*records: RunRecord) -> RunAgreement:
    """Compute agreement over the given records."""
    return agree_across_runs(records)


class TestCollectingARungsDigests:
    def test_it_keeps_only_digest_rows(self) -> None:
        agreement = agreement_of(trace_record("A30"), trace_record("A100"))

        collected = report_cli.rung_digests(agreement, "tiny")
        suffixes = [name["suffix"] for name, _ in collected]

        assert suffixes == [DIGEST_SUFFIX] * len(GRAPH)

    def test_it_returns_them_in_execution_order(self) -> None:
        agreement = agreement_of(trace_record("A30"), trace_record("A100"))

        collected = report_cli.rung_digests(agreement, "tiny")

        assert [pair[0]["step"] for pair in collected] == list(range(len(GRAPH)))

    def test_a_rung_no_record_carries_collects_nothing(self) -> None:
        agreement = agreement_of(trace_record("A30"), trace_record("A100"))

        assert report_cli.rung_digests(agreement, "xl") == ()

    def test_it_does_not_mix_two_rungs(self) -> None:
        pair = ("tiny", "xl")
        agreement = agreement_of(trace_record("A30", rungs=pair), trace_record("A100", rungs=pair))

        collected = report_cli.rung_digests(agreement, "xl")

        assert {p[0]["rung"] for p in collected} == {"xl"}


class TestFindingTheDivergence:
    def test_identical_runs_have_none(self) -> None:
        agreement = agreement_of(trace_record("A30"), trace_record("A100"))

        assert report_cli.divergences(report_cli.rung_digests(agreement, "tiny")) == ()

    def test_it_keeps_every_differing_tensor_from_the_first_onward(self) -> None:
        agreement = agreement_of(trace_record("A30"), trace_record("A100", diverge_from=3))

        differing = report_cli.divergences(report_cli.rung_digests(agreement, "tiny"))

        assert [pair[0]["step"] for pair in differing] == [3, 4, 5, 6]

    def test_the_first_one_is_the_operation_that_carried_it(self) -> None:
        agreement = agreement_of(trace_record("A30"), trace_record("A100", diverge_from=4))

        first = report_cli.divergences(report_cli.rung_digests(agreement, "tiny"))[0][0]

        assert (first["path"], first["kind"]) == ("transformer.h.0.attn.c_proj", "in")


class TestNamingTheRungsARecordCovers:
    def test_it_orders_them_as_the_trace_declares_them(self) -> None:
        # Not alphabetically. A record sorts its observations by name, so
        # name order would print "large, medium, tiny, xl" and bury the
        # contrast the rung set was chosen for. The step counter cannot
        # supply the order either: every rung gets a fresh trace and so
        # starts at step zero.
        rungs = ("tiny", "medium", "large", "xl")
        agreement = agreement_of(
            trace_record("A30", rungs=rungs), trace_record("A100", rungs=rungs)
        )

        assert report_cli.traced_rungs(agreement) == TRACE_RUNGS

    def test_a_subset_keeps_the_declared_order(self) -> None:
        pair = ("xl", "tiny")
        agreement = agreement_of(trace_record("A30", rungs=pair), trace_record("A100", rungs=pair))

        assert report_cli.traced_rungs(agreement) == ("tiny", "xl")

    def test_a_rung_the_declaration_does_not_know_is_listed_not_dropped(self) -> None:
        # A record from a trace whose rung set has since changed still gets
        # reported, after the declared ones.
        rungs = ("tiny", "small")
        agreement = agreement_of(
            trace_record("A30", rungs=rungs), trace_record("A100", rungs=rungs)
        )

        assert report_cli.traced_rungs(agreement) == ("tiny", "small")

    def test_a_record_with_no_traced_tensors_names_no_rungs(self) -> None:
        empty = run_record(
            experiment=TRACE_EXPERIMENT,
            label=trace_label(("tiny",)),
            fingerprint=RunFingerprint(
                image_digest="sha256:b002cffc",
                gpu_model="A30",
                driver_version="580.82.07",
                determinism=PINNED,
            ),
            observations=(Observation(name=trace_loss_name("tiny"), value=6.25),),
            payload_digest="",
        )

        assert report_cli.traced_rungs(agreement_of(empty, empty)) == ()


class TestTheRenderedRungSection:
    def test_agreement_is_stated_rather_than_left_to_an_absence(self) -> None:
        agreement = agreement_of(trace_record("A30"), trace_record("A100"))

        lines = report_cli.rung_lines(agreement, "tiny")

        assert lines == (
            "rung tiny",
            "  loss distinct=1 runs=0,1  6.25 6.25",
            f"  {len(GRAPH)} tensors traced by every run, 0 differ",
            "  -> every traced tensor is bit-identical across these runs",
        )

    def test_it_names_the_module_where_the_difference_starts(self) -> None:
        agreement = agreement_of(trace_record("A30"), trace_record("A100", diverge_from=4))

        lines = report_cli.rung_lines(agreement, "tiny")

        assert lines[3] == (
            "  -> first difference at step 4: "
            "Conv1D.transformer.h.0.attn.c_proj (input #0), runs=0|1"
        )

    def test_it_counts_what_was_traced_and_what_differed(self) -> None:
        agreement = agreement_of(trace_record("A30"), trace_record("A100", diverge_from=5))

        lines = report_cli.rung_lines(agreement, "tiny")

        assert lines[2] == f"  {len(GRAPH)} tensors traced by every run, 2 differ"

    def test_it_prints_the_loss_control_at_full_precision(self) -> None:
        agreement = agreement_of(
            trace_record("A30", loss=6.250983715057373),
            trace_record("A100", loss=6.250984191894531),
        )

        lines = report_cli.rung_lines(agreement, "tiny")

        assert lines[1] == "  loss distinct=2 runs=0|1  6.250983715057373 6.2509841918945312"

    def test_it_stops_listing_after_the_first_few(self) -> None:
        long_graph = agreement_of(trace_record("A30"), trace_record("A100", diverge_from=0))

        lines = report_cli.rung_lines(long_graph, "tiny")
        listed = [line for line in lines if line.startswith("    step ")]

        assert len(listed) == report_cli.FOLLOWING_SHOWN + 1

    def test_a_rung_nobody_traced_says_so(self) -> None:
        agreement = agreement_of(trace_record("A30"), trace_record("A100"))

        assert report_cli.rung_lines(agreement, "xl") == (
            "rung xl",
            "  no traced tensors shared by every run",
        )

    def test_a_rung_whose_loss_one_run_omitted_is_not_reported_as_agreeing(self) -> None:
        full = trace_record("A30")
        without_loss = run_record(
            experiment=TRACE_EXPERIMENT,
            label=trace_label(("tiny",)),
            fingerprint=full["fingerprint"],
            observations=tuple(
                o for o in full["observations"] if o["name"] != trace_loss_name("tiny")
            ),
            payload_digest="",
        )

        lines = report_cli.rung_lines(agreement_of(full, without_loss), "tiny")

        assert lines[1] == "  loss not reported by every run"


class TestSayingWhichRunsAgreed:
    def test_every_run_agreeing_is_one_group(self) -> None:
        assert agreement_groups((1.0, 1.0, 1.0)) == "0,1,2"

    def test_the_odd_run_out_is_named_by_index(self) -> None:
        assert agreement_groups((1.0, 2.0, 1.0)) == "0,2|1"

    def test_a_leading_odd_run_still_reads_left_to_right(self) -> None:
        # Groups are ordered by FIRST APPEARANCE, not by size, so run 0 is
        # always in the first group and the rendering does not silently
        # reorder the runs the header just listed.
        assert agreement_groups((2.0, 1.0, 1.0)) == "0|1,2"

    def test_three_different_values_are_three_groups(self) -> None:
        assert agreement_groups((1.0, 2.0, 3.0)) == "0|1|2"

    def test_two_runs_are_rendered_as_a_pair(self) -> None:
        assert agreement_groups((1.0, 2.0)) == "0|1"

    def test_it_appears_on_the_first_difference_line(self) -> None:
        # Which card is alone is the finding: the earlier ladder work showed
        # the odd card MOVES between rungs and conditions, and a bare count
        # cannot show that.
        agreement = agreement_of(
            trace_record("V100"),
            trace_record("A30", diverge_from=4),
            trace_record("A100", diverge_from=4),
        )

        lines = report_cli.rung_lines(agreement, "tiny")

        assert lines[3].endswith("(input #0), runs=0|1,2")


class TestSayingWhichConditionARunUsed:
    def test_an_unset_workspace_reads_as_unset(self) -> None:
        assert report_cli.describe_condition(trace_record("A30")) == "unset"

    def test_the_intervention_reads_as_zero(self) -> None:
        assert report_cli.describe_condition(trace_record("A30", workspace=0.0)) == "0"

    def test_a_record_that_never_recorded_it_says_so_rather_than_guessing(self) -> None:
        # "did not set the variable" and "cannot say whether it set the
        # variable" are different facts, and only one is a measurement. The
        # six ladder records from 2026-08-27 are the second kind.
        assert report_cli.describe_condition(trace_record("A30", workspace=None)) == "NOT RECORDED"


class TestTheWholeReport:
    def test_it_heads_with_the_run_count_and_each_card(self) -> None:
        lines = report_cli.report_lines(
            (("a30.json", trace_record("NVIDIA A30")), ("a100.json", trace_record("A100 80GB")))
        )

        assert lines[0] == f"2 runs, experiment {TRACE_EXPERIMENT}"
        assert "NVIDIA A30" in lines[1]
        assert "A100 80GB" in lines[2]

    def test_each_run_line_names_the_condition_it_ran_under(self) -> None:
        # Nothing in a RunFingerprint carries CUBLASLT_WORKSPACE_SIZE, so
        # these two runs difference as "identical configuration" despite
        # differing in the one variable under study. This line is the only
        # place the report can say which arm each run is.
        lines = report_cli.report_lines(
            (
                ("default.json", trace_record("A30")),
                ("nosplitk.json", trace_record("A30", workspace=0.0)),
            )
        )

        assert lines[1].endswith("cublaslt_workspace=unset")
        assert lines[2].endswith("cublaslt_workspace=0")

    def test_the_condition_is_compared_like_any_other_observation(self) -> None:
        # It is an observation, so two runs under different arms disagree on
        # it -- which means a set of records that were supposed to share an
        # arm and did not is visible rather than assumed.
        agreement = agreement_of(trace_record("A30"), trace_record("A100", workspace=0.0))
        entries = [e for e in agreement["shared"] if e["name"] == WORKSPACE_NAME]

        assert [e["distinct"] for e in entries] == [2]

    def test_it_warns_when_the_runs_did_not_share_an_image(self) -> None:
        other = trace_record("A100")
        other["fingerprint"]["image_digest"] = "sha256:something-else"

        lines = report_cli.report_lines((("a30.json", trace_record("A30")), ("a100.json", other)))

        assert [line for line in lines if "do not all share one image_digest" in line] != []

    def test_observations_only_some_runs_carry_are_reported_not_dropped(self) -> None:
        # A module graph that differs between runs shows up here. Silently
        # comparing the intersection would report agreement over the tensors
        # that happen to pair and say nothing about the ones that do not.
        lines = report_cli.report_lines(
            (
                ("a30.json", trace_record("A30")),
                ("a100.json", trace_record("A100", steps=len(GRAPH) - 1)),
            )
        )

        assert [line for line in lines if "observations not reported by every run" in line] != []

    def test_one_run_alone_is_refused_because_agreement_needs_a_set(self) -> None:
        with pytest.raises(ValueError, match="at least two runs"):
            report_cli.report_lines((("a30.json", trace_record("A30")),))


class TestTheCommandLine:
    def test_main_prints_a_report_for_a_directory(self, tmp_path: pathlib.Path) -> None:
        directory = write_records(
            tmp_path / "trace",
            {"a30": trace_record("A30"), "a100": trace_record("A100", diverge_from=2)},
        )

        assert report_cli.main(["--dir", str(directory)]) == 0

    def test_a_missing_directory_is_refused_by_name(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError, match="no such directory"):
            report_cli.main(["--dir", str(tmp_path / "absent")])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--rung"):
            report_cli.main(["--dir", str(tmp_path), "--rung", "tiny"])

    def test_the_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        directory = write_records(
            tmp_path / "trace", {"a30": trace_record("A30"), "a100": trace_record("A100")}
        )
        saved = sys.argv
        sys.argv = ["modeltrainer-probe-trace-report", "--dir", str(directory)]
        try:
            with pytest.raises(SystemExit) as raised:
                report_cli.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_running_the_module_as_main_actually_reports(self, tmp_path: pathlib.Path) -> None:
        directory = write_records(
            tmp_path / "trace", {"a30": trace_record("A30"), "a100": trace_record("A100")}
        )
        module_name = "model_trainer.cli.probe_trace_report"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["modeltrainer-probe-trace-report", "--dir", str(directory)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
