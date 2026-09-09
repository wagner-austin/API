"""The corpus-representation entry, over a fake corpus and a real model.

WHAT IS FAKED AND WHY. The hub loader returns a REAL tiny GPT-2 and the plan
table is stood in for, exactly as the question-set benchmark's tests do: the
production plan edits a 124-million-parameter base once per accepted triple,
each edit preceded by an optimisation, so a suite that could only reach the
real table would either run it or leave this path uncovered.

WHAT IS NOT FAKED is the refusal this entry exists to make. The curated
triples are grounded in one exact corpus, so the digest check is the thing
standing between a reject rate and a reject rate about the wrong text, and it
runs against the real digest of the real fake corpus rather than a stub.

THE EDITED PATH IS COVERED BY `test_triple_edit_arm.py`, not here. A plan
whose triples are all rejected reaches every line of this module and leaves
the model untouched; a plan whose triples were accepted would additionally
spend an optimisation per triple to exercise arithmetic that has its own
tests. What is asserted here is the plumbing: the record's shape, its
identity, and the refusals.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator, Mapping

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record

from model_trainer.cli import _measurement_hooks, _test_hooks
from model_trainer.cli import triple_edit_benchmark as bench
from model_trainer.core.services.model.cartridge_plans import corpus_digest
from model_trainer.core.services.model.editing.grounding import TripleCandidate
from model_trainer.core.services.model.editing.triple_edit_plans import (
    TRIPLE_EDIT_EXPERIMENT,
    TripleEditPlan,
)
from tests._qa_benchmark_support import DOCUMENTS, install_fakes, restore_fakes

#: Rejected by the gate: its sentence is in no training text. Chosen so this
#: module never spends an optimisation, for the reason the docstring gives.
_UNGROUNDED = TripleCandidate(
    item_id="t00",
    subject="NavProbe",
    relation="was never measured beside",
    object="ClearGBM",
    source_document="fake.md",
    source_sentence="A sentence that appears in no training text at all here.",
)

_TINY_PLAN: TripleEditPlan = {
    "qa_plan": "tiny",
    "corpus_digest": corpus_digest(DOCUMENTS),
    "candidates": (_UNGROUNDED,),
    "site": {
        "layer": 0,
        "module_template": "transformer.h.{}.mlp.c_proj",
        "fact_token": "subject_last",
    },
    "value_steps": 2,
    "value_learning_rate": 0.1,
}

_WRONG_CORPUS_PLAN: TripleEditPlan = {**_TINY_PLAN, "corpus_digest": "0" * 64}


def _fake_plans() -> Mapping[str, TripleEditPlan]:
    """Stand in for the production triple-edit table.

    Returns:
        One runnable plan and one pinned to a corpus that is not this one.
    """
    return {"tiny-triples": _TINY_PLAN, "tiny-wrong-corpus": _WRONG_CORPUS_PLAN}


@pytest.fixture(autouse=True)
def _fakes() -> Generator[None, None, None]:
    """Point every seam at a fake for the duration of one test."""
    install_fakes()
    _measurement_hooks.triple_edit_plans = _fake_plans
    yield
    _measurement_hooks.triple_edit_plans = _measurement_hooks._default_triple_edit_plans
    restore_fakes()


def _argv(tmp_path: pathlib.Path, plan: str = "tiny-triples") -> list[str]:
    """Build a command line for one run.

    Args:
        tmp_path: Where the record is written.
        plan: Which plan to ask for.

    Returns:
        The arguments, excluding the program name.
    """
    return [
        "--plan",
        plan,
        "--corpus",
        str(tmp_path),
        "--device",
        "cpu",
        "--controls",
        "none",
        "--out",
        str(tmp_path / "nested" / "record.json"),
    ]


class TestTheRecord:
    def test_it_writes_a_decodable_record(self, tmp_path: pathlib.Path) -> None:
        code = bench.main(_argv(tmp_path))

        assert code == 0
        restored = decode_run_record(
            load_json_str((tmp_path / "nested" / "record.json").read_text(encoding="utf-8"))
        )
        assert restored["experiment"] == TRIPLE_EDIT_EXPERIMENT

    def test_the_payload_digest_names_the_question_set_it_asked(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The field that separates two runs of one plan over different item
        sets. Empty here would put this arm back where the question-set arm
        was before its own digest existed.
        """
        record = bench.triple_edit_run_record(
            "tiny-triples",
            corpus=tmp_path,
            device="cpu",
            remove_split_k=False,
            math_attention=False,
        )

        assert record["payload_digest"].startswith("sha256:")

    def test_the_label_carries_the_dose_and_the_site(self, tmp_path: pathlib.Path) -> None:
        record = bench.triple_edit_run_record(
            "tiny-triples",
            corpus=tmp_path,
            device="cpu",
            remove_split_k=False,
            math_attention=False,
        )

        assert record["label"].startswith("tiny-triples-tiny-L0-subject_last-vs2-vlr0.1-")

    def test_a_run_that_accepted_nothing_reports_the_base_unchanged(
        self, tmp_path: pathlib.Path
    ) -> None:
        record = bench.triple_edit_run_record(
            "tiny-triples",
            corpus=tmp_path,
            device="cpu",
            remove_split_k=False,
            math_attention=False,
        )
        named = {
            observation["name"]: observation["value"] for observation in record["observations"]
        }

        assert named["edits_applied"] == 0.0
        assert named["triples_rejected"] == 1.0
        assert named["edited_accuracy"] == named["base_accuracy"]

    def test_the_treated_arm_is_recorded_in_the_fingerprint(self, tmp_path: pathlib.Path) -> None:
        """A treated record must not be mistakable for an untreated one."""
        treated = bench.triple_edit_run_record(
            "tiny-triples",
            corpus=tmp_path,
            device="cpu",
            remove_split_k=True,
            math_attention=True,
        )

        assert treated["fingerprint"]["determinism"]["settings"]


class TestTheRefusals:
    def test_a_corpus_the_triples_were_not_curated_against_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        """THE CHECK THIS ENTRY EXISTS TO MAKE.

        A reject rate is a claim about one exact text. Measured against
        another, it is a number about a corpus nobody curated for -- and it
        would look exactly like a result.
        """
        with pytest.raises(AppError) as raised:
            bench.triple_edit_run_record(
                "tiny-wrong-corpus",
                corpus=tmp_path,
                device="cpu",
                remove_split_k=False,
                math_attention=False,
            )

        assert raised.value.code is ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE

    def test_an_unknown_plan_names_the_known_ones(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="tiny-triples"):
            bench.triple_edit_run_record(
                "no-such-plan",
                corpus=tmp_path,
                device="cpu",
                remove_split_k=False,
                math_attention=False,
            )

    def test_a_missing_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError):
            bench.main(["--plan", "tiny-triples", "--device", "cpu"])


class TestTheEntrypoint:
    def test_it_exits_with_the_command_s_code(self, tmp_path: pathlib.Path) -> None:
        """A console script that swallowed the code would report success from
        a run that wrote nothing.
        """
        import sys

        argv = ["triple-edit-benchmark", *_argv(tmp_path)]
        original = sys.argv
        sys.argv = argv
        try:
            with pytest.raises(SystemExit) as raised:
                bench.entrypoint()
        finally:
            sys.argv = original

        assert raised.value.code == 0

    def test_running_it_as_a_module_actually_measures(self, tmp_path: pathlib.Path) -> None:
        """Without the ``__main__`` guard the module imports, runs nothing and
        exits 0 -- which is indistinguishable from a successful measurement.
        """
        import runpy
        import sys

        module_name = "model_trainer.cli.triple_edit_benchmark"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        assert (tmp_path / "nested" / "record.json").is_file()

    def test_the_corpus_reader_is_the_hook_this_suite_installed(self) -> None:
        """A module-level import of the reader would make the seam unusable,
        which is how a benchmark ends up reading a real wiki inside a suite.
        Checked by IDENTITY against the fake, not by non-emptiness.
        """
        from tests._qa_benchmark_support import _fake_corpus_reader

        assert _test_hooks.read_corpus_documents is _fake_corpus_reader
