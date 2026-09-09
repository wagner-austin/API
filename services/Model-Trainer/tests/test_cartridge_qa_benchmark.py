"""The question-set entry, exercised on a real model over a fake corpus.

WHAT IS REAL AND WHAT IS NOT. The arms are real: a real GPT-2 is built, a real
cartridge is drawn and trained, real items are built from held-out text, and
real cloze scoring runs. Faked are the two seams that would otherwise need a
model cache and a wiki checkout -- the hub loader and the corpus reader --
plus the plan table, because the real plan trains three cartridges over a 124M
base.

THE ASSERTIONS THAT MATTER are about what the record can be read as later. An
arm scored on items another arm never saw, a spread reported without the mean
it qualifies, or a base arm re-scored per seed and its spread reported as
measured, would all produce a file that looks complete and says something
false.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
from platform_core.json_utils import load_json_str
from platform_core.run_record import NO_PAYLOAD, decode_run_record
from platform_ml.determinism import (
    ATTENTION_MATH_ONLY,
    ATTENTION_SETTING,
    SPLIT_K_REMOVED,
    SPLIT_K_SETTING,
)

from model_trainer.cli import cartridge_qa_benchmark as bench
from model_trainer.core.services.model.cartridge_qa_plans import QA_EXPERIMENT
from tests._qa_benchmark_support import (
    DOCUMENTS as _DOCUMENTS,
)
from tests._qa_benchmark_support import (
    TINY_PLAN,
    install_fakes,
    restore_fakes,
)
from tests._qa_benchmark_support import (
    Tokenizer as _Tokenizer,
)
from tests._qa_benchmark_support import (
    values as _values,
)


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the shared fakes, and put the real hooks back afterwards."""
    install_fakes()
    yield None
    restore_fakes()


class TestBuildQuestionSet:
    def test_items_come_from_held_out_text_and_answers_from_training_text(self) -> None:
        """The property that keeps this from measuring memorisation.

        A cartridge trains on the training windows; if items were built from
        those, it would win by recalling the sentence. An answer that appears
        in no training window would instead be unanswerable from the corpus.
        """
        tokenizer = _Tokenizer()
        encoded = [tokenizer.encode(document) for document in _DOCUMENTS]

        items, training_text = bench.build_question_set(
            _DOCUMENTS, encoded, bench.HFTokenizerEncoder(tokenizer), TINY_PLAN
        )

        assert items != []
        for item in items:
            assert item["answer"] in training_text

    def test_the_training_text_excludes_the_held_out_windows(self) -> None:
        tokenizer = _Tokenizer()
        encoded = [tokenizer.encode(document) for document in _DOCUMENTS]

        _items, training_text = bench.build_question_set(
            _DOCUMENTS, encoded, bench.HFTokenizerEncoder(tokenizer), TINY_PLAN
        )

        window = TINY_PLAN["window"]
        first_held = tokenizer.decode(encoded[0][:window])
        assert first_held not in training_text


class TestMeasureQaPlan:
    def test_every_observation_is_named_once(self, tmp_path: pathlib.Path) -> None:
        observations = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")[
            "observations"
        ]

        names = [observation["name"] for observation in observations]
        assert len(names) == len(set(names))

    def test_it_names_both_instruments_and_the_baseline(self, tmp_path: pathlib.Path) -> None:
        """Both, because they were measured to disagree.

        On gpt2 the accuracy arm did not move while the answer-likelihood arm
        halved; a record carrying only one would report half the finding.
        """
        observations = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")[
            "observations"
        ]

        named = _values(observations)
        assert "base_accuracy" in named
        assert "retrieval_accuracy" in named
        assert "chance_accuracy" in named
        assert "cartridge-accuracy-gain_mean" in named
        assert "cartridge-accuracy-gain_spread" in named
        assert "cartridge-answer-nll-gain_mean" in named
        assert "cartridge-answer-nll-gain_spread" in named

    def test_chance_follows_the_distractor_count(self, tmp_path: pathlib.Path) -> None:
        observations = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")[
            "observations"
        ]

        named = _values(observations)
        assert named["chance_accuracy"] == pytest.approx(1.0 / (TINY_PLAN["distractor_count"] + 1))

    def test_every_gain_carries_a_spread_beside_its_mean(self, tmp_path: pathlib.Path) -> None:
        """A mean without its spread is what let a 0.02 difference read as a finding."""
        observations = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")[
            "observations"
        ]

        named = _values(observations)
        for arm in ("cartridge-accuracy-gain", "cartridge-answer-nll-gain"):
            assert f"{arm}_mean" in named
            assert f"{arm}_spread" in named
            assert named[f"{arm}_spread"] >= 0.0

    def test_the_retrieval_gain_is_the_difference_it_claims_to_be(
        self, tmp_path: pathlib.Path
    ) -> None:
        observations = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")[
            "observations"
        ]

        named = _values(observations)
        assert named["retrieval_accuracy_gain"] == pytest.approx(
            named["retrieval_accuracy"] - named["base_accuracy"]
        )

    def test_the_item_count_is_reported(self, tmp_path: pathlib.Path) -> None:
        """A gain over six items and one over six hundred read very differently."""
        observations = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")[
            "observations"
        ]

        named = _values(observations)
        assert 0.0 < named["items"] <= float(TINY_PLAN["max_items"])


class TestRunRecord:
    def test_it_carries_the_question_set_experiment(self, tmp_path: pathlib.Path) -> None:
        """Not the loss experiment's, so the two cannot be differenced."""
        record = bench.qa_run_record(
            "tiny", corpus=tmp_path, device="cpu", remove_split_k=False, math_attention=False
        )

        assert record["experiment"] == QA_EXPERIMENT
        assert record["label"].startswith(
            "tiny-tiny-under-test-w8-s2-c8-m48-e1-lr0.05-d2-n6-seeds7.8.9-"
        )

    def test_the_payload_digest_is_the_question_set_this_run_asked(
        self, tmp_path: pathlib.Path
    ) -> None:
        """THE FIELD THAT SEPARATES TWO RUNS OF ONE PLAN.

        `qa-record.json` and `qa-svc-gpt2.json` share an experiment, a label
        and a fingerprint while measuring 24 and 32 questions, because this
        field held :data:`~platform_core.run_record.NO_PAYLOAD` in both. It is
        asserted against the digest of the items the same plan builds rather
        than against a constant, so a change in how items are derived moves
        both sides and this test keeps checking the wiring rather than a
        frozen hash.
        """
        measured = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")
        record = bench.qa_run_record(
            "tiny", corpus=tmp_path, device="cpu", remove_split_k=False, math_attention=False
        )

        assert record["payload_digest"] == measured["question_set_digest"]
        assert record["payload_digest"] != NO_PAYLOAD
        # The corpus digest is in the LABEL, so the payload must not repeat
        # it: two digests that always agree are one digest.
        assert measured["corpus_digest"] not in record["payload_digest"]

    def test_an_unknown_plan_names_the_known_ones(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="tiny"):
            bench.qa_run_record(
                "no-such-plan",
                corpus=tmp_path,
                device="cpu",
                remove_split_k=False,
                math_attention=False,
            )

    def test_the_treated_arm_is_recorded_in_the_fingerprint(self, tmp_path: pathlib.Path) -> None:
        """A treated record must not be mistakable for an untreated one."""
        treated = bench.qa_run_record(
            "tiny", corpus=tmp_path, device="cpu", remove_split_k=True, math_attention=True
        )

        settings = dict(treated["fingerprint"]["determinism"]["settings"])
        assert settings[SPLIT_K_SETTING] == SPLIT_K_REMOVED
        assert settings[ATTENTION_SETTING] == ATTENTION_MATH_ONLY


class TestTheCommandLine:
    def _argv(self, tmp_path: pathlib.Path) -> list[str]:
        return [
            "--plan",
            "tiny",
            "--corpus",
            str(tmp_path),
            "--device",
            "cpu",
            "--controls",
            "none",
            "--out",
            str(tmp_path / "nested" / "record.json"),
        ]

    def test_it_writes_a_decodable_record(self, tmp_path: pathlib.Path) -> None:
        code = bench.main(self._argv(tmp_path))

        assert code == 0
        restored = decode_run_record(
            load_json_str((tmp_path / "nested" / "record.json").read_text(encoding="utf-8"))
        )
        assert restored["experiment"] == QA_EXPERIMENT

    def test_a_missing_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--controls"):
            bench.main(["--plan", "tiny", "--corpus", str(tmp_path), "--device", "cpu"])

    def test_the_output_path_is_still_required_once_the_arm_is_given(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The controls flag is parsed first, so keep the --out refusal covered."""
        with pytest.raises(ValueError, match="--out"):
            bench.main(
                [
                    "--plan",
                    "tiny",
                    "--corpus",
                    str(tmp_path),
                    "--device",
                    "cpu",
                    "--controls",
                    "none",
                ]
            )

    def test_the_console_entry_point_exits_zero(self, tmp_path: pathlib.Path) -> None:
        saved = sys.argv
        sys.argv = ["modeltrainer-cartridge-qa", *self._argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                bench.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_running_it_as_a_module_actually_measures(self, tmp_path: pathlib.Path) -> None:
        """Without the __main__ guard the module imports, runs nothing, exits 0."""
        module_name = "model_trainer.cli.cartridge_qa_benchmark"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", *self._argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        assert (tmp_path / "nested" / "record.json").is_file()
