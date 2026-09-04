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
from collections.abc import Generator, Mapping, Sequence

import pytest
from platform_core.json_utils import load_json_str
from platform_core.run_record import Observation, decode_run_record
from platform_ml.determinism import (
    ATTENTION_MATH_ONLY,
    ATTENTION_SETTING,
    SPLIT_K_REMOVED,
    SPLIT_K_SETTING,
)

from model_trainer.cli import _measurement_hooks, _test_hooks
from model_trainer.cli import cartridge_qa_benchmark as bench
from model_trainer.core.contracts.model import QuantizationConfig
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_qa_plans import QA_EXPERIMENT, QaPlan
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import LMModelProto

#: A plan small enough to run in a test and shaped like the real one.
#:
#: Sized against the tiny rung's 64 positions, which is the binding
#: constraint: a 48-token budget plus an 8-slot prefix is 56, leaving room for
#: an item and some evidence without reaching the position embedding's end.
TINY_PLAN: QaPlan = {
    "model_id": "tiny-under-test",
    "window": 8,
    "held_out_stride": 2,
    "num_slots": 8,
    "max_seq_len": 48,
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
    "distractor_count": 2,
    "max_items": 6,
}

#: Four documents, each naming its own subject in several sentences.
#:
#: FOUR RATHER THAN TWO because a distractor may not be a term from the item's
#: own document: with two documents an item could draw only one distractor,
#: and the builder correctly refuses. Each subject recurs so that it lands in
#: both a held-out window and a training one, which is what makes its item
#: answerable from the corpus rather than a guess.
#: NO SENTENCE BEGINS WITH ITS SUBJECT, and that is a constraint of the fake
#: tokenizer rather than of the corpus. It is word-level, so each name is one
#: token; a name at position zero is the sequence's first token, which no
#: causal model can score because nothing precedes it, and `answer_nll`
#: correctly refuses the item. Real byte-pair encoding splits these names into
#: several tokens and the question does not arise.
_DOCUMENTS: tuple[str, ...] = tuple(
    (
        f"The engine called {name} rebuilt the measurement path inside one core. "
        f"A later pass moved {name} onto a faster route for speed and memory. "
        f"The team measured {name} against the usual baseline over many weeks. "
        f"Written notes about {name} explain the design in considerable detail."
    )
    for name in ("ClearGBM", "TankpitBot", "NavProbe", "CoverGate")
)


class _Tokenizer:
    """A reversible word-level tokenizer inside the tiny rung's vocabulary.

    WORD-LEVEL RATHER THAN CHARACTER-LEVEL, and the reason is a real
    constraint rather than convenience. The tiny rung has 64 positions, the
    item builder requires sentences of at least
    :data:`~corpus_cloze.MIN_SENTENCE_CHARS` characters, and a character
    tokenizer turns such a sentence into sixty-odd tokens -- more than the
    whole window, before any evidence. `with_evidence` then correctly refuses
    every item, and the test measures nothing.

    REVERSIBLE because the pipeline decodes windows back to text to build
    items from them. A hashing tokenizer would encode fine and decode to
    nothing, so the vocabulary is kept both ways and grown on demand.
    """

    _to_id: dict[str, int]
    _to_word: dict[int, str]

    def __init__(self) -> None:
        self._to_id = {}
        self._to_word = {}

    @property
    def eos_token_id(self) -> int | None:
        return 0

    @property
    def pad_token_id(self) -> int | None:
        return 1

    def __len__(self) -> int:
        return PROBE_SHAPES["tiny"]["vocab_size"]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            known = self._to_id.get(word)
            if known is None:
                # Ids start at 2 so neither collides with eos or pad, and stay
                # inside the rung's vocabulary or the embedding lookup fails.
                known = len(self._to_id) + 2
                assert known < len(self), "the fake corpus outgrew the tiny vocabulary"
                self._to_id[word] = known
                self._to_word[known] = word
            ids.append(known)
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._to_word[value] for value in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.encode(token)[0] if token.split() else 0


def _fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader."""
    assert model_id_or_path == TINY_PLAN["model_id"]
    return _Tokenizer()


def _fake_model(model_id_or_path: str, quantization: QuantizationConfig | None) -> LMModelProto:
    """Stand in for the hub model loader, returning a real tiny GPT-2.

    The model is real; only its provenance is faked, so the arms run real
    attention without needing a cache.
    """
    assert model_id_or_path == TINY_PLAN["model_id"]
    assert quantization is None
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


def _fake_plans() -> Mapping[str, QaPlan]:
    """Stand in for the production plan table, with one runnable plan."""
    return {"tiny": TINY_PLAN}


def _fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Stand in for the corpus reader."""
    return _DOCUMENTS


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards."""
    _measurement_hooks.qa_plans = _fake_plans
    _test_hooks.read_corpus_documents = _fake_corpus_reader
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model
    yield None
    _measurement_hooks.qa_plans = _measurement_hooks._default_qa_plans
    _test_hooks.read_corpus_documents = _test_hooks._default_read_corpus_documents
    hf_hooks.Hooks.reset()


def _values(observations: Sequence[Observation]) -> dict[str, float]:
    """Read a record's observations into a name-to-value mapping.

    Args:
        observations: The observations to read.

    Returns:
        Each observation's value, keyed by its name.
    """
    return {observation["name"]: observation["value"] for observation in observations}


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
        observations, _digest = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")

        names = [observation["name"] for observation in observations]
        assert len(names) == len(set(names))

    def test_it_names_both_instruments_and_the_baseline(self, tmp_path: pathlib.Path) -> None:
        """Both, because they were measured to disagree.

        On gpt2 the accuracy arm did not move while the answer-likelihood arm
        halved; a record carrying only one would report half the finding.
        """
        observations, _digest = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")

        named = _values(observations)
        assert "base_accuracy" in named
        assert "retrieval_accuracy" in named
        assert "chance_accuracy" in named
        assert "cartridge-accuracy-gain_mean" in named
        assert "cartridge-accuracy-gain_spread" in named
        assert "cartridge-answer-nll-gain_mean" in named
        assert "cartridge-answer-nll-gain_spread" in named

    def test_chance_follows_the_distractor_count(self, tmp_path: pathlib.Path) -> None:
        observations, _digest = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")

        named = _values(observations)
        assert named["chance_accuracy"] == pytest.approx(1.0 / (TINY_PLAN["distractor_count"] + 1))

    def test_every_gain_carries_a_spread_beside_its_mean(self, tmp_path: pathlib.Path) -> None:
        """A mean without its spread is what let a 0.02 difference read as a finding."""
        observations, _digest = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")

        named = _values(observations)
        for arm in ("cartridge-accuracy-gain", "cartridge-answer-nll-gain"):
            assert f"{arm}_mean" in named
            assert f"{arm}_spread" in named
            assert named[f"{arm}_spread"] >= 0.0

    def test_the_retrieval_gain_is_the_difference_it_claims_to_be(
        self, tmp_path: pathlib.Path
    ) -> None:
        observations, _digest = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")

        named = _values(observations)
        assert named["retrieval_accuracy_gain"] == pytest.approx(
            named["retrieval_accuracy"] - named["base_accuracy"]
        )

    def test_the_item_count_is_reported(self, tmp_path: pathlib.Path) -> None:
        """A gain over six items and one over six hundred read very differently."""
        observations, _digest = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")

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
