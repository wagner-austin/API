"""The cartridge measurement entry, exercised on a real model over fake corpora.

WHAT IS REAL HERE AND WHAT IS NOT. The arms are real: a real GPT-2 is built,
real cartridges are drawn and trained, real held-out items are scored, and the
record is assembled from the numbers that came back. What is faked is the two
seams that would otherwise need a model cache and a wiki checkout -- the hub
loader and the corpus reader -- plus the plan table, because every plan in the
real one is minutes of GPU per arm.

THE ASSERTION THAT MATTERS MOST is the one about the record's names. A record
carries floats, and a reader six months from now has only the observation
names to tell them what was measured. A sweep whose two points collided under
one name, or a separation verdict recorded without the floor it was judged
against, would produce a file that looks complete and cannot be read.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator, Mapping

import pytest
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record
from platform_ml.determinism import (
    ATTENTION_MATH_ONLY,
    ATTENTION_SETTING,
    SPLIT_K_REMOVED,
    SPLIT_K_SETTING,
)

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import cartridge_benchmark as bench
from model_trainer.core.contracts.model import QuantizationConfig
from model_trainer.core.contracts.replicated_measurement import replicate
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_plans import CARTRIDGE_EXPERIMENT, CartridgePlan
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import LMModelProto
from tests.core.services.model.backends.hf_lm.testing import FakeHFTokenizer

#: A plan small enough to run in a test and shaped like the real one: a sweep
#: with two points so the separation logic has a pair to compare, a
#: composition arm, and three seeds because fewer is refused.
TINY_PLAN: CartridgePlan = {
    "model_id": "tiny-under-test",
    "window": 8,
    "held_out_stride": 3,
    "slot_counts": (2, 4),
    "composition_slots": 4,
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
}


#: Vocabulary the fake tokenizer encodes into. Matched to the tiny rung's own
#: so no document can produce an id past the embedding table's last row.
_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]


def _fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader.

    Returns the suite's existing `FakeHFTokenizer` rather than a private copy:
    it already encodes text as ``ord(c) % vocab_size``, which is exactly what
    a measurement needs from a tokenizer -- text in, distinct ids out -- and a
    second spelling of it here would be free to drift from the one every other
    backend test uses.
    """
    assert model_id_or_path == TINY_PLAN["model_id"]
    return FakeHFTokenizer(vocab_size=_VOCAB)


def _fake_model(model_id_or_path: str, quantization: QuantizationConfig | None) -> LMModelProto:
    """Stand in for the hub model loader, returning a real tiny GPT-2.

    The model is real. Only its PROVENANCE is faked: it comes from the
    deterministic init the probe uses rather than from the hub, so the test
    needs no cache and the arms still run real attention.
    """
    assert model_id_or_path == TINY_PLAN["model_id"]
    assert quantization is None
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


def _documents(marker: str) -> tuple[str, ...]:
    """Build a corpus long enough to window and split.

    Args:
        marker: Character that makes this corpus different from another.

    Returns:
        Four documents of 24 characters each, giving twelve windows of eight.
    """
    return tuple(f"{marker}{index}" * 12 for index in range(4))


def _fake_plans() -> Mapping[str, CartridgePlan]:
    """Stand in for the production plan table, with one runnable plan."""
    return {"tiny": TINY_PLAN}


def _fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Stand in for the corpus reader, keyed on the directory's own name.

    Two different directories must give two different corpora, because the
    composition arm's whole point is that its second cartridge carries
    something the first does not.
    """
    return _documents(corpus_dir.name[0])


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the three fakes, and put the real hooks back afterwards."""
    measurement_hooks.cartridge_plans = _fake_plans
    cli_hooks.read_corpus_documents = _fake_corpus_reader
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model
    yield None
    measurement_hooks.cartridge_plans = measurement_hooks._default_cartridge_plans
    cli_hooks.read_corpus_documents = cli_hooks._default_read_corpus_documents
    hf_hooks.Hooks.reset()


class TestSweepObservations:
    def test_each_step_is_named_by_the_pair_it_compares(self) -> None:
        sweep = [
            replicate("slots-2", [(7, 0.70), (8, 0.71), (9, 0.72)]),
            replicate("slots-8", [(7, 0.90), (8, 0.91), (9, 0.92)]),
        ]

        named = bench.sweep_observations(sweep, 0.05)

        assert named == (
            {"name": "slots-2_to_slots-8_difference", "value": pytest.approx(0.20)},
            {"name": "slots-2_to_slots-8_separated", "value": 1.0},
        )

    def test_a_step_inside_the_noise_records_a_zero(self) -> None:
        """Recorded as a number, not omitted.

        An absent observation reads as "not measured"; a zero reads as
        "measured, and it did not clear the floor", which is the finding.
        """
        sweep = [
            replicate("slots-128", [(7, 0.90), (8, 0.91), (9, 0.92)]),
            replicate("slots-512", [(7, 0.91), (8, 0.92), (9, 0.93)]),
        ]

        named = bench.sweep_observations(sweep, 0.05)

        assert named[1] == {"name": "slots-128_to_slots-512_separated", "value": 0.0}

    def test_a_one_point_sweep_compares_nothing(self) -> None:
        assert (
            bench.sweep_observations([replicate("only", [(7, 0.1), (8, 0.1), (9, 0.1)])], 0.05)
            == ()
        )


class TestMeasurePlan:
    def test_every_arm_is_named_once(self, tmp_path: pathlib.Path) -> None:
        """Two arms sharing an observation name would silently overwrite.

        The sweep points differ only in slot count, so this is the collision
        that would actually happen.
        """
        first = tmp_path / "alpha"
        second = tmp_path / "beta"
        first.mkdir()
        second.mkdir()

        observations, _digest = bench.measure_plan(
            TINY_PLAN, corpus=first, second_corpus=second, device="cpu"
        )

        names = [observation["name"] for observation in observations]
        assert len(names) == len(set(names))

    def test_it_names_the_floor_and_the_retention(self, tmp_path: pathlib.Path) -> None:
        first = tmp_path / "alpha"
        second = tmp_path / "beta"
        first.mkdir()
        second.mkdir()

        observations, _digest = bench.measure_plan(
            TINY_PLAN, corpus=first, second_corpus=second, device="cpu"
        )

        named = {observation["name"] for observation in observations}
        assert "sweep_noise_floor" in named
        assert "composition_noise_floor" in named
        assert "composition_retention" in named
        assert "untrained-slots-4_mean" in named
        assert "slots-2_mean" in named
        assert "slots-4_spread" in named
        assert "composition-alone_mean" in named
        assert "composition-composed_mean" in named

    def test_the_sweep_floor_comes_from_the_sweep_arms_alone(self, tmp_path: pathlib.Path) -> None:
        """The regression guard for a defect that shipped in the first record.

        The floor was once the largest spread of ANY arm, and the largest
        belonged to the composed arm -- which trains two cartridges over a
        doubled prefix and is noisier for reasons unrelated to the sweep. At
        0.0671 it buried a sweep step of +0.0584 that two independent runs had
        found real.
        """
        first = tmp_path / "alpha"
        second = tmp_path / "beta"
        first.mkdir()
        second.mkdir()

        observations, _digest = bench.measure_plan(
            TINY_PLAN, corpus=first, second_corpus=second, device="cpu"
        )
        values = {observation["name"]: observation["value"] for observation in observations}

        assert values["sweep_noise_floor"] == pytest.approx(
            max(values[f"slots-{count}_spread"] for count in TINY_PLAN["slot_counts"])
        )
        assert values["composition_noise_floor"] == pytest.approx(
            max(values["composition-alone_spread"], values["composition-composed_spread"])
        )

    def test_the_digest_is_of_the_primary_corpus(self, tmp_path: pathlib.Path) -> None:
        """Not of the second, which varies independently and is not what the label names."""
        first = tmp_path / "alpha"
        second = tmp_path / "beta"
        first.mkdir()
        second.mkdir()

        _observations, digest = bench.measure_plan(
            TINY_PLAN, corpus=first, second_corpus=second, device="cpu"
        )

        from model_trainer.core.services.model.cartridge_plans import corpus_digest

        assert digest == corpus_digest(_documents("a"))


class TestRunRecord:
    def test_it_carries_the_experiment_and_a_corpus_stamped_label(
        self, tmp_path: pathlib.Path
    ) -> None:
        first = tmp_path / "alpha"
        second = tmp_path / "beta"
        first.mkdir()
        second.mkdir()

        record = bench.cartridge_run_record(
            "tiny",
            corpus=first,
            second_corpus=second,
            device="cpu",
            remove_split_k=False,
            math_attention=False,
        )

        assert record["experiment"] == CARTRIDGE_EXPERIMENT
        assert record["label"].startswith("tiny-tiny-under-test-w8-s3-e1-lr0.05-slots2.4-c4-seeds")

    def test_an_unknown_plan_names_the_known_ones(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="tiny"):
            bench.cartridge_run_record(
                "no-such-plan",
                corpus=tmp_path,
                second_corpus=tmp_path,
                device="cpu",
                remove_split_k=False,
                math_attention=False,
            )

    def test_the_treated_arm_is_recorded_in_the_fingerprint(self, tmp_path: pathlib.Path) -> None:
        """THE POINT OF THE FLAG.

        A record measured under the controls has to be distinguishable from
        one measured without them, or the cross-card comparison this exists
        for would silently difference two different configurations. The
        settings keys are written only when the control was applied, so their
        presence is the evidence.
        """
        first = tmp_path / "alpha"
        second = tmp_path / "beta"
        first.mkdir()
        second.mkdir()

        treated = bench.cartridge_run_record(
            "tiny",
            corpus=first,
            second_corpus=second,
            device="cpu",
            remove_split_k=True,
            math_attention=True,
        )
        untreated = bench.cartridge_run_record(
            "tiny",
            corpus=first,
            second_corpus=second,
            device="cpu",
            remove_split_k=False,
            math_attention=False,
        )

        treated_settings = dict(treated["fingerprint"]["determinism"]["settings"])
        untreated_settings = dict(untreated["fingerprint"]["determinism"]["settings"])
        assert treated_settings[SPLIT_K_SETTING] == SPLIT_K_REMOVED
        assert treated_settings[ATTENTION_SETTING] == ATTENTION_MATH_ONLY
        assert SPLIT_K_SETTING not in untreated_settings
        assert ATTENTION_SETTING not in untreated_settings


class TestMain:
    def test_it_writes_a_decodable_record(self, tmp_path: pathlib.Path) -> None:
        first = tmp_path / "alpha"
        second = tmp_path / "beta"
        first.mkdir()
        second.mkdir()
        out = tmp_path / "nested" / "record.json"

        code = bench.main(
            [
                "--plan",
                "tiny",
                "--corpus",
                str(first),
                "--second-corpus",
                str(second),
                "--device",
                "cpu",
                "--controls",
                "none",
                "--out",
                str(out),
            ]
        )

        assert code == 0
        restored = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert restored["experiment"] == CARTRIDGE_EXPERIMENT

    def test_the_controls_arm_is_required(self, tmp_path: pathlib.Path) -> None:
        """Same reasoning as the second corpus: no default would be honest.

        A cartridge record whose posture was guessed names a condition it may
        not have run under, and the two arms produce different numbers -- that
        is the whole reason the flag exists.
        """
        first = tmp_path / "alpha"
        second = tmp_path / "beta"
        first.mkdir()
        second.mkdir()

        with pytest.raises(ValueError, match="--controls"):
            bench.main(
                [
                    "--plan",
                    "tiny",
                    "--corpus",
                    str(first),
                    "--second-corpus",
                    str(second),
                    "--device",
                    "cpu",
                    "--out",
                    str(tmp_path / "r.json"),
                ]
            )

    def test_an_unknown_controls_arm_names_the_known_ones(self, tmp_path: pathlib.Path) -> None:
        first = tmp_path / "alpha"
        second = tmp_path / "beta"
        first.mkdir()
        second.mkdir()

        with pytest.raises(ValueError, match="both"):
            bench.main(
                [
                    "--plan",
                    "tiny",
                    "--corpus",
                    str(first),
                    "--second-corpus",
                    str(second),
                    "--device",
                    "cpu",
                    "--controls",
                    "split-k-and-attention",
                    "--out",
                    str(tmp_path / "r.json"),
                ]
            )

    def test_the_second_corpus_is_required(self, tmp_path: pathlib.Path) -> None:
        """There is no default that would be honest.

        Composing two cartridges trained on two halves of ONE corpus measured
        94% retention, and the number was an artifact of each half already
        predicting the other. A default second corpus would make that the
        easy path.
        """
        with pytest.raises(ValueError, match="--second-corpus"):
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
                    "--out",
                    str(tmp_path / "r.json"),
                ]
            )


class TestInvocationForms:
    """The console script and `python -m` must do the same thing.

    A module under `cli/` that defines `entrypoint` but carries no
    `if __name__ == "__main__"` block is importable, runnable and does
    NOTHING -- exits 0, writes no file -- while the console script works. The
    two forms then disagree, and the broken one looks exactly like a
    measurement that legitimately produced nothing. That shape cost real time
    on 2026-08-27; both forms are exercised here so it cannot come back.
    """

    def _argv(self, tmp_path: pathlib.Path) -> list[str]:
        """Build a complete command line against two staged corpora.

        Args:
            tmp_path: The test's temporary directory.

        Returns:
            The flags, without a program name.
        """
        first = tmp_path / "alpha"
        second = tmp_path / "beta"
        first.mkdir()
        second.mkdir()
        return [
            "--plan",
            "tiny",
            "--corpus",
            str(first),
            "--second-corpus",
            str(second),
            "--device",
            "cpu",
            "--controls",
            "none",
            "--out",
            str(tmp_path / "record.json"),
        ]

    def test_the_console_entry_point_runs_and_exits_zero(self, tmp_path: pathlib.Path) -> None:
        saved = sys.argv
        sys.argv = ["modeltrainer-cartridge-benchmark", *self._argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                bench.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0
        assert (tmp_path / "record.json").is_file()

    def test_running_it_as_a_module_actually_measures(self, tmp_path: pathlib.Path) -> None:
        """And writes the record, which is the half that silently went missing."""
        module_name = "model_trainer.cli.cartridge_benchmark"
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
        restored = decode_run_record(
            load_json_str((tmp_path / "record.json").read_text(encoding="utf-8"))
        )
        assert restored["experiment"] == CARTRIDGE_EXPERIMENT
