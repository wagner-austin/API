"""The composition-scaling entry, exercised on a real model over fake corpora.

Same split as the two-cartridge benchmark's suite: the arms are real -- a
real tiny GPT-2, real cartridges trained and composed, real held-out scoring
-- and the faked seams are the hub loaders, the corpus reader and the plan
table, because the production plan is dozens of GPU-minutes of cartridges.

The assertions concentrate on the record's names and on the two properties
the sweep exists to guarantee: that the retention trend is judged against a
floor built from composed arms alone, and that the fixed-policy alone arms
agree exactly across compartment counts -- the internal replication check
that fails loudly if training ever stops being a function of its seed.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator, Mapping

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import cartridge_composition_sweep as sweep
from model_trainer.core.contracts.model import QuantizationConfig
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_measurement import (
    measure_composition_scaling,
)
from model_trainer.core.services.model.cartridge_plans import (
    COMPOSITION_SWEEP_EXPERIMENT,
    COMPOSITION_SWEEP_PLANS,
    CompositionSweepPlan,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import LMModelProto
from tests.core.services.model.backends.hf_lm.testing import FakeHFTokenizer

#: A plan small enough to run in a test and shaped like the real one: two
#: compartment counts so the trend logic has a pair, both policies exercised,
#: and three seeds because fewer is refused. Budget 6 divides both counts.
TINY_SWEEP_PLAN: CompositionSweepPlan = {
    "model_id": "tiny-under-test",
    "window": 8,
    "held_out_stride": 3,
    "compartment_counts": (2, 3),
    "fixed_slots": 2,
    "total_slot_budget": 6,
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
}

_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]


def _fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader."""
    assert model_id_or_path == TINY_SWEEP_PLAN["model_id"]
    return FakeHFTokenizer(vocab_size=_VOCAB)


def _fake_model(model_id_or_path: str, quantization: QuantizationConfig | None) -> LMModelProto:
    """Stand in for the hub model loader, returning a real tiny GPT-2."""
    assert model_id_or_path == TINY_SWEEP_PLAN["model_id"]
    assert quantization is None
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


def _documents(marker: str) -> tuple[str, ...]:
    """Four documents of 24 characters, giving twelve windows of eight.

    Args:
        marker: Character that makes this corpus different from another.

    Returns:
        The corpus bodies.
    """
    return tuple(f"{marker}{index}" * 12 for index in range(4))


def _fake_plans() -> Mapping[str, CompositionSweepPlan]:
    """Stand in for the production plan table, with one runnable plan."""
    return {"tiny": TINY_SWEEP_PLAN}


def _fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Stand in for the corpus reader, keyed on the directory's own name.

    A directory whose name starts with ``s`` yields one document rather than
    four, so a test can stage a corpus too short to match the primary's
    training set without inventing a second reader.
    """
    if corpus_dir.name.startswith("s"):
        return _documents(corpus_dir.name[0])[:1]
    return _documents(corpus_dir.name[0])


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards."""
    measurement_hooks.composition_sweep_plans = _fake_plans
    cli_hooks.read_corpus_documents = _fake_corpus_reader
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model
    yield None
    measurement_hooks.composition_sweep_plans = measurement_hooks._default_composition_sweep_plans
    cli_hooks.read_corpus_documents = cli_hooks._default_read_corpus_documents
    hf_hooks.Hooks.reset()


def _staged(tmp_path: pathlib.Path, names: tuple[str, ...]) -> list[pathlib.Path]:
    """Create one directory per corpus name.

    Args:
        tmp_path: The test's temporary directory.
        names: Directory names; the fake reader keys corpora on them.

    Returns:
        The created paths, in order.
    """
    created: list[pathlib.Path] = []
    for name in names:
        path = tmp_path / name
        path.mkdir()
        created.append(path)
    return created


class TestPolicySlots:
    def test_the_fixed_policy_ignores_the_count(self) -> None:
        assert sweep.policy_slots(TINY_SWEEP_PLAN, "fixed", 2) == 2
        assert sweep.policy_slots(TINY_SWEEP_PLAN, "fixed", 3) == 2

    def test_the_budget_policy_divides_the_total(self) -> None:
        assert sweep.policy_slots(TINY_SWEEP_PLAN, "budget", 2) == 3
        assert sweep.policy_slots(TINY_SWEEP_PLAN, "budget", 3) == 2

    def test_an_uneven_division_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not divide evenly"):
            sweep.policy_slots(TINY_SWEEP_PLAN, "budget", 4)

    def test_an_unknown_policy_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="fixed, budget"):
            sweep.policy_slots(TINY_SWEEP_PLAN, "typo", 2)


class TestMeasureCompositionScaling:
    """The measurement function itself, on the arms the CLI cannot reach."""

    def test_no_other_cartridges_composes_the_first_alone(self) -> None:
        """The empty composition is the first cartridge, exactly.

        ``functools.reduce`` over no others returns the first slots object
        untouched, so the alone and composed arms must agree to the digit.
        Covered directly because no plan asks for a one-compartment
        composition, and the branch would otherwise be dead code nobody
        measured.
        """
        from model_trainer.core.services.model.cartridge_corpus import (
            build_windows,
            split_by_stride,
        )

        tokenizer = FakeHFTokenizer(vocab_size=_VOCAB)
        encoded = [tokenizer.encode(document) for document in _documents("a")]
        train, held_out = split_by_stride(
            build_windows(encoded, window=8, device="cpu"), held_out_stride=3
        )
        model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        from model_trainer.core.services.finetuning.strategies.cartridge import (
            require_cache_capable,
        )

        alone, composed, untrained_composed, cross = measure_composition_scaling(
            require_cache_capable(model),
            first_train=train,
            other_trains=(),
            held_out=held_out,
            arm="solo-n1",
            num_slots=2,
            seeds=(7, 8, 9),
            epochs=1,
            learning_rate=0.05,
        )

        assert cross == ()
        assert composed["gains"] == alone["gains"]
        assert untrained_composed["gains"] == alone["gains"]
        assert alone["arm"] == "solo-n1-alone"
        assert composed["arm"] == "solo-n1-composed"
        assert untrained_composed["arm"] == "solo-n1-untrained-composed"


class TestMeasureSweep:
    def test_every_arm_is_named_once(self, tmp_path: pathlib.Path) -> None:
        primary, *others = _staged(tmp_path, ("alpha", "beta", "gamma"))

        observations, _digest = sweep.measure_sweep(
            TINY_SWEEP_PLAN, corpus=primary, other_corpora=others, device="cpu"
        )

        names = [observation["name"] for observation in observations]
        assert len(names) == len(set(names))

    def test_it_names_the_retentions_the_floors_and_the_cross_gains(
        self, tmp_path: pathlib.Path
    ) -> None:
        primary, *others = _staged(tmp_path, ("alpha", "beta", "gamma"))

        observations, _digest = sweep.measure_sweep(
            TINY_SWEEP_PLAN, corpus=primary, other_corpora=others, device="cpu"
        )

        named = {observation["name"] for observation in observations}
        assert "fixed-n2_retention" in named
        assert "budget-n3_retention" in named
        assert "fixed_composed_noise_floor" in named
        assert "budget_composed_noise_floor" in named
        assert "fixed-n2-cross-0_mean" in named
        assert "fixed-n3-cross-1_mean" in named
        assert "fixed-n2_slots_per_cartridge" in named
        assert "fixed-n2-composed_to_fixed-n3-composed_difference" in named
        assert "budget-n2-composed_to_budget-n3-composed_separated" in named
        assert "fixed-n2-untrained-composed_mean" in named
        assert "budget-n3_untrained_retention" in named
        assert "fixed-n2-composed_to_fixed-n2-untrained-composed_difference" in named
        assert "budget-n3-composed_to_budget-n3-untrained-composed_separated" in named

    def test_the_floor_comes_from_the_composed_arms_alone(self, tmp_path: pathlib.Path) -> None:
        """The per-kind floor rule, inherited from the two-cartridge defect."""
        primary, *others = _staged(tmp_path, ("alpha", "beta", "gamma"))

        observations, _digest = sweep.measure_sweep(
            TINY_SWEEP_PLAN, corpus=primary, other_corpora=others, device="cpu"
        )
        values = {observation["name"]: observation["value"] for observation in observations}

        for policy in sweep.POLICIES:
            assert values[f"{policy}_composed_noise_floor"] == pytest.approx(
                max(
                    values[f"{policy}-n{count}-composed_spread"]
                    for count in TINY_SWEEP_PLAN["compartment_counts"]
                )
            )

    def test_the_fixed_alone_arms_agree_exactly_across_counts(self, tmp_path: pathlib.Path) -> None:
        """The internal replication check.

        Under the fixed policy the alone arm is the same corpus, slot count
        and seeds at every compartment count, trained independently each
        time. Exact agreement is what "a plan is a function of its seeds"
        means; approximate agreement here would mean training still consumes
        state the seed does not name.
        """
        primary, *others = _staged(tmp_path, ("alpha", "beta", "gamma"))

        observations, _digest = sweep.measure_sweep(
            TINY_SWEEP_PLAN, corpus=primary, other_corpora=others, device="cpu"
        )
        values = {observation["name"]: observation["value"] for observation in observations}

        assert values["fixed-n2-alone_mean"] == values["fixed-n3-alone_mean"]
        assert values["fixed-n2-alone_spread"] == values["fixed-n3-alone_spread"]

    def test_the_slot_policies_record_their_arithmetic(self, tmp_path: pathlib.Path) -> None:
        primary, *others = _staged(tmp_path, ("alpha", "beta", "gamma"))

        observations, _digest = sweep.measure_sweep(
            TINY_SWEEP_PLAN, corpus=primary, other_corpora=others, device="cpu"
        )
        values = {observation["name"]: observation["value"] for observation in observations}

        assert values["fixed-n2_slots_per_cartridge"] == 2.0
        assert values["fixed-n3_slots_per_cartridge"] == 2.0
        assert values["budget-n2_slots_per_cartridge"] == 3.0
        assert values["budget-n3_slots_per_cartridge"] == 2.0

    def test_too_few_other_corpora_are_refused_up_front(self, tmp_path: pathlib.Path) -> None:
        primary, other = _staged(tmp_path, ("alpha", "beta"))

        with pytest.raises(ValueError, match="needs 2 other corpora; 1 supplied"):
            sweep.measure_sweep(
                TINY_SWEEP_PLAN, corpus=primary, other_corpora=[other], device="cpu"
            )

    def test_a_short_other_corpus_is_refused_not_weakened(self, tmp_path: pathlib.Path) -> None:
        primary, ok, short = _staged(tmp_path, ("alpha", "beta", "short"))

        with pytest.raises(AppError, match="misread as a composition cost"):
            sweep.measure_sweep(
                TINY_SWEEP_PLAN, corpus=primary, other_corpora=[ok, short], device="cpu"
            )

    def test_the_digest_is_of_the_primary_corpus(self, tmp_path: pathlib.Path) -> None:
        primary, *others = _staged(tmp_path, ("alpha", "beta", "gamma"))

        _observations, digest = sweep.measure_sweep(
            TINY_SWEEP_PLAN, corpus=primary, other_corpora=others, device="cpu"
        )

        from model_trainer.core.services.model.cartridge_plans import corpus_digest

        assert digest == corpus_digest(_documents("a"))


class TestRunRecord:
    def test_it_carries_the_experiment_and_a_corpus_stamped_label(
        self, tmp_path: pathlib.Path
    ) -> None:
        primary, *others = _staged(tmp_path, ("alpha", "beta", "gamma"))

        record = sweep.composition_sweep_run_record(
            "tiny", corpus=primary, other_corpora=others, device="cpu"
        )

        assert record["experiment"] == COMPOSITION_SWEEP_EXPERIMENT
        assert record["label"].startswith(
            "tiny-tiny-under-test-w8-s3-e1-lr0.05-n2.3-f2-b6-seeds7.8.9-"
        )

    def test_an_unknown_plan_names_the_known_ones(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="tiny"):
            sweep.composition_sweep_run_record(
                "no-such-plan", corpus=tmp_path, other_corpora=[], device="cpu"
            )


class TestHookDefault:
    def test_the_production_hook_serves_the_declared_table(self) -> None:
        assert measurement_hooks._default_composition_sweep_plans() is COMPOSITION_SWEEP_PLANS


def _argv(tmp_path: pathlib.Path, *, others: str) -> list[str]:
    """Build a complete command line against staged corpora.

    Args:
        tmp_path: The test's temporary directory.
        others: The ``--other-corpora`` value, verbatim.

    Returns:
        The flags, without a program name.
    """
    return [
        "--plan",
        "tiny",
        "--corpus",
        str(tmp_path / "alpha"),
        "--other-corpora",
        others,
        "--device",
        "cpu",
        "--out",
        str(tmp_path / "nested" / "record.json"),
    ]


class TestMain:
    def test_it_writes_a_decodable_record(self, tmp_path: pathlib.Path) -> None:
        _staged(tmp_path, ("alpha", "beta", "gamma"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'}"

        code = sweep.main(_argv(tmp_path, others=others))

        assert code == 0
        restored = decode_run_record(
            load_json_str((tmp_path / "nested" / "record.json").read_text(encoding="utf-8"))
        )
        assert restored["experiment"] == COMPOSITION_SWEEP_EXPERIMENT

    def test_a_trailing_comma_names_no_extra_corpus(self, tmp_path: pathlib.Path) -> None:
        """``a,b,`` is two corpora, not two and an empty path.

        An empty entry would become ``Path(".")``, and the reader would then
        ingest whatever directory the process happened to run in -- a corpus
        nobody named, silently joined to the measurement.
        """
        _staged(tmp_path, ("alpha", "beta", "gamma"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'},"

        assert sweep.main(_argv(tmp_path, others=others)) == 0

    def test_the_other_corpora_flag_is_required(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--other-corpora"):
            sweep.main(
                [
                    "--plan",
                    "tiny",
                    "--corpus",
                    str(tmp_path),
                    "--device",
                    "cpu",
                    "--out",
                    str(tmp_path / "r.json"),
                ]
            )


class TestInvocationForms:
    """The console entry and `python -m` must both measure and write."""

    def test_the_console_entry_point_runs_and_exits_zero(self, tmp_path: pathlib.Path) -> None:
        _staged(tmp_path, ("alpha", "beta", "gamma"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'}"
        saved = sys.argv
        sys.argv = ["modeltrainer-cartridge-composition-sweep", *_argv(tmp_path, others=others)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                sweep.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0
        assert (tmp_path / "nested" / "record.json").is_file()

    def test_running_it_as_a_module_actually_measures(self, tmp_path: pathlib.Path) -> None:
        _staged(tmp_path, ("alpha", "beta", "gamma"))
        others = f"{tmp_path / 'beta'},{tmp_path / 'gamma'}"
        module_name = "model_trainer.cli.cartridge_composition_sweep"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", *_argv(tmp_path, others=others)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        restored = decode_run_record(
            load_json_str((tmp_path / "nested" / "record.json").read_text(encoding="utf-8"))
        )
        assert restored["experiment"] == COMPOSITION_SWEEP_EXPERIMENT
