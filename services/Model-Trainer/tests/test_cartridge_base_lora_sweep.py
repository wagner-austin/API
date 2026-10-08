"""The base-LoRA sweep entry, exercised on a real model over fake corpora.

Same split as the sibling sweeps' suites -- real tiny GPT-2, real PEFT
adaptation, real training and scoring; faked hub loaders, corpus reader and
plan table. The assertions concentrate on the record's names (both cell
families, the companion-cross arms, the LoRA's own epoch losses, per-family
floors), the contamination refusals, and the production plan's
comparability against the recorded diverse grid.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator, Mapping
from typing import TypedDict

import pytest
from platform_core.json_utils import load_json_str
from platform_core.run_record import RunRecord, decode_run_record

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import cartridge_base_lora_sweep as sweep
from model_trainer.cli import cartridge_lora_policy as policy
from model_trainer.cli.cartridge_crowd_adapters import LORA_TRAIN_SEED, POOL_SEED_BASE
from model_trainer.core.contracts.model import QuantizationConfig, StoredBf16Precision
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_pool_plans import (
    BASE_LORA_SWEEP_EXPERIMENT,
    BASE_LORA_SWEEP_PLANS,
    DIVERSE_COMPANION_SWEEP_PLANS,
    BaseLoraSweepPlan,
    base_lora_sweep_label,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import LMModelProto
from tests._module_run import run_module_as_main
from tests.core.services.model.backends.hf_lm.testing import FakeHFTokenizer

#: The module-scoped walk is shared by TestTheWalk, so this file's tests run on
#: one xdist worker (tests/test_xdist_grouping.py says why).
pytestmark = pytest.mark.xdist_group("test_cartridge_base_lora_sweep.py")

#: A plan small enough to run in a test and shaped like the real one: two
#: counts for the n-axis, a two-corpus pool so the crowd draw is live.
TINY_LORA_PLAN: BaseLoraSweepPlan = {
    "model_id": "gpt2",  # a real policy id (the fakes return a tiny GPT-2 anyway)
    "window": 8,
    "held_out_stride": 3,
    "compartment_counts": (2, 3),
    "slots": 2,
    "probability": 0.5,
    "max_companions": 2,
    "lora_rank": 2,
    "lora_alpha": 4,
    "lora_epochs": 1,
    "lora_learning_rate": 0.05,
    "max_drawn": 2,
    "pool_members_per_corpus": 1,
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
}

_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]


def _fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader."""
    assert model_id_or_path == TINY_LORA_PLAN["model_id"]
    return FakeHFTokenizer(vocab_size=_VOCAB)


def _fake_model(
    model_id_or_path: str, quantization: QuantizationConfig | StoredBf16Precision | None
) -> LMModelProto:
    """Stand in for the hub model loader, returning a real tiny GPT-2."""
    assert model_id_or_path == TINY_LORA_PLAN["model_id"]
    assert quantization is None
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


def _documents(marker: str) -> tuple[str, ...]:
    """Four documents of 24 characters each, twelve windows of eight.

    Args:
        marker: Character that makes this corpus different from another.

    Returns:
        The corpus bodies.
    """
    return tuple(f"{marker}{index}" * 12 for index in range(4))


def _fake_plans() -> Mapping[str, BaseLoraSweepPlan]:
    """Stand in for the production plan table, with one runnable plan."""
    return {"tiny": TINY_LORA_PLAN}


def _fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Stand in for the corpus reader, keyed on the directory's own name."""
    return _documents(corpus_dir.name[0])


def _install_fakes() -> None:
    """Point the plan, corpus and hub hooks at this file's fakes."""
    measurement_hooks.base_lora_sweep_plans = _fake_plans
    cli_hooks.read_corpus_documents = _fake_corpus_reader
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model


def _restore_hooks() -> None:
    """Put the production hooks back."""
    measurement_hooks.base_lora_sweep_plans = measurement_hooks._default_base_lora_sweep_plans
    cli_hooks.read_corpus_documents = cli_hooks._default_read_corpus_documents
    hf_hooks.Hooks.reset()


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards."""
    _install_fakes()
    yield None
    _restore_hooks()


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


def _argv(tmp_path: pathlib.Path, *, others: str, pool: str) -> list[str]:
    """Build a complete command line against staged corpora.

    Args:
        tmp_path: The test's temporary directory.
        others: The ``--other-corpora`` value, verbatim.
        pool: The ``--pool-corpora`` value, verbatim.

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
        "--pool-corpora",
        pool,
        "--device",
        "cpu",
        "--out",
        str(tmp_path / "nested" / "record.json"),
    ]


class _Walk(TypedDict):
    """One ``python -m`` run of the tiny grid and the record it wrote."""

    code: int | str | None
    record: RunRecord


@pytest.fixture(name="walk", scope="module")
def _walk(tmp_path_factory: pytest.TempPathFactory) -> _Walk:
    """Walk the tiny grid once through ``python -m``, for :class:`TestTheWalk`.

    Module-scoped, so it runs before the function-scoped ``wired`` fixture
    and installs the fakes itself; the pool value ends in a comma, which the
    walk must read as no third pool corpus.

    Args:
        tmp_path_factory: Source of the walk's own directory.

    Returns:
        The exit code and the decoded record.
    """
    root = tmp_path_factory.mktemp("walk")
    _staged(root, ("alpha", "beta", "gamma", "delta", "echo"))
    _install_fakes()
    try:
        code = run_module_as_main(
            "model_trainer.cli.cartridge_base_lora_sweep",
            _argv(
                root,
                others=f"{root / 'beta'},{root / 'gamma'}",
                pool=f"{root / 'delta'},{root / 'echo'},",
            ),
        )
    finally:
        _restore_hooks()
    text = (root / "nested" / "record.json").read_text(encoding="utf-8")
    return {"code": code, "record": decode_run_record(load_json_str(text))}


class TestTheWalk:
    """What the one walk of the tiny grid recorded.

    ONE WALK SERVES EVERY TEST HERE (MCPs board task 2f90d785). Seven tests
    each walked this same grid, 3.4 to 6.3 s apiece alone and up to 17 s in
    CI: two read ``measure_grid``'s names, one the run record, two
    ``main()`` and two the console entry and ``python -m``. ``python -m``
    runs the ``__main__`` guard, which calls ``entrypoint()``, which calls
    ``main()`` on the process arguments, which builds the run record from
    ``measure_grid``'s observations unchanged; so the module-scoped ``walk``
    proves every form on one execution, and each test keeps its own
    assertion on what that execution wrote.
    """

    def test_the_module_run_exits_zero_with_a_decodable_record(self, walk: _Walk) -> None:
        # The pool value ends in a comma, which must name no third corpus:
        # the fake reader keys on a directory's first character and an empty
        # name has none, and the plan refuses a pool that is not two strong.
        assert walk["code"] == 0
        assert walk["record"]["experiment"] == BASE_LORA_SWEEP_EXPERIMENT

    def test_it_carries_a_corpus_stamped_label(self, walk: _Walk) -> None:
        assert walk["record"]["label"].startswith(
            "tiny-gpt2-w8-s3-e1-lr0.05-n2.3-c2-p0.5-K2-R2-a4-le1-llr0.05-D2-m1-seeds7.8.9-"
        )

    def test_every_arm_is_named_once(self, walk: _Walk) -> None:
        names = [observation["name"] for observation in walk["record"]["observations"]]
        assert len(names) == len(set(names))

    def test_it_names_both_families_the_crosses_and_the_losses(self, walk: _Walk) -> None:
        named = {observation["name"] for observation in walk["record"]["observations"]}
        assert "max_drawn" in named
        assert "lora-train-epoch-0_loss" in named
        # The seed pairing is the evidence: every arm carries one gain per
        # seed, and the SAME seeds across arms of a cell, so a paired
        # re-analysis is possible from the record alone.
        for seed in (7, 8, 9):
            assert f"lora-plain-n2-alone_seed{seed}_gain" in named
            assert f"lora-plain-n2-composed_seed{seed}_gain" in named
            assert f"lora-diverse-n3-untrained-composed_seed{seed}_gain" in named
            assert f"lora-companion-cross-0_seed{seed}_gain" in named
        assert "lora-companion-cross-0_mean" in named
        assert "lora-companion-cross-1_spread" in named
        assert "lora-plain-n2-alone_mean" in named
        assert "lora-plain-n3-composed_spread" in named
        assert "lora-plain-n2-cross-0_mean" in named
        assert "lora-diverse-n2-alone_mean" in named
        assert "lora-diverse-n3-untrained-composed_mean" in named
        assert "lora-plain_composed_noise_floor" in named
        assert "lora-diverse_composed_noise_floor" in named


class TestMeasureGrid:
    def test_a_pool_count_mismatching_the_plan_is_refused(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma, delta = _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))

        with pytest.raises(ValueError, match="pool of 2 corpora"):
            sweep.measure_grid(
                TINY_LORA_PLAN,
                corpus=primary,
                plan_name="tiny",
                other_corpora=[beta, gamma],
                pool_corpora=[delta],
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
            )

    def test_a_repeated_pool_corpus_is_refused(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma, delta = _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))

        with pytest.raises(ValueError, match="narrows the crowd"):
            sweep.measure_grid(
                TINY_LORA_PLAN,
                corpus=primary,
                plan_name="tiny",
                other_corpora=[beta, gamma],
                pool_corpora=[delta, delta],
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
            )

    def test_a_pool_corpus_that_is_also_measured_is_refused(self, tmp_path: pathlib.Path) -> None:
        primary, beta, gamma, delta = _staged(tmp_path, ("alpha", "beta", "gamma", "delta"))

        with pytest.raises(ValueError, match="carry the answer in its LoRA"):
            sweep.measure_grid(
                TINY_LORA_PLAN,
                corpus=primary,
                plan_name="tiny",
                other_corpora=[beta, gamma],
                pool_corpora=[delta, gamma],
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
            )

    def test_too_few_other_corpora_are_refused_up_front(self, tmp_path: pathlib.Path) -> None:
        primary, beta, delta, echo = _staged(tmp_path, ("alpha", "beta", "delta", "echo"))

        with pytest.raises(ValueError, match="needs 2 other corpora; 1 supplied"):
            sweep.measure_grid(
                TINY_LORA_PLAN,
                corpus=primary,
                plan_name="tiny",
                other_corpora=[beta],
                pool_corpora=[delta, echo],
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
            )


class TestRunRecord:
    def test_an_unknown_plan_names_the_known_ones(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="tiny"):
            sweep.base_lora_sweep_run_record(
                "no-such-plan",
                corpus=tmp_path,
                other_corpora=[],
                pool_corpora=[],
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
            )


class TestHookDefault:
    def test_the_production_hook_serves_the_declared_table(self) -> None:
        assert measurement_hooks._default_base_lora_sweep_plans() is BASE_LORA_SWEEP_PLANS


class TestProductionPlan:
    def test_the_lora_plan_matches_the_diverse_grid_on_every_measurement_field(self) -> None:
        """The cells subtract against the recorded diverse grid only if the
        measurement knobs agree; the LoRA knobs are the plan's own."""
        diverse = DIVERSE_COMPANION_SWEEP_PLANS["gpt2-companions-diverse"]
        lora = BASE_LORA_SWEEP_PLANS["gpt2-base-lora"]
        assert lora["model_id"] == diverse["model_id"]
        assert lora["window"] == diverse["window"]
        assert lora["held_out_stride"] == diverse["held_out_stride"]
        assert lora["compartment_counts"] == diverse["compartment_counts"]
        assert lora["slots"] == diverse["slots"]
        assert lora["probability"] == diverse["probability"]
        assert lora["max_companions"] == diverse["max_companions"]
        assert lora["seeds"] == diverse["seeds"]
        assert lora["epochs"] == diverse["epochs"]
        assert lora["learning_rate"] == diverse["learning_rate"]
        assert lora["lora_rank"] == 8
        assert lora["max_drawn"] == 8
        assert lora["pool_members_per_corpus"] == 3

    def test_the_medium_rung_differs_from_the_recorded_plan_only_in_the_base(self) -> None:
        """The medium row isolates parameter count: every other field --
        schedule and LoRA knobs deliberately included -- must equal the
        recorded plan, or scale is confounded with tuning."""
        recorded = BASE_LORA_SWEEP_PLANS["gpt2-base-lora"]
        medium = BASE_LORA_SWEEP_PLANS["gpt2-medium-base-lora"]
        assert medium["model_id"] == "gpt2-medium"
        assert {**medium, "model_id": recorded["model_id"]} == recorded

    def test_the_xl_rung_differs_from_the_recorded_plan_only_in_the_base(self) -> None:
        """The 1.5B rung of the depth ladder, same isolation rule."""
        recorded = BASE_LORA_SWEEP_PLANS["gpt2-base-lora"]
        xl = BASE_LORA_SWEEP_PLANS["gpt2-xl-base-lora"]
        assert xl["model_id"] == "gpt2-xl"
        assert {**xl, "model_id": recorded["model_id"]} == recorded

    def test_the_7b_rung_differs_from_the_recorded_plan_only_in_the_base(self) -> None:
        """The architecture jump keeps every measurement knob: precision is
        the CLI's loading policy, deliberately not a plan field."""
        recorded = BASE_LORA_SWEEP_PLANS["gpt2-base-lora"]
        pythia = BASE_LORA_SWEEP_PLANS["pythia-6.9b-base-lora"]
        assert pythia["model_id"] == "EleutherAI/pythia-6.9b"
        assert {**pythia, "model_id": recorded["model_id"]} == recorded


class TestArchitecturePolicy:
    """The two per-base decisions the CLI makes before anything loads."""

    def test_every_gpt2_rung_adapts_the_fused_conv_and_loads_fp32(self) -> None:
        for model_id in ("gpt2", "gpt2-medium", "gpt2-xl"):
            assert policy.target_modules_for(model_id) == ("c_attn",)
            assert policy.quantization_for(model_id) is None

    def test_the_7b_rung_adapts_neox_attention_under_nf4(self) -> None:
        assert policy.target_modules_for("EleutherAI/pythia-6.9b") == ("query_key_value",)
        quantization = policy.quantization_for("EleutherAI/pythia-6.9b")
        assert quantization == {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        }

    def test_an_undeclared_base_refuses_in_both_policies(self) -> None:
        """A heuristic here would silently adapt the wrong modules or load
        an undeclared 7B in fp32; both must refuse by name instead."""
        with pytest.raises(ValueError, match="no LoRA target modules"):
            policy.target_modules_for("mistralai/Mistral-7B-v0.1")
        with pytest.raises(ValueError, match="no loading precision"):
            policy.quantization_for("mistralai/Mistral-7B-v0.1")

    def test_the_seed_geography_cannot_collide(self) -> None:
        """The LoRA and pool seeds sit past every measurement offset."""
        plan = BASE_LORA_SWEEP_PLANS["gpt2-base-lora"]
        from model_trainer.cli.cartridge_companion_sweep import COMPANION_SEED_STRIDE

        largest_measurement_seed = max(plan["seeds"]) + (
            COMPANION_SEED_STRIDE + plan["max_companions"] - 1
        ) * len(plan["seeds"])
        assert largest_measurement_seed < LORA_TRAIN_SEED < POOL_SEED_BASE
        pool_size = plan["max_companions"] * plan["pool_members_per_corpus"]
        assert plan["max_drawn"] <= pool_size

    def test_the_label_carries_every_lora_knob(self) -> None:
        label = base_lora_sweep_label(
            "gpt2-base-lora", BASE_LORA_SWEEP_PLANS["gpt2-base-lora"], digest="0" * 64
        )
        assert label.startswith(
            "gpt2-base-lora-gpt2-w256-s4-e12-lr0.01-n4.8-c64-p0.5-K3-R8-a16-le3-llr0.0001-D8-m3-seeds7.8.9-"
        )


class TestMain:
    def test_the_pool_corpora_flag_is_required(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--pool-corpora"):
            sweep.main(
                [
                    "--plan",
                    "tiny",
                    "--corpus",
                    str(tmp_path),
                    "--other-corpora",
                    str(tmp_path),
                    "--device",
                    "cpu",
                    "--out",
                    str(tmp_path / "r.json"),
                ]
            )
