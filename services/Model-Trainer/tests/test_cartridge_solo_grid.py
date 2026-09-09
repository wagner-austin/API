"""The solo-grid CLI, exercised on a real tiny model over a fake corpus.

Same split as the solo-seeds suite it extends: real tiny GPT-2, real
cartridge training and scoring, real window arithmetic; faked hub
loaders, corpus reader and plan table. What the assertions concentrate
on: the grid is complete with one gain row per (cell, seed), each cell's
summary derives from exactly its own rows, the anchor cell's knobs are
byte-equal to the recorded plan row's (the in-grid reproduction
contract), and the base loads ONCE for the whole grid.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import cartridge_solo_grid as grid
from model_trainer.cli.cartridge_headroom import GEOMETRY_PLAN_NAME
from model_trainer.cli.cartridge_solo_seeds import SOLO_SEEDS
from model_trainer.core.contracts.model import QuantizationConfig, StoredBf16Precision
from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_measurement import (
    held_out_gain,
    train_cartridge,
)
from model_trainer.core.services.model.cartridge_pool_plans import (
    BASE_LORA_SWEEP_PLANS,
    BaseLoraSweepPlan,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import LMModelProto
from tests.core.services.model.backends.hf_lm.testing import FakeHFTokenizer

_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]

_DOCUMENT_CHARS = 96

#: Tiny knobs behind the same plan-hook seam the production path reads;
#: only window/stride/epochs are consumed by the grid (lr and slots are
#: the grid's own axes).
TINY_GEOMETRY_PLAN: BaseLoraSweepPlan = {
    "model_id": "gpt2",
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

_LOADS: list[str] = []


def _fake_plans() -> dict[str, BaseLoraSweepPlan]:
    """Stand in for the production plan table, geometry row included."""
    return {GEOMETRY_PLAN_NAME: TINY_GEOMETRY_PLAN}


def _fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader."""
    return FakeHFTokenizer(vocab_size=_VOCAB)


def _fake_model(
    model_id_or_path: str, quantization: QuantizationConfig | StoredBf16Precision | None
) -> LMModelProto:
    """Stand in for the hub loader, recording every load."""
    _LOADS.append(model_id_or_path)
    # tiny-len512: the production slot axis reaches 256 slots, and a
    # 64-position table cannot host 256 slots plus a window.
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny-len512"])
    return model


def _documents(marker: str) -> tuple[str, ...]:
    """Two documents, distinct by marker character."""
    return tuple(f"{marker}{index}" * (_DOCUMENT_CHARS // 2) for index in range(2))


def _fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Stand in for the corpus reader, keyed on the directory's name."""
    return _documents(corpus_dir.name[0])


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards."""
    _LOADS.clear()
    measurement_hooks.base_lora_sweep_plans = _fake_plans
    cli_hooks.read_corpus_documents = _fake_corpus_reader
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model
    yield None
    measurement_hooks.base_lora_sweep_plans = measurement_hooks._default_base_lora_sweep_plans
    cli_hooks.read_corpus_documents = cli_hooks._default_read_corpus_documents
    hf_hooks.Hooks.reset()


def _staged(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    """Create one corpus directory."""
    path = tmp_path / name
    path.mkdir()
    return path


class TestMeasureSoloGrid:
    def test_the_grid_is_complete_and_the_base_loads_once(self, tmp_path: pathlib.Path) -> None:
        alpha = _staged(tmp_path, "alpha")

        observations, _digest = grid.measure_solo_grid(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7, 8),
            learning_rates=(0.05, 0.02),
            slot_counts=(2, 3),
            device="cpu",
        )

        names = [o["name"] for o in observations]
        assert len(names) == len(set(names))
        for lr in (0.05, 0.02):
            for slots in (2, 3):
                token = grid.cell_token(learning_rate=lr, num_slots=slots)
                for seed in (7, 8):
                    assert f"grid-gpt2-{token}-seed{seed}_gain" in names
                assert f"grid-gpt2-{token}_gain_mean" in names
                assert f"grid-gpt2-{token}_gain_spread" in names
        # 3 corpus rows + 4 cells * (2 seeds + mean + spread)
        assert len(names) == 3 + 4 * 4
        assert _LOADS == ["gpt2"]

    def test_a_cells_summary_derives_from_exactly_its_own_rows(
        self, tmp_path: pathlib.Path
    ) -> None:
        alpha = _staged(tmp_path, "alpha")

        observations, _digest = grid.measure_solo_grid(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7, 8),
            learning_rates=(0.05,),
            slot_counts=(2,),
            device="cpu",
        )
        recorded = {o["name"]: o["value"] for o in observations}
        token = grid.cell_token(learning_rate=0.05, num_slots=2)
        gains = [
            recorded[f"grid-gpt2-{token}-seed7_gain"],
            recorded[f"grid-gpt2-{token}-seed8_gain"],
        ]
        assert recorded[f"grid-gpt2-{token}_gain_mean"] == sum(gains) / 2
        assert recorded[f"grid-gpt2-{token}_gain_spread"] == max(gains) - min(gains)

    def test_the_anchor_cell_equals_an_independent_solo_recomputation(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The in-grid reproduction contract, in miniature.

        A cell run inside the grid must equal the same training run
        standalone -- same seed, same knobs, same split -- or the anchor
        cell could not stand in for the recorded solo run.
        """
        alpha = _staged(tmp_path, "alpha")

        observations, _digest = grid.measure_solo_grid(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7,),
            learning_rates=(0.05,),
            slot_counts=(2,),
            device="cpu",
        )
        recorded = {o["name"]: o["value"] for o in observations}

        tokenizer = _fake_tokenizer("gpt2")
        encoded = [tokenizer.encode(document) for document in _documents("a")]
        train, held_out = split_by_stride(
            build_windows(encoded, window=8, device="cpu"), held_out_stride=3
        )
        base = require_cache_capable(_fake_model("gpt2", None))
        slots = train_cartridge(base, train, num_slots=2, seed=7, epochs=1, learning_rate=0.05)
        expected = held_out_gain(CartridgeModel(base=base, slots=slots), held_out)

        token = grid.cell_token(learning_rate=0.05, num_slots=2)
        assert recorded[f"grid-gpt2-{token}-seed7_gain"] == expected

    def test_empty_axes_and_seeds_are_refused(self, tmp_path: pathlib.Path) -> None:
        alpha = _staged(tmp_path, "alpha")
        with pytest.raises(ValueError, match="no seeds named"):
            grid.measure_solo_grid(
                alpha,
                model_id="gpt2",
                load_precision=None,
                seeds=(),
                learning_rates=(0.05,),
                slot_counts=(2,),
                device="cpu",
            )
        with pytest.raises(ValueError, match="no learning rates named"):
            grid.measure_solo_grid(
                alpha,
                model_id="gpt2",
                load_precision=None,
                seeds=(7,),
                learning_rates=(),
                slot_counts=(2,),
                device="cpu",
            )
        with pytest.raises(ValueError, match="no slot counts named"):
            grid.measure_solo_grid(
                alpha,
                model_id="gpt2",
                load_precision=None,
                seeds=(7,),
                learning_rates=(0.05,),
                slot_counts=(),
                device="cpu",
            )


class TestGridConstants:
    def test_the_axes_bracket_the_recorded_knobs(self) -> None:
        assert grid.GRID_LEARNING_RATES == (0.001, 0.003, 0.01, 0.03)
        assert grid.GRID_SLOT_COUNTS == (64, 256)
        assert grid.ANCHOR_LEARNING_RATE in grid.GRID_LEARNING_RATES
        assert grid.ANCHOR_SLOTS in grid.GRID_SLOT_COUNTS

    def test_the_anchor_is_the_recorded_plan_rows_knobs(self) -> None:
        """The in-grid anchor must be byte-equal to the recorded solo knobs.

        Asserted against the PRODUCTION row, not the test fake: this is
        the contract that lets the anchor cell reproduce the recorded
        nine-seed solo record.
        """
        row = BASE_LORA_SWEEP_PLANS[GEOMETRY_PLAN_NAME]
        assert row["learning_rate"] == grid.ANCHOR_LEARNING_RATE
        assert row["slots"] == grid.ANCHOR_SLOTS

    def test_the_cell_token_is_stable(self) -> None:
        assert grid.cell_token(learning_rate=0.003, num_slots=256) == "lr0.003-c256"

    def test_the_label_names_both_axes(self) -> None:
        label = grid.solo_grid_label(
            model_id="EleutherAI/pythia-6.9b",
            seeds=SOLO_SEEDS,
            learning_rates=grid.GRID_LEARNING_RATES,
            slot_counts=grid.GRID_SLOT_COUNTS,
            precision_token="-storedbf16",
            digest="0" * 12,
        )
        assert label == "cartridge-solo-grid-pythia-6.9b-storedbf16-l4-c2-n9-000000000000"


class TestMain:
    def test_main_walks_the_production_grid_and_writes_the_record(
        self, tmp_path: pathlib.Path
    ) -> None:
        alpha = _staged(tmp_path, "alpha")
        out = tmp_path / "record" / "grid.json"

        code = grid.main(
            [
                "--model-id",
                "gpt2",
                "--precision",
                "policy",
                "--corpus",
                str(alpha),
                "--device",
                "cpu",
                "--out",
                str(out),
            ]
        )

        assert code == 0
        record = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert record["experiment"] == grid.SOLO_GRID_EXPERIMENT
        assert record["label"].startswith("cartridge-solo-grid-gpt2-l4-c2-n9-")
        names = {o["name"] for o in record["observations"]}
        for lr in grid.GRID_LEARNING_RATES:
            for slots in grid.GRID_SLOT_COUNTS:
                token = grid.cell_token(learning_rate=lr, num_slots=slots)
                assert f"grid-gpt2-{token}_gain_mean" in names
        assert _LOADS == ["gpt2"]

    def test_entrypoint_reads_process_argv_and_exits_with_mains_code(
        self, tmp_path: pathlib.Path
    ) -> None:
        alpha = _staged(tmp_path, "alpha")
        out = tmp_path / "grid.json"
        saved = sys.argv
        sys.argv = [
            "prog",
            "--model-id",
            "gpt2",
            "--precision",
            "policy",
            "--corpus",
            str(alpha),
            "--device",
            "cpu",
            "--out",
            str(out),
        ]
        try:
            with pytest.raises(SystemExit) as caught:
                grid.entrypoint()
        finally:
            sys.argv = saved

        assert caught.value.code == 0
        assert out.exists()

    def test_running_it_as_a_module_actually_measures(self, tmp_path: pathlib.Path) -> None:
        alpha = _staged(tmp_path, "alpha")
        out = tmp_path / "module" / "grid.json"
        module_name = "model_trainer.cli.cartridge_solo_grid"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = [
            "x",
            "--model-id",
            "gpt2",
            "--precision",
            "policy",
            "--corpus",
            str(alpha),
            "--device",
            "cpu",
            "--out",
            str(out),
        ]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        assert out.is_file()
