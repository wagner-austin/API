"""The declared-cells solo CLI, exercised on a real tiny model.

Same split as the solo-seeds suite: real tiny GPT-2, real cartridge
training and scoring, real window arithmetic; faked hub loaders, corpus
reader and plan table. The assertions concentrate on what the declared
surface must guarantee: one gain row per (cell, seed) with the cell's
summary derived from exactly its own rows, tokens colliding loudly
instead of merging, the recorded grid set's tokens and label segment
byte-stable, the anchor configuration shared verbatim by both sets and
equal to the recorded plan row, and the base loading ONCE per walk.
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

#: Small corpora keep the production cell sets runnable in a test: two
#: 16-character documents give four 8-token windows, so a 48-epoch cell
#: is 96 tiny steps rather than thousands.
_DOCUMENT_CHARS = 32

#: Tiny knobs behind the same plan-hook seam the production path reads;
#: only window and stride are consumed by the cell walk (epochs, lr and
#: slots are the cells' own).
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

#: Two tiny declared cells for the unit paths that do not need the
#: production sets.
TINY_CELLS: tuple[grid.SoloGridCell, ...] = (
    {"token": "a", "learning_rate": 0.05, "num_slots": 2, "epochs": 1},
    {"token": "b", "learning_rate": 0.02, "num_slots": 3, "epochs": 2},
)

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
    # tiny-len512: the production capacity axis reaches 256 slots, and a
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
    def test_every_cell_seed_pair_has_a_row_and_the_base_loads_once(
        self, tmp_path: pathlib.Path
    ) -> None:
        alpha = _staged(tmp_path, "alpha")

        observations, _digest = grid.measure_solo_grid(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7, 8),
            cells=TINY_CELLS,
            device="cpu",
            checkpoints=tmp_path / "checkpoints",
        )

        names = [o["name"] for o in observations]
        assert len(names) == len(set(names))
        for cell in TINY_CELLS:
            token = cell["token"]
            for seed in (7, 8):
                assert f"grid-gpt2-{token}-seed{seed}_gain" in names
            assert f"grid-gpt2-{token}_gain_mean" in names
            assert f"grid-gpt2-{token}_gain_spread" in names
        # 3 corpus rows + 2 cells * (2 seeds + mean + spread)
        assert len(names) == 3 + 2 * 4
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
            cells=TINY_CELLS[:1],
            device="cpu",
            checkpoints=tmp_path / "checkpoints",
        )
        recorded = {o["name"]: o["value"] for o in observations}
        gains = [recorded["grid-gpt2-a-seed7_gain"], recorded["grid-gpt2-a-seed8_gain"]]
        assert recorded["grid-gpt2-a_gain_mean"] == sum(gains) / 2
        assert recorded["grid-gpt2-a_gain_spread"] == max(gains) - min(gains)

    def test_a_cell_equals_an_independent_solo_recomputation(self, tmp_path: pathlib.Path) -> None:
        """The reproduction contract in miniature: a cell run inside the
        walk equals the same training run standalone -- same seed, same
        knobs, same split -- which is what lets the anchor cell stand in
        for the certified solo record.
        """
        alpha = _staged(tmp_path, "alpha")

        observations, _digest = grid.measure_solo_grid(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7,),
            cells=TINY_CELLS[1:],
            device="cpu",
            checkpoints=tmp_path / "checkpoints",
        )
        recorded = {o["name"]: o["value"] for o in observations}

        tokenizer = _fake_tokenizer("gpt2")
        encoded = [tokenizer.encode(document) for document in _documents("a")]
        train, held_out = split_by_stride(
            build_windows(encoded, window=8, device="cpu"), held_out_stride=3
        )
        base = require_cache_capable(_fake_model("gpt2", None))
        slots = train_cartridge(base, train, num_slots=3, seed=7, epochs=2, learning_rate=0.02)
        expected = held_out_gain(CartridgeModel(base=base, slots=slots), held_out)

        assert recorded["grid-gpt2-b-seed7_gain"] == expected

    def test_empty_inputs_and_colliding_tokens_are_refused(self, tmp_path: pathlib.Path) -> None:
        alpha = _staged(tmp_path, "alpha")
        with pytest.raises(ValueError, match="no seeds named"):
            grid.measure_solo_grid(
                alpha,
                model_id="gpt2",
                load_precision=None,
                seeds=(),
                cells=TINY_CELLS,
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
            )
        with pytest.raises(ValueError, match="no cells named"):
            grid.measure_solo_grid(
                alpha,
                model_id="gpt2",
                load_precision=None,
                seeds=(7,),
                cells=(),
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
            )
        collided: grid.SoloGridCell = {
            "token": "a",
            "learning_rate": 0.02,
            "num_slots": 3,
            "epochs": 2,
        }
        colliding = (TINY_CELLS[0], collided)
        with pytest.raises(ValueError, match="duplicate cell token"):
            grid.measure_solo_grid(
                alpha,
                model_id="gpt2",
                load_precision=None,
                seeds=(7,),
                cells=colliding,
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
            )


class TestDeclaredSets:
    def test_the_grid_set_preserves_the_certified_record_forms(self) -> None:
        """Tokens and label segment byte-equal to the bc18d701 record's."""
        assert grid.GRID_CELL_SET["name"] == "grid"
        assert grid.GRID_CELL_SET["label_segment"] == "l4-c2"
        assert tuple(cell["token"] for cell in grid.GRID_CELL_SET["cells"]) == (
            "lr0.001-c64",
            "lr0.001-c256",
            "lr0.003-c64",
            "lr0.003-c256",
            "lr0.01-c64",
            "lr0.01-c256",
            "lr0.03-c64",
            "lr0.03-c256",
        )
        assert all(cell["epochs"] == 12 for cell in grid.GRID_CELL_SET["cells"])

    def test_the_epochs_line_varies_exposure_alone(self) -> None:
        cells = grid.EPOCHS_LINE_CELL_SET["cells"]
        assert grid.EPOCHS_LINE_CELL_SET["name"] == "epochs-line"
        assert grid.EPOCHS_LINE_CELL_SET["label_segment"] == "e12.24.48"
        assert tuple(cell["epochs"] for cell in cells) == (12, 24, 48)
        assert all(cell["learning_rate"] == grid.ANCHOR_LEARNING_RATE for cell in cells)
        assert all(cell["num_slots"] == grid.ANCHOR_SLOTS for cell in cells)

    def test_the_anchor_configuration_is_shared_verbatim_and_is_the_recorded_row(
        self,
    ) -> None:
        """Both sets carry the identical anchor cell, and its knobs are
        the recorded plan row's -- the contract that lets one cell pair
        by name across records and reproduce the certified solo runs.
        """
        grid_anchor = grid.GRID_CELL_SET["cells"][4]
        line_anchor = grid.EPOCHS_LINE_CELL_SET["cells"][0]
        assert grid_anchor == line_anchor
        assert grid_anchor["token"] == grid.ANCHOR_TOKEN
        row = BASE_LORA_SWEEP_PLANS[GEOMETRY_PLAN_NAME]
        assert row["learning_rate"] == grid.ANCHOR_LEARNING_RATE
        assert row["slots"] == grid.ANCHOR_SLOTS
        assert row["epochs"] == grid.ANCHOR_EPOCHS
        assert grid_anchor["learning_rate"] == grid.ANCHOR_LEARNING_RATE
        assert grid_anchor["num_slots"] == grid.ANCHOR_SLOTS
        assert grid_anchor["epochs"] == grid.ANCHOR_EPOCHS

    def test_the_selector_resolves_both_sets_and_refuses_others(self) -> None:
        assert grid.cell_set_for("grid") == grid.GRID_CELL_SET
        assert grid.cell_set_for("epochs-line") == grid.EPOCHS_LINE_CELL_SET
        with pytest.raises(ValueError, match="unknown cell set"):
            grid.cell_set_for("warmup-line")

    def test_the_label_carries_the_sets_declared_segment(self) -> None:
        label = grid.solo_grid_label(
            model_id="EleutherAI/pythia-6.9b",
            seeds=SOLO_SEEDS,
            label_segment=grid.EPOCHS_LINE_CELL_SET["label_segment"],
            precision_token="-storedbf16",
            digest="0" * 12,
        )
        assert label == "cartridge-solo-grid-pythia-6.9b-storedbf16-e12.24.48-n9-000000000000"


class TestMain:
    def test_main_walks_the_selected_set_and_writes_the_record(
        self, tmp_path: pathlib.Path
    ) -> None:
        alpha = _staged(tmp_path, "alpha")
        out = tmp_path / "record" / "line.json"

        code = grid.main(
            [
                "--model-id",
                "gpt2",
                "--precision",
                "policy",
                "--cells",
                "epochs-line",
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
        assert record["label"].startswith("cartridge-solo-grid-gpt2-e12.24.48-n9-")
        names = {o["name"] for o in record["observations"]}
        for cell in grid.EPOCHS_LINE_CELL_SET["cells"]:
            assert f"grid-gpt2-{cell['token']}_gain_mean" in names
        assert _LOADS == ["gpt2"]

    def test_entrypoint_reads_process_argv_and_exits_with_mains_code(
        self, tmp_path: pathlib.Path
    ) -> None:
        alpha = _staged(tmp_path, "alpha")
        out = tmp_path / "line.json"
        saved = sys.argv
        sys.argv = [
            "prog",
            "--model-id",
            "gpt2",
            "--precision",
            "policy",
            "--cells",
            "epochs-line",
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
        out = tmp_path / "module" / "line.json"
        module_name = "model_trainer.cli.cartridge_solo_grid"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = [
            "x",
            "--model-id",
            "gpt2",
            "--precision",
            "policy",
            "--cells",
            "epochs-line",
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
