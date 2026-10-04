"""The full-wiki family: each cell is the base with one field moved.

The property every test here protects is ATTRIBUTION. A cell that moved a
second field would produce a number nobody could assign to its axis, so each
axis is checked by rebuilding the expected cell from the base with exactly the
field the axis owns replaced, and comparing whole plans -- a field added to
:class:`QaPlan` later is then covered without anyone naming it here.
"""

from __future__ import annotations

import pytest

from model_trainer.core.contracts.qa_plan import QaPlan
from model_trainer.core.services.model.cartridge_qa_axes import (
    EPOCHS,
    FULL_WIKI_BASE,
    SCALE_RUNGS,
    SLOT_AXIS_MAX_SEQ_LEN,
    SLOT_AXIS_MODEL,
    SLOT_COUNTS,
    WINDOWS,
    epoch_cells,
    full_wiki_family,
    merged_plan_tables,
    scale_rungs,
    slot_cells,
    window_cells,
)
from model_trainer.core.services.model.cartridge_qa_plans import QA_PLANS
from model_trainer.core.services.model.cartridge_qa_power import resolvable_floor

#: Smallest realised item count of any cell, measured 2026-10-04 on
#: ~/PROJECTS/wiki (885 pages): the 512-token window cell. Every other cell
#: yields more (3,753 at window 128, 3,793 for pythia's tokenizer, 3,827 at
#: the base's 256).
_SMALLEST_CELL_ITEMS = 3278


def _base() -> QaPlan:
    return QA_PLANS[FULL_WIKI_BASE]


class TestTheScaleLadder:
    def test_each_rung_moves_only_the_base_and_its_precision(self) -> None:
        base = _base()

        for name, rung in scale_rungs(base).items():
            expected: QaPlan = {
                **base,
                "model_id": rung["model_id"],
                "precision_selector": rung["precision_selector"],
            }
            assert rung == expected, name

    def test_it_reaches_past_the_crossing_to_the_7b_base(self) -> None:
        """The crossing is reported at ~774M; a ladder must run past it.

        Until 2026-10-04 the only rung above gpt2-xl was declared on a corpus
        whose plans the power gate refused, so the ladder reached past its
        finding on paper and nowhere else.
        """
        rungs = scale_rungs(_base())

        assert [rung["model_id"] for rung in rungs.values()] == [
            model_id for model_id, _selector in SCALE_RUNGS
        ]
        assert rungs["pythia-6.9b-full-wiki-qa"]["model_id"] == "EleutherAI/pythia-6.9b"
        assert rungs["gpt2-large-full-wiki-qa"]["model_id"] == "gpt2-large"

    def test_the_7b_rung_loads_bf16_and_the_gpt2_rungs_their_policy(self) -> None:
        """fp32 7B is 27.6GB against the A30's 24GB; bf16 is 13.8GB."""
        rungs = scale_rungs(_base())

        assert rungs["pythia-6.9b-full-wiki-qa"]["precision_selector"] == "stored-bf16"
        for name in ("gpt2-medium-full-wiki-qa", "gpt2-large-full-wiki-qa", "gpt2-xl-full-wiki-qa"):
            assert rungs[name]["precision_selector"] == "policy", name


class TestTheCapacityAxis:
    def test_each_cell_moves_only_its_slot_count_from_the_crossing_rung(self) -> None:
        rung = scale_rungs(_base())["gpt2-large-full-wiki-qa"]

        for name, cell in slot_cells(rung).items():
            expected: QaPlan = {
                **rung,
                "num_slots": cell["num_slots"],
                "max_seq_len": SLOT_AXIS_MAX_SEQ_LEN,
            }
            assert cell == expected, name

    def test_it_spans_the_doubling_ladder_under_one_budget(self) -> None:
        """Every cell held to 768, or the smallest cartridge faces the most evidence."""
        cells = slot_cells(scale_rungs(_base())["gpt2-large-full-wiki-qa"])

        assert sorted(cells) == [f"gpt2-large-full-wiki-qa-slots-{n}" for n in (128, 256, 32, 64)]
        assert tuple(cell["num_slots"] for cell in cells.values()) == SLOT_COUNTS
        assert {cell["max_seq_len"] for cell in cells.values()} == {768}
        assert {cell["model_id"] for cell in cells.values()} == {SLOT_AXIS_MODEL}

    def test_the_budget_leaves_room_for_the_largest_prefix(self) -> None:
        """gpt2-large has 1024 positions; the 256-slot cell must still fit."""
        assert SLOT_AXIS_MAX_SEQ_LEN + max(SLOT_COUNTS) == 1024


class TestTheWindowAndEpochAxes:
    def test_each_window_cell_moves_only_its_window(self) -> None:
        base = _base()
        cells = window_cells(base)

        assert sorted(cells) == ["gpt2-full-wiki-qa-window-128", "gpt2-full-wiki-qa-window-512"]
        for name, cell in cells.items():
            expected: QaPlan = {**base, "window": cell["window"]}
            assert cell == expected, name
        assert tuple(cell["window"] for cell in cells.values()) == WINDOWS

    def test_each_epoch_cell_moves_only_its_epochs(self) -> None:
        base = _base()
        cells = epoch_cells(base)

        assert sorted(cells) == [
            "gpt2-full-wiki-qa-epochs-24",
            "gpt2-full-wiki-qa-epochs-3",
            "gpt2-full-wiki-qa-epochs-6",
        ]
        for name, cell in cells.items():
            expected: QaPlan = {**base, "epochs": cell["epochs"]}
            assert cell == expected, name
        assert tuple(cell["epochs"] for cell in cells.values()) == EPOCHS

    def test_neither_axis_repeats_the_base_s_own_value(self) -> None:
        """The base IS each axis's point at 256 and 12; a cell there would be a twin."""
        base = _base()

        assert base["window"] not in WINDOWS
        assert base["epochs"] not in EPOCHS

    def test_every_window_cell_leaves_room_for_the_cartridge_in_training(self) -> None:
        """A training window plus the prefix must fit gpt2's 1024 positions."""
        for cell in window_cells(_base()).values():
            assert cell["window"] + cell["num_slots"] <= 1024


class TestTheFamily:
    def test_it_is_the_ladder_and_the_three_axes_and_nothing_else(self) -> None:
        family = full_wiki_family(_base())

        assert len(family) == len(SCALE_RUNGS) + len(SLOT_COUNTS) + len(WINDOWS) + len(EPOCHS)
        assert FULL_WIKI_BASE not in family

    def test_the_registry_serves_every_cell_unchanged(self) -> None:
        for name, plan in full_wiki_family(_base()).items():
            assert QA_PLANS[name] == plan, name

    @pytest.mark.parametrize("name", sorted(full_wiki_family(QA_PLANS[FULL_WIKI_BASE])))
    def test_every_cell_clears_the_gate_on_its_measured_corpus(self, name: str) -> None:
        """The property the api-wiki plans lacked: each one can actually run.

        Checked through the gate's own floor at the smallest realised count
        any cell yields, so a cell declaring a finer effect than the corpus
        can resolve fails here rather than after a cluster allocation.

        Args:
            name: Cell name in the family.
        """
        plan = QA_PLANS[name]

        floor = resolvable_floor(_SMALLEST_CELL_ITEMS, plan["alpha"], plan["mcnemar_test"])

        assert floor < plan["smallest_effect_of_interest"]
        assert plan["max_items"] > _SMALLEST_CELL_ITEMS


class TestMergedPlanTables:
    def test_it_joins_disjoint_tables(self) -> None:
        base = _base()

        merged = merged_plan_tables({"a": base}, {"b": base})

        assert merged == {"a": base, "b": base}

    def test_a_name_defined_twice_is_refused_rather_than_overwritten(self) -> None:
        """A silent overwrite would file one plan's numbers under another's name."""
        base = _base()
        other: QaPlan = {**base, "epochs": 1}

        with pytest.raises(ValueError, match="plan 'a' is defined twice"):
            merged_plan_tables({"a": base}, {"a": other})
