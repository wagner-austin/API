"""The declared GEMM shape tables, held to what their origins claim.

Split from ``test_gemm_probe.py`` when that file crossed the size ceiling:
these classes read the DECLARED tables and import no torch, while that file
runs the probe. The split follows the same line the source modules draw
between :mod:`gemm_shapes` and :mod:`gemm_probe`.
"""

from __future__ import annotations

import pytest

from model_trainer.core.services.model.gemm_shapes import (
    BOUNDARY_INNER,
    BOUNDARY_INNERS,
    BOUNDARY_NAME,
    BOUNDARY_ROW,
    BOUNDARY_ROWS,
    DIGEST_SUFFIX,
    GEMM_BATCHED,
    GEMM_BOUNDARY,
    GEMM_COLS,
    GEMM_CROSSOVER,
    GEMM_SHAPES,
    GEMM_SWEEP,
    SUM_SUFFIX,
    SWEEP_INNERS,
    SWEEP_NAME,
    SWEEP_ROWS,
    GemmShape,
    gemm_label,
    probed_shapes,
    require_unique_labels,
)

#: A shape in the same orientation as the real ones, for label tests.
SMALL: GemmShape = {"rows": 8, "inner": 16, "cols": 4, "origin": "test"}


class TestTheShapeTable:
    def test_every_shape_has_positive_dimensions(self) -> None:
        bad = [
            name for name, s in GEMM_SHAPES.items() if min(s["rows"], s["inner"], s["cols"]) <= 0
        ]

        assert bad == []

    def test_every_qkv_shape_is_three_times_its_hidden_size(self) -> None:
        # GPT-2's c_attn projects hidden -> 3*hidden, one matmul for Q, K and
        # V together. A QKV entry with M != 3K would be measuring a shape the
        # model does not issue, which is the failure the origin strings claim
        # these exist to end.
        qkv = {name: s for name, s in GEMM_SHAPES.items() if name.endswith("-attn-qkv")}
        wrong = {name for name, s in qkv.items() if s["rows"] != 3 * s["inner"]}

        assert wrong == set()
        assert sorted(qkv) == [
            "large-attn-qkv",
            "medium-attn-qkv",
            "small-attn-qkv",
            "tiny-attn-qkv",
            "xl-attn-qkv",
        ]

    def test_the_qkv_shape_the_trace_broke_on_is_present(self) -> None:
        # transformer.h.0.attn.c_attn at the large rung: hidden 1280, so
        # M=3840 K=1280. This is the v24 four-card trace's first divergence
        # and the one shape no earlier table contained.
        assert (GEMM_SHAPES["large-attn-qkv"]["rows"], GEMM_SHAPES["large-attn-qkv"]["inner"]) == (
            3840,
            1280,
        )

    def test_every_shape_shares_the_ladder_sequence_length(self) -> None:
        # The ladder runs one sequence of the gate rung's length, so a shape
        # with a different N is not a call the ladder ever issued and cannot
        # be read against a rung.
        assert [s["cols"] for s in GEMM_SHAPES.values()] == [GEMM_COLS for _ in GEMM_SHAPES]

    def test_every_shape_says_where_it_came_from(self) -> None:
        # A shape with no origin is a number nobody can trace to a rung.
        assert [n for n, s in GEMM_SHAPES.items() if not s["origin"].strip()] == []

    def test_labels_are_unique_across_shapes_and_measurements(self) -> None:
        labels = [
            gemm_label(n, s, suffix)
            for n, s in GEMM_SHAPES.items()
            for suffix in (DIGEST_SUFFIX, SUM_SUFFIX)
        ]

        assert sorted(labels) == sorted(set(labels))

    def test_a_label_carries_the_dimensions_and_the_measurement(self) -> None:
        assert (
            gemm_label("medium-mlp-proj", GEMM_SHAPES["medium-mlp-proj"], DIGEST_SUFFIX)
            == "gemm-medium-mlp-proj-M1024-K4096-N64|digest48"
        )

    def test_resizing_a_shape_renames_its_measurements(self) -> None:
        # The property that keeps a re-dimensioned probe from recording under
        # a name whose earlier value came from different arithmetic.
        widened: GemmShape = {**SMALL, "inner": SMALL["inner"] * 2}

        assert gemm_label("x", SMALL, SUM_SUFFIX) != gemm_label("x", widened, SUM_SUFFIX)


class TestTheSweepGrid:
    """The sweep exists because the ladder's shapes could not answer.

    Of the 32 shapes the ladder issues, exactly TWO drew the same algorithm on
    the V100 and the A30 -- so "same kernel implies same result" rested on one
    instance. A grid does not choose which cell of the 2x2 a point lands in.
    """

    def test_it_is_the_full_cross_of_the_declared_lists(self) -> None:
        assert len(GEMM_SWEEP) == len(SWEEP_ROWS) * len(SWEEP_INNERS)

    def test_every_declared_pair_is_present(self) -> None:
        present = {(s["rows"], s["inner"]) for s in GEMM_SWEEP}

        assert present == {(r, k) for r in SWEEP_ROWS for k in SWEEP_INNERS}

    def test_it_reaches_the_small_inner_where_the_cards_already_agreed(self) -> None:
        # Both shapes where the V100 and A30 chose alike had K=128. A sweep
        # that skipped it would have re-created the shortage it exists to fix.
        assert 128 in SWEEP_INNERS

    def test_it_spans_more_than_one_order_of_magnitude_in_the_summed_dimension(self) -> None:
        # K is what split-K partitions, so it is the axis a device-dependent
        # reduction order enters through; a grid clustered at one K could not
        # separate "same kernel" from "small enough not to matter".
        assert max(SWEEP_INNERS) >= 16 * min(SWEEP_INNERS)

    def test_every_probed_shape_gets_its_own_label(self) -> None:
        # Two entries sharing a label would silently drop an observation --
        # and `run_record` would reject the duplicate name much further from
        # the cause. `probed_shapes` refuses first.
        labels = [gemm_label(n, s, DIGEST_SUFFIX) for n, s in probed_shapes()]

        assert sorted(labels) == sorted(set(labels))

    def test_the_whole_grid_shares_one_name_so_labels_stay_readable(self) -> None:
        # Per-point names encoding the dimensions produced
        # `gemm-sweep-M1024-K1024-M1024-K1024-N64`, since gemm_label appends
        # them anyway. One name for the grid keeps the label honest.
        assert gemm_label(SWEEP_NAME, GEMM_SWEEP[0], DIGEST_SUFFIX).count("-M") == 1

    def test_probed_shapes_carries_every_table(self) -> None:
        # Including the two tables declared for the TIMING benchmark: until
        # 2026-08-31 every digest ever compared ran at N=64, so agreement at
        # a real batch dimension was unmeasured, not established.
        assert len(probed_shapes()) == (
            len(GEMM_SHAPES)
            + len(GEMM_SWEEP)
            + len(GEMM_BOUNDARY)
            + len(GEMM_BATCHED)
            + len(GEMM_CROSSOVER)
        )

    def test_the_real_tables_pass_the_label_check(self) -> None:
        assert require_unique_labels(probed_shapes()) == probed_shapes()


class TestTheBoundaryBracket:
    """Two lines through the shape the v24 four-card trace broke on.

    The trace's rungs move M and K together -- QKV is always M=3K -- so they
    cannot say which axis carries the break. These can.
    """

    def test_it_holds_one_axis_on_each_line(self) -> None:
        on_k_line = {s["rows"] for s in GEMM_BOUNDARY if s["inner"] != BOUNDARY_INNER}
        on_m_line = {s["inner"] for s in GEMM_BOUNDARY if s["rows"] != BOUNDARY_ROW}

        assert on_k_line == {BOUNDARY_ROW}
        assert on_m_line == {BOUNDARY_INNER}

    def test_every_declared_point_is_present(self) -> None:
        present = {(s["rows"], s["inner"]) for s in GEMM_BOUNDARY}
        declared = {(rows, BOUNDARY_INNER) for rows in BOUNDARY_ROWS}
        declared |= {(BOUNDARY_ROW, inner) for inner in BOUNDARY_INNERS}

        assert present == declared

    def test_the_crossing_point_is_emitted_once(self) -> None:
        # The lines are DEFINED to meet at the shape under study. Emitting it
        # twice would make `require_unique_labels` refuse the whole table.
        points = [(s["rows"], s["inner"]) for s in GEMM_BOUNDARY]

        assert points.count((BOUNDARY_ROW, BOUNDARY_INNER)) == 1
        assert len(points) == len(set(points))

    def test_it_brackets_the_shape_that_broke(self) -> None:
        # large-attn-qkv is M=3840 K=1280, the v24 trace's first divergence.
        assert (BOUNDARY_ROW, BOUNDARY_INNER) == (3840, 1280)
        assert min(BOUNDARY_INNERS) < BOUNDARY_INNER < max(BOUNDARY_INNERS)
        assert min(BOUNDARY_ROWS) < BOUNDARY_ROW < max(BOUNDARY_ROWS)

    def test_it_spans_the_two_rungs_that_disagree_about_agreeing(self) -> None:
        # medium (K=1024) agreed on all four cards; large (K=1280) did not.
        # A bracket that did not contain both endpoints could not locate the
        # boundary between them, only confirm it exists.
        assert 1024 in BOUNDARY_INNERS
        assert 1280 in BOUNDARY_INNERS

    def test_it_reaches_a_k_no_power_of_two_grid_contains(self) -> None:
        # The sweep is powers of two. 1152 and 1408 are the half-multiples of
        # 256 either side of 1280, which is where the break first shows.
        assert 1152 in BOUNDARY_INNERS
        assert 1408 in BOUNDARY_INNERS
        assert not set(BOUNDARY_INNERS) <= set(SWEEP_INNERS)

    def test_the_whole_bracket_shares_one_name(self) -> None:
        assert gemm_label(BOUNDARY_NAME, GEMM_BOUNDARY[0], DIGEST_SUFFIX).count("-M") == 1

    def test_two_entries_sharing_a_label_are_refused(self) -> None:
        twin = (("dup", SMALL), ("dup", SMALL))

        with pytest.raises(ValueError, match="share a label"):
            require_unique_labels(twin)

    def test_one_shape_under_two_names_is_not_a_label_collision(self) -> None:
        # The overlap the tables deliberately have: same dimensions, different
        # names, so the labels differ and both survive to be measured.
        pairs = (("a", SMALL), ("b", SMALL))

        assert require_unique_labels(pairs) == pairs
