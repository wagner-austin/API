"""The search's pure half: candidates, scoring, halving, all deterministic.

Two runs of one search must agree move for move -- a stochastic walk
whose results cannot be replayed would fail law nine before it started
-- so every test here pins exact outputs, not shapes.
"""

from __future__ import annotations

import pytest

from rw_bot.harness.search import (
    Candidate,
    SearchError,
    apply_moves,
    candidate_label,
    keep_top,
    paired_delta,
    sampled_pairs,
    single_moves,
)
from rw_bot.policy.doctrine import DEFAULT_DOCTRINE


def test_labels_name_the_moves_in_order() -> None:
    assert candidate_label((("raid", 5), ("tech", 2))) == "raid5-tech2"
    assert candidate_label((("flame", 0),)) == "flame0"


def test_single_moves_cover_the_space_in_field_order() -> None:
    space = {"tech": (2,), "raid": (3, 5)}
    assert single_moves(space) == (
        (("raid", 3),),
        (("raid", 5),),
        (("tech", 2),),
    )


def test_sampled_pairs_are_deterministic_and_cross_fields() -> None:
    space = {"raid": (3, 5), "tech": (2,), "flame": (0,)}
    first = sampled_pairs(space, 3, seed=7)
    again = sampled_pairs(space, 3, seed=7)
    assert first == again
    assert len(first) == 3
    for moves in first:
        fields = [field for field, _ in moves]
        assert len(fields) == 2
        assert fields[0] != fields[1]


def test_sampling_more_than_exists_returns_the_whole_pool() -> None:
    space = {"raid": (3,), "tech": (2,)}
    assert sampled_pairs(space, 99, seed=1) == (((("raid", 3)), ("tech", 2)),)


def test_moves_apply_onto_the_base_and_rename_the_variant() -> None:
    variant = apply_moves(DEFAULT_DOCTRINE, (("raid", 5), ("tech", 2)))
    assert variant["raid"] == 5
    assert variant["tech"] == 2
    assert variant["name"] == "raid5-tech2"
    assert variant["flame"] == DEFAULT_DOCTRINE["flame"]
    assert DEFAULT_DOCTRINE["name"] != "raid5-tech2"


def test_a_typoed_knob_stops_the_search() -> None:
    with pytest.raises(SearchError) as caught:
        apply_moves(DEFAULT_DOCTRINE, (("riad", 5),))
    assert caught.value.code == "RW-SEARCH-001"


def test_paired_delta_summarizes_shared_seeds_only() -> None:
    margins = {
        "control": {1: 2.0, 2: -1.0, 3: 1.0},
        "arm": {1: 3.0, 2: -2.0, 9: 2.5},
    }
    pairs, mean, sd = paired_delta(margins, "arm", "control")
    assert pairs == 2
    assert mean == 0.0
    assert sd == 1.0
    assert paired_delta(margins, "ghost", "control") == (0, 0.0, 0.0)


def test_the_halving_keeps_the_best_with_stable_ties() -> None:
    a: Candidate = (("raid", 3),)
    b: Candidate = (("raid", 5),)
    c: Candidate = (("tech", 2),)
    scores: dict[Candidate, float] = {a: 0.5, b: 0.5, c: -1.0}
    assert keep_top(scores, 1) == (a,)
    assert keep_top(scores, 2) == (a, b)
    assert keep_top(scores, 0) == ()
