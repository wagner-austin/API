"""Exact row accounting over real decoded source projections, without effect seams."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.crsp_comparison import CrspRowDifference, compare_crsp_rows
from covenant_ml.markets.crsp_rows import crsp_row_key, decode_crsp_rows, encode_crsp_rows


def _rows(relation: str) -> tuple[JSONObject, ...]:
    """Decode four distinct rows using the production CSV codec."""
    examples: dict[str, JSONObject] = {
        "membership": {
            "permno": 1,
            "namedt": "2018-01-01",
            "nameendt": "2019-12-31",
            "shrcd": 10,
            "exchcd": 1,
        },
        "returns": {
            "permno": 1,
            "dlycaldt": "2019-06-03",
            "dlyprevdt": "2019-05-31",
            "dlyret": 0.1,
            "dlyretmissflg": "NA",
            "dlyretdurflg": "D1",
            "dlydelflg": "N",
            "dlyprc": 10.0,
            "dlyprcflg": "TR",
            "dlyvol": 100.0,
        },
        "calendar": {"caldt": "2019-06-03", "exchangegroup": "CIZ", "tradingflg": "Y"},
        "delists": {
            "permno": 1,
            "delistingdt": "2019-06-03",
            "deldlydt": "2019-06-04",
            "delamtdt": None,
            "delnextdt": None,
            "delret": -0.5,
            "delretmisstype": "NA",
        },
    }
    base = examples[relation]
    rows = tuple(
        {**base, "caldt": f"2019-06-0{index}"}
        if relation == "calendar"
        else {**base, "permno": index}
        for index in range(1, 5)
    )
    return _decoded(rows, relation)


def _decoded(rows: tuple[JSONObject, ...], relation: str) -> tuple[JSONObject, ...]:
    """Use the actual reader, retaining explicit nullable values."""
    columns = tuple(rows[0])
    return decode_crsp_rows(
        encode_crsp_rows(rows, columns), relation, columns, "2026-10-31T00:00:00Z"
    )


@pytest.mark.parametrize("relation", ["membership", "returns", "calendar", "delists"])
def test_every_row_is_accounted_and_order_does_not_change_the_result(relation: str) -> None:
    rows = _rows(relation)
    edits: dict[str, JSONObject] = {
        "membership": {"shrcd": 11},
        "returns": {"dlyret": 0.2},
        "calendar": {"exchangegroup": "N/A", "tradingflg": "N"},
        "delists": {"delret": -0.25},
    }
    changed = _decoded(({**rows[1], **edits[relation]},), relation)[0]
    old = rows[:3]
    fresh = (rows[3], changed, rows[0])
    expected = CrspRowDifference(1, (rows[3],), (rows[2],), ((rows[1], changed),))
    result = compare_crsp_rows(old, fresh, relation)
    assert result == expected
    assert compare_crsp_rows(tuple(reversed(old)), tuple(reversed(fresh)), relation) == expected
    assert result.unchanged + len(result.removed) + len(result.changed) == len(old)
    assert result.unchanged + len(result.added) + len(result.changed) == len(fresh)
    assert rows == _rows(relation)
    assert compare_crsp_rows(rows, rows, relation) == CrspRowDifference(4, (), (), ())


@pytest.mark.parametrize(
    ("relation", "field", "value"),
    [
        ("membership", "permno", 9),
        ("membership", "namedt", "2018-01-02"),
        ("membership", "nameendt", "2019-12-30"),
        ("returns", "permno", 9),
        ("returns", "dlycaldt", "2019-06-04"),
        ("calendar", "caldt", "2019-06-05"),
        ("delists", "permno", 9),
        ("delists", "delistingdt", "2019-06-02"),
    ],
)
def test_editing_each_natural_key_field_means_removal_and_addition(
    relation: str, field: str, value: JSONValue
) -> None:
    old = _rows(relation)[:1]
    fresh = _decoded(({**old[0], field: value},), relation)
    assert compare_crsp_rows(old, fresh, relation) == CrspRowDifference(0, fresh, old, ())


@pytest.mark.parametrize("relation", ["returns", "delists"])
@pytest.mark.parametrize("value", [0.0, 0.00000000000001])
def test_null_and_small_numeric_changes_are_not_imputed_or_rounded(
    relation: str, value: float
) -> None:
    field = "dlyret" if relation == "returns" else "delret"
    row = _rows(relation)[0]
    missing = _decoded(({**row, field: None},), relation)
    zero = _decoded(({**row, field: 0.0},), relation)
    numeric = _decoded(({**row, field: value},), relation)
    assert compare_crsp_rows(missing, numeric, relation) == CrspRowDifference(
        0, (), (), ((missing[0], numeric[0]),)
    )
    expected = (
        CrspRowDifference(1, (), (), ())
        if value == 0
        else CrspRowDifference(0, (), (), ((zero[0], numeric[0]),))
    )
    assert compare_crsp_rows(zero, numeric, relation) == expected


@pytest.mark.parametrize("relation", ["membership", "returns", "calendar", "delists"])
@pytest.mark.parametrize("duplicate_old", [True, False])
def test_duplicates_on_either_side_are_refused_even_when_the_rows_are_identical(
    relation: str, duplicate_old: bool
) -> None:
    rows = _rows(relation)[:1]
    duplicate = rows + rows
    with pytest.raises(ValueError, match="CRSP_COMPARISON_KEY: duplicate source key"):
        compare_crsp_rows(
            duplicate if duplicate_old else rows,
            rows if duplicate_old else duplicate,
            relation,
        )


def test_empty_relations_are_explicit_and_unknown_roles_are_refused() -> None:
    rows = _rows("delists")
    assert compare_crsp_rows((), (), "delists") == CrspRowDifference(0, (), (), ())
    assert compare_crsp_rows((), rows, "delists") == CrspRowDifference(0, rows, (), ())
    assert compare_crsp_rows(rows, (), "delists") == CrspRowDifference(0, (), rows, ())
    with pytest.raises(ValueError, match="CRSP_COMPARISON_RELATION: require"):
        compare_crsp_rows((), (), "unknown")
    with pytest.raises(KeyError, match="unknown"):
        crsp_row_key({}, "unknown")


def test_key_spelling_matches_the_existing_source_identity() -> None:
    assert crsp_row_key(_rows("membership")[0], "membership") == (
        "1",
        "2018-01-01",
        "2019-12-31",
    )
    assert crsp_row_key(_rows("returns")[0], "returns") == ("1", "2019-06-03")
    assert crsp_row_key(_rows("calendar")[0], "calendar") == ("2019-06-01",)
    assert crsp_row_key(_rows("delists")[0], "delists") == ("1", "2019-06-03")
