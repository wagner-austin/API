"""Actual BLS HTML and targeted mutations prove explicit native field boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from covenant_ml.markets.bls_records import encode_cpi_release
from covenant_ml.markets.bls_release import decode_bls_release

from .test_bls_records import release_document

RECEIVED = "2026-10-14T06:43:48.157563Z"
TABLE = '<table class="regular" id="cpi_pressa">'
CELL = '<td headers="cpi_pressa.r.1 cpi_pressa.h.1.9"><span class="datavalue">3.4</span></td>'


def native_html() -> str:
    """Read the complete captured native summary, including unrelated page content."""
    return (Path(__file__).parent / "fixtures" / "bls_august_2026.html").read_text("utf-8")


def test_actual_native_release_and_exact_publication_boundary() -> None:
    for receipt in (RECEIVED, "2026-09-11T12:30:00Z"):
        assert (
            encode_cpi_release(
                decode_bls_release(
                    native_html(),
                    target_month="2026-08",
                    received_at=receipt,
                )
            )
            == release_document()
        )


def test_publication_must_precede_receipt() -> None:
    with pytest.raises(ValueError, match="CPI-007"):
        decode_bls_release(
            native_html(), target_month="2026-08", received_at="2026-09-11T12:29:59Z"
        )


def test_current_august_release_cannot_score_september() -> None:
    with pytest.raises(ValueError, match="CPI-005"):
        decode_bls_release(native_html(), target_month="2026-09", received_at=RECEIVED)


@pytest.mark.parametrize(
    "old,new,code",
    [
        (TABLE, TABLE + "<table>", "CPI-003"),
        (TABLE, TABLE.replace('id="cpi_pressa"', 'id="cpi_pressa" id="cpi_pressa"'), "CPI-003"),
        ('id="cpi_pressa.r.1"', 'id="cpi_pressa.r.1" id="cpi_pressa.r.1"', "CPI-003"),
        ('<p class="sub0">All items</p>', "<td>nested</td>", "CPI-003"),
        ('<p class="sub0">All items</p>', "</table>", "CPI-003"),
        (TABLE, '<table id="different">', "CPI-003"),
        ("Transmission of material in this release", "Different publication preface", "CPI-004"),
        ("CONSUMER PRICE INDEX - AUGUST 2026", "CONSUMER PRICE INDEX - JULY 2026", "CPI-005"),
        ("8:30 a.m. (ET) Friday", "9:30 a.m. (ET) Friday", "CPI-005"),
        ("Friday, September 11, 2026", "Thursday, September 11, 2026", "CPI-005"),
        ("Friday, September 11, 2026", "Friday, Unknown 11, 2026", "CPI-005"),
        (
            "Table A. Percent changes in CPI for All Urban Consumers (CPI-U)",
            "Table A. Percent changes in CPI for All Urban Consumers (CPI-W)",
            "CPI-006",
        ),
        ('<p class="sub0">All items</p>', '<p class="sub0">Core items</p>', "CPI-006"),
        ("ended<br />Aug. 2026", "ended<br />Jul. 2026", "CPI-006"),
        (CELL, CELL.replace(">3.4<", ">3.45<"), "CPI-006"),
        (CELL, CELL.replace(">3.4<", ">NaN<"), "CPI-006"),
        (CELL, "", "CPI-004"),
        (CELL, CELL + CELL, "CPI-004"),
        ('id="cpi_pressa.r.1"', 'id="wrong-row"', "CPI-004"),
        ('id="cpi_pressa.h.1.9"', 'id="wrong-column"', "CPI-004"),
    ],
)
def test_native_structure_and_identity_mutations(old: str, new: str, code: str) -> None:
    source = native_html()
    assert source.count(old) == 1
    with pytest.raises(ValueError, match=code):
        decode_bls_release(source.replace(old, new), target_month="2026-08", received_at=RECEIVED)


@pytest.mark.parametrize(
    "change", ["duplicate_table", "duplicate_header", "open_pre", "open_table"]
)
def test_duplicate_and_unclosed_documents(change: str) -> None:
    source = native_html()
    start = source.index(TABLE)
    end = source.index("</table>", start)
    if change == "duplicate_table":
        source += source[start : end + len("</table>")]
    elif change == "duplicate_header":
        header_start = source.index("<PRE>")
        source += source[header_start : source.index("</PRE>", header_start) + len("</PRE>")]
    elif change == "open_pre":
        source += "<pre>unfinished"
    else:
        source = source[:end]
    with pytest.raises(ValueError, match="CPI-004" if change == "duplicate_header" else "CPI-003"):
        decode_bls_release(source, target_month="2026-08", received_at=RECEIVED)


def test_may_column_and_negative_reported_decimal() -> None:
    source = native_html().replace(
        "CONSUMER PRICE INDEX - AUGUST 2026", "CONSUMER PRICE INDEX - MAY 2026"
    )
    source = source.replace("ended<br />Aug. 2026", "ended<br />May 2026")
    source = source.replace(CELL, CELL.replace(">3.4<", ">-0.1<"))
    expected = dict(release_document(), target_month="2026-05", annual_change=-0.1)
    assert (
        encode_cpi_release(
            decode_bls_release(
                source,
                target_month="2026-05",
                received_at=RECEIVED,
            )
        )
        == expected
    )
