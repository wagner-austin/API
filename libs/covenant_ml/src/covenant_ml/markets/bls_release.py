"""Strict publication and annual all-items CPI-U extraction from native BLS HTML."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from ._bls_html import ReleaseHTML
from .bls_records import CPIRelease, decode_cpi_release
from .validation import parse_timestamp, require_day, require_nonempty

_MONTHS: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_WEEKDAYS: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_CAPTION = "Table A. Percent changes in CPI for All Urban Consumers (CPI-U): U.S. city average"


def _require_single(values: tuple[str, ...], label: str) -> str:
    """Require one nonempty native field rather than selecting among duplicates.

    Args:
        values: Matched native field values.
        label: Traceable field identity.

    Returns:
        Sole validated field.

    Raises:
        ValueError: CPI-004 for absent or ambiguous native fields.
    """
    if len(values) != 1:
        raise ValueError(f"[CPI-004] require exactly one {label}")
    return require_nonempty({label: values[0]}, label)


def _publication(header: str, expected_target: str) -> tuple[str, str]:
    """Resolve the native Eastern release date and verify its target headline.

    Args:
        header: Normalized native release PRE text.
        expected_target: Validated canonical target month.

    Returns:
        UTC publication time and native publication identifier.

    Raises:
        ValueError: CPI-005 for malformed header, target, weekday or identifier.
            Invalid native calendar dates also propagate.
    """
    headings = tuple(
        (match.group(1), match.group(2))
        for match in re.finditer(r"CONSUMER PRICE INDEX - ([A-Z]+) ([0-9]{4})", header)
    )
    stamps: list[tuple[str, str, str, str, str]] = [
        (match.group(1), match.group(2), match.group(3), match.group(4), match.group(5))
        for match in re.finditer(
            r"Transmission of material in this release is embargoed until "
            r"8:30 a\.m\. \(ET\) ([A-Za-z]+), ([A-Za-z]+) ([0-9]{1,2}), ([0-9]{4}) "
            r"(USDL-[0-9]{2}-[0-9]{4})",
            header,
        )
    ]
    target_date = datetime.fromisoformat(expected_target + "-01")
    if (
        headings != ((_MONTHS[target_date.month - 1].upper(), str(target_date.year)),)
        or len(stamps) != 1
    ):
        raise ValueError("[CPI-005] target headline or publication header is inconsistent")
    weekday, month, day, year, identifier = stamps[0]
    if month not in _MONTHS:
        raise ValueError("[CPI-005] unsupported publication month name")
    published = datetime(
        int(year), _MONTHS.index(month) + 1, int(day), 8, 30, tzinfo=ZoneInfo("America/New_York")
    )
    if weekday != _WEEKDAYS[published.weekday()]:
        raise ValueError("[CPI-005] publication weekday differs from its calendar date")
    return published.astimezone(UTC).isoformat(), identifier


def decode_bls_release(html: str, *, target_month: str, received_at: str) -> CPIRelease:
    """Read the explicitly published annual CPI-U field for one required target.

    Args:
        html: Exact decoded native summary or dated archive HTML.
        target_month: Required canonical YYYY-MM identity, never inferred from prices.
        received_at: Actual aware response receipt timestamp.

    Returns:
        Immutable directly reported annual CPI-U release.

    Raises:
        ValueError: CPI-003/CPI-004 for structural ambiguity; CPI-005 for header
            drift; CPI-006 for a changed annual column, row or decimal value;
            CPI-007 for receipt before publication. Codec errors propagate.
    """
    require_day({"day": target_month + "-01"}, "day")
    received = parse_timestamp(received_at)
    parser = ReleaseHTML()
    parser.feed(html)
    parser.close()
    parser.require_complete()
    header = _require_single(
        tuple(
            text
            for tag, _, text in parser.elements
            if tag == "pre" and "Transmission of material in this release" in text
        ),
        "release header",
    )
    published, identifier = _publication(header, target_month)
    caption = _require_single(
        tuple(text for tag, _, text in parser.elements if tag == "caption"), "Table A caption"
    )
    row = _require_single(
        tuple(
            text
            for tag, attrs, text in parser.elements
            if tag == "th" and attrs.get("id") == "cpi_pressa.r.1"
        ),
        "all-items row",
    )
    column = _require_single(
        tuple(
            text
            for tag, attrs, text in parser.elements
            if tag == "th" and attrs.get("id") == "cpi_pressa.h.1.9"
        ),
        "annual column",
    )
    value = _require_single(
        tuple(
            text
            for tag, attrs, text in parser.elements
            if tag == "td" and attrs.get("headers") == "cpi_pressa.r.1 cpi_pressa.h.1.9"
        ),
        "annual CPI-U cell",
    )
    month_index = int(target_month[5:7]) - 1
    abbreviation = _MONTHS[month_index][:3] + ("" if month_index == 4 else ".")
    expected_column = f"Un- adjusted 12-mos. ended {abbreviation} {target_month[:4]}"
    if (
        caption != _CAPTION
        or row != "All items"
        or column != expected_column
        or re.fullmatch(r"-?[0-9]+\.[0-9]", value) is None
    ):
        raise ValueError("[CPI-006] CPI-U identity, annual target column or decimal field changed")
    release = decode_cpi_release(
        {
            "target_month": target_month,
            "published_at": published,
            "release_id": identifier,
            "annual_change": float(value),
        }
    )
    if received < parse_timestamp(published):
        raise ValueError("[CPI-007] response receipt precedes official publication")
    return release
