"""Immutable issuer, accession and financial-period evidence records."""

from __future__ import annotations

import re
from math import isfinite

from platform_core.json_utils import JSONObject, require_float, require_str
from typing_extensions import ReadOnly, TypedDict

from .validation import parse_timestamp, require_bounded_int, require_day, require_nonempty


class Disclosure(TypedDict):
    """Issuer-specific filing identity and actual acceptance timestamp."""

    cik: ReadOnly[int]
    accession: ReadOnly[str]
    form: ReadOnly[str]
    filed: ReadOnly[str]
    accepted_at: ReadOnly[str]


class FinancialFact(TypedDict):
    """One accession's numeric statement for an exact tag, unit and period."""

    cik: ReadOnly[int]
    accession: ReadOnly[str]
    form: ReadOnly[str]
    filed: ReadOnly[str]
    tag: ReadOnly[str]
    unit: ReadOnly[str]
    start: ReadOnly[str | None]
    end: ReadOnly[str]
    value: ReadOnly[float]


def require_accession(document: JSONObject, key: str) -> str:
    """Require the native accession spelling without guessing its issuer.

    Args:
        document: Untrusted JSON object.
        key: Required accession field.

    Returns:
        Accession with ten, two and six digits separated by hyphens.

    Raises:
        ValueError: DISCLOSURE-001 for invalid accession spelling.
        TypeError: Missing or non-string accession.
    """
    accession = require_str(document, key)
    if re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession) is None:
        raise ValueError("[DISCLOSURE-001] accession must use ten-two-six digit groups")
    return accession


def encode_disclosure(record: Disclosure) -> JSONObject:
    """Encode a validated filing's identity and publication evidence.

    Args:
        record: Immutable disclosure.

    Returns:
        JSON filing object.
    """
    return {
        "cik": record["cik"],
        "accession": record["accession"],
        "form": record["form"],
        "filed": record["filed"],
        "accepted_at": record["accepted_at"],
    }


def decode_disclosure(document: JSONObject) -> Disclosure:
    """Validate a filing without replacing acceptance time with filing date.

    Args:
        document: Untrusted normalized filing object.

    Returns:
        Immutable disclosure with an aware acceptance timestamp.

    Raises:
        ValueError: Invalid issuer, accession, date or timestamp.
        TypeError: Missing or incorrectly typed fields.
    """
    accepted = require_nonempty(document, "accepted_at")
    parse_timestamp(accepted)
    return Disclosure(
        cik=require_bounded_int(document, "cik", 1, 9999999999),
        accession=require_accession(document, "accession"),
        form=require_nonempty(document, "form"),
        filed=require_day(document, "filed"),
        accepted_at=accepted,
    )


def encode_financial_fact(record: FinancialFact) -> JSONObject:
    """Encode a fact while preserving instant versus duration semantics.

    Args:
        record: Immutable financial observation.

    Returns:
        JSON record with an explicit nullable period start.
    """
    return {
        "cik": record["cik"],
        "accession": record["accession"],
        "form": record["form"],
        "filed": record["filed"],
        "tag": record["tag"],
        "unit": record["unit"],
        "start": record["start"],
        "end": record["end"],
        "value": record["value"],
    }


def decode_financial_fact(document: JSONObject) -> FinancialFact:
    """Require finite financial evidence and an explicit exact period.

    Args:
        document: Untrusted normalized fact object.

    Returns:
        Immutable monetary or other numeric statement in the declared unit.

    Raises:
        ValueError: DISCLOSURE-002 for absent start, invalid chronology or
            nonfinite value; shared validators identify malformed identities.
        TypeError: Missing or incorrectly typed fields.
    """
    if "start" not in document:
        raise ValueError("[DISCLOSURE-002] period start must be explicit, including null")
    start = None if document["start"] is None else require_day(document, "start")
    end, filed = require_day(document, "end"), require_day(document, "filed")
    value = require_float(document, "value")
    if not isfinite(value) or end > filed or (start is not None and start > end):
        raise ValueError("[DISCLOSURE-002] fact must be finite with start <= end <= filed")
    return FinancialFact(
        cik=require_bounded_int(document, "cik", 1, 9999999999),
        accession=require_accession(document, "accession"),
        form=require_nonempty(document, "form"),
        filed=filed,
        tag=require_nonempty(document, "tag"),
        unit=require_nonempty(document, "unit"),
        start=start,
        end=end,
        value=value,
    )
