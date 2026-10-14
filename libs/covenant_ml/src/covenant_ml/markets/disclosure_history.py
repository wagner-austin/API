"""Native disclosure decoding and revision-safe selection at a decision time."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_list,
    require_str,
)

from .disclosure_records import (
    Disclosure,
    FinancialFact,
    decode_disclosure,
    decode_financial_fact,
)
from .validation import parse_timestamp, require_bounded_int, require_day, require_nonempty


def decode_submission_columns(document: JSONObject, cik: int) -> tuple[Disclosure, ...]:
    """Decode either explicitly selected recent or historical submission columns.

    Args:
        document: Native parallel-column object, not its recent-filings wrapper.
        cik: Issuer established by the requested submissions document.

    Returns:
        All disclosures in source order, with unique accessions.

    Raises:
        ValueError: DISCLOSURE-003 for unequal array lengths or duplicate
            accessions; normalized record failures propagate.
        TypeError: Missing or malformed native arrays or fields.
    """
    issuer = require_bounded_int({"cik": cik}, "cik", 1, 9999999999)
    accessions = require_list(document, "accessionNumber")
    forms = require_list(document, "form")
    filed = require_list(document, "filingDate")
    accepted = require_list(document, "acceptanceDateTime")
    if len({len(accessions), len(forms), len(filed), len(accepted)}) != 1:
        raise ValueError("[DISCLOSURE-003] native submission arrays must have equal lengths")
    records = tuple(
        decode_disclosure(
            {
                "cik": issuer,
                "accession": accession,
                "form": forms[index],
                "filed": filed[index],
                "accepted_at": accepted[index],
            }
        )
        for index, accession in enumerate(accessions)
    )
    if len({record["accession"] for record in records}) != len(records):
        raise ValueError("[DISCLOSURE-003] native submission accessions must be unique")
    return records


def decode_company_facts(
    document: JSONObject, cik: int, taxonomy: str, tag: str, unit: str
) -> tuple[FinancialFact, ...]:
    """Decode one explicitly requested issuer, taxonomy, concept and unit.

    Args:
        document: Native companyfacts response.
        cik: Expected issuer from request identity.
        taxonomy: Exact taxonomy key, for example us-gaap.
        tag: Exact concept key, for example Assets.
        unit: Exact native unit; no conversion or alternative tag is attempted.

    Returns:
        Validated native observations, preserving accessions and duration starts.

    Raises:
        ValueError: DISCLOSURE-004 for issuer mismatch; fact validation failures
            propagate, including explicit null native duration starts.
        TypeError: Missing or malformed required concepts, units or fields.
    """
    issuer = require_bounded_int(document, "cik", 1, 9999999999)
    if issuer != require_bounded_int({"cik": cik}, "cik", 1, 9999999999):
        raise ValueError("[DISCLOSURE-004] company facts do not match requested issuer")
    namespace = require_nonempty({"taxonomy": taxonomy}, "taxonomy")
    concept = require_nonempty({"tag": tag}, "tag")
    measure = require_nonempty({"unit": unit}, "unit")
    native = require_dict(require_dict(require_dict(document, "facts"), namespace), concept)
    records: list[FinancialFact] = []
    for raw in require_list(require_dict(native, "units"), measure):
        row = narrow_json_to_dict(raw)
        start = require_str(row, "start") if "start" in row else None
        records.append(
            decode_financial_fact(
                {
                    "cik": issuer,
                    "accession": require_str(row, "accn"),
                    "form": require_str(row, "form"),
                    "filed": require_str(row, "filed"),
                    "tag": namespace + ":" + concept,
                    "unit": measure,
                    "start": start,
                    "end": require_str(row, "end"),
                    "value": require_float(row, "val"),
                }
            )
        )
    return tuple(records)


def select_disclosed_fact(
    facts: tuple[FinancialFact, ...],
    disclosures: tuple[Disclosure, ...],
    *,
    start: str | None,
    end: str,
    cutoff: str,
) -> tuple[FinancialFact, Disclosure] | None:
    """Select the latest accepted revision of one exact financial period.

    Args:
        facts: Observations for one issuer, namespaced concept and unit.
        disclosures: Complete accession metadata for every supplied observation.
        start: Exact duration start, or None for an instantaneous balance.
        end: Exact period end, never a substitute for disclosure time.
        cutoff: Aware decision timestamp; acceptance must be strictly earlier.

    Returns:
        Latest eligible fact and its filing, or None when none was available.

    Raises:
        ValueError: DISCLOSURE-005 for mixed identities or duplicate filings;
            DISCLOSURE-006 for absent or inconsistent accession metadata;
            DISCLOSURE-007 for duplicate observations or acceptance-time ties;
            DISCLOSURE-008 for invalid requested period chronology.
    """
    decision = parse_timestamp(cutoff)
    require_day({"end": end}, "end")
    if start is not None:
        require_day({"start": start}, "start")
        if start > end:
            raise ValueError("[DISCLOSURE-008] requested start must not follow end")
    identities = {(fact["cik"], fact["tag"], fact["unit"]) for fact in facts}
    filings = {(row["cik"], row["accession"]): row for row in disclosures}
    if len(identities) > 1 or len(filings) != len(disclosures):
        raise ValueError("[DISCLOSURE-005] require one fact identity and unique filings")
    seen: set[tuple[str, str | None, str]] = set()
    candidates: dict[int, tuple[FinancialFact, Disclosure]] = {}
    for fact in facts:
        identity = (fact["cik"], fact["accession"])
        if identity not in filings:
            raise ValueError("[DISCLOSURE-006] fact has no issuer-matched acceptance metadata")
        filing = filings[identity]
        if (fact["form"], fact["filed"]) != (filing["form"], filing["filed"]):
            raise ValueError("[DISCLOSURE-006] fact form or filed date differs from accession")
        key = (fact["accession"], fact["start"], fact["end"])
        if key in seen:
            raise ValueError("[DISCLOSURE-007] duplicate fact for accession and exact period")
        seen.add(key)
        accepted = parse_timestamp(filing["accepted_at"])
        if fact["start"] != start or fact["end"] != end or accepted >= decision:
            continue
        if accepted in candidates:
            raise ValueError("[DISCLOSURE-007] tied acceptance times have no declared ordering")
        candidates[accepted] = (fact, filing)
    return candidates[max(candidates)] if candidates else None
