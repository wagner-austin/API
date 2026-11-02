"""Real encrypted extraction and metadata-only audit for cohort derivation tests."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from platform_core.json_utils import JSONObject, dump_json_str

from covenant_ml.markets.crsp_cohort_records import CrspCohortPlan, decode_crsp_cohort_plan
from covenant_ml.markets.crsp_queries import crsp_scopes

from ._crsp_database import database_protocol, partition_rows
from ._crsp_extraction_fixtures import (
    Extraction,
    FakeDatabase,
    database_rows,
    extraction,
    installed,
    limits,
    run_vendor,
)


class CohortCase(NamedTuple):
    """One frozen validation bundle and a two-case audit anchored to one security."""

    extraction: Extraction
    plan: CrspCohortPlan
    audit: Path
    document: JSONObject
    rows: dict[str, list[JSONObject]]
    destination: Path


def cohort_case(root: Path) -> CohortCase:
    """Write a full fixture extraction with real encryption and two private-row anchors."""
    case = extraction(root)
    with installed(FakeDatabase(database_rows())):
        receipt = run_vendor(case, limits())
    protocol = database_protocol()
    rows = partition_rows(database_rows(), protocol, crsp_scopes(protocol)[0])
    document: JSONObject = {
        "results": [
            {
                "kind": "licensed-v3-source-path-audit",
                "bundle_sha256": receipt["train_validation"]["bundle"]["sha256"],
                "all_paths_accounted": True,
                "eligible_paths": 20,
                "accepted_paths": 18,
                "refused_paths": 2,
                "refused_cases": [
                    {
                        "path_ordinal": ordinal,
                        "period": "training",
                        "entry": {
                            "present": True,
                            "row_sha256": sha256(dump_json_str(row).encode()).hexdigest(),
                        },
                    }
                    for ordinal, row in enumerate(rows["returns"][:2], start=1)
                ],
            }
        ]
    }
    audit = root / "audit.json"
    payload = dump_json_str(document).encode()
    audit.write_bytes(payload)
    plan = decode_crsp_cohort_plan(
        {
            "bundle_sha256": receipt["train_validation"]["bundle"]["sha256"],
            "protocol_sha256": receipt["protocol"]["sha256"],
            "audit_sha256": sha256(payload).hexdigest(),
            "expected_cases": 2,
            "max_source_rows": 20000,
            "max_cohort_bytes": 4_000_000,
        }
    )
    return CohortCase(case, plan, audit, document, rows, root / "cohort")


__all__ = ["CohortCase", "cohort_case"]
