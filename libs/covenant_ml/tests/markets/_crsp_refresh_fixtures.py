"""Real encrypted diagnostic inputs and the explicit database seam for refresh tests."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from platform_core.json_utils import JSONObject, require_dict, require_str

from covenant_ml.markets.crsp_cohort import derive_crsp_cohort
from covenant_ml.markets.crsp_extraction_records import CrspLimits
from covenant_ml.markets.crsp_refresh import compare_crsp_vintages

from ._crsp_cohort_fixtures import CohortCase, cohort_case
from ._crsp_extraction_fixtures import FakeDatabase, database_rows, installed
from ._encrypted_fixtures import evaluator_keys


class RefreshCase(NamedTuple):
    """Pinned original cohort and exclusive fresh-comparison destination."""

    cohort: CohortCase
    digest: str
    destination: Path


def refresh_case(root: Path) -> RefreshCase:
    """Derive a real encrypted cohort while using the fixed fixture clock."""
    case = cohort_case(root)
    with installed(FakeDatabase(database_rows())):
        result = derive_crsp_cohort(
            case.extraction.output / "train-validation" / "bundle.json",
            case.audit,
            case.plan,
            evaluator_keys(root),
            scratch=case.extraction.scratch,
            destination=case.destination,
        )
    return RefreshCase(case, require_str(require_dict(result, "bundle"), "sha256"), root / "fresh")


def refresh(case: RefreshCase, bounds: CrspLimits, setup: Path | None = None) -> JSONObject:
    """Run the actual comparison using the independently pinned fixture lineage."""
    cohort = case.cohort
    return compare_crsp_vintages(
        cohort.destination / "bundle.json",
        case.digest,
        cohort.plan,
        evaluator_keys(cohort.extraction.root),
        scratch=cohort.extraction.scratch,
        destination=case.destination,
        limits=bounds,
        setup=setup,
    )


__all__ = ["RefreshCase", "refresh", "refresh_case"]
