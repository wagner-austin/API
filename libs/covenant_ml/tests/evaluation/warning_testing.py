"""Constructed dated cohorts for real warning workflow tests, never research evidence."""

from __future__ import annotations

from platform_core.json_utils import require_dict

from covenant_ml.evaluation.warning_folds import WarningFold
from covenant_ml.evaluation.warning_panel import WarningRow, decode_warning_row

from .test_warning_panel import row_document


def cohort(prefix: str, shift: int, breach_shift: int | None = None) -> tuple[WarningRow, ...]:
    """Build four borrowers with declared ratios and two future incidents."""
    result: list[WarningRow] = []
    for index in range(4):
        document = row_document(f"{prefix}{index}", shift)
        require_dict(document, "disclosure").update(
            debt_to_ebitda=float(index + 1),
            interest_cover=float(5 - index),
            current_ratio=float(4 - index),
        )
        require_dict(document, "followup")["first_breach_ms"] = (
            70 + (shift if breach_shift is None else breach_shift) if index >= 2 else None
        )
        result.append(decode_warning_row(document))
    return tuple(result)


def study() -> tuple[tuple[WarningFold, ...], WarningFold]:
    """Define two earlier folds and a later reserved calibration/holdout cohort."""
    first = WarningFold(
        fitting=cohort("fit", 0), calibration=cohort("cal", 200), validation=cohort("val", 400)
    )
    second = WarningFold(
        fitting=(*first["fitting"], *first["calibration"], *first["validation"]),
        calibration=cohort("cal2", 600),
        validation=cohort("val2", 800),
    )
    final = WarningFold(
        fitting=(*second["fitting"], *second["calibration"], *second["validation"]),
        calibration=cohort("reserved", 1000),
        validation=(*cohort("hold", 1200), *cohort("hold", 1230, 1200)),
    )
    return (first, second), final


__all__ = ["cohort", "study"]
