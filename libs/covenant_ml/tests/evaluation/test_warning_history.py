"""Contradictory borrower histories must fail before chronological learning."""

from __future__ import annotations

import pytest
from platform_core.json_utils import require_dict

from covenant_ml.evaluation.warning_folds import WarningFold
from covenant_ml.evaluation.warning_history import require_warning_history
from covenant_ml.evaluation.warning_panel import decode_warning_row, encode_warning_row
from covenant_ml.evaluation.warning_study import run_warning_study

from .test_warning_study import candidates
from .warning_testing import cohort, study


@pytest.mark.parametrize("stage", ["fitting", "calibration", "validation"])
@pytest.mark.parametrize("contradiction", ["different_breach", "event_free"])
def test_contradictory_history_rejected_before_learning(stage: str, contradiction: str) -> None:
    folds, final = study()
    first = folds[0]
    partitions = {"fitting": ("fit", 0), "calibration": ("cal", 200), "validation": ("val", 400)}
    prefix, shift = partitions[stage]
    repeated = list(cohort(prefix, shift + 30))
    if contradiction == "event_free":
        document = encode_warning_row(repeated[2])
        require_dict(document, "followup")["first_breach_ms"] = None
        repeated[2] = decode_warning_row(document)
        document = encode_warning_row(repeated[3])
        require_dict(document, "followup")["first_breach_ms"] = 70 + shift
        repeated[3] = decode_warning_row(document)
    changed = WarningFold(
        fitting=(*first["fitting"], *repeated) if stage == "fitting" else first["fitting"],
        calibration=(
            (*first["calibration"], *repeated) if stage == "calibration" else first["calibration"]
        ),
        validation=(
            (*first["validation"], *repeated) if stage == "validation" else first["validation"]
        ),
    )
    with pytest.raises(ValueError, match="WARN-024"):
        run_warning_study(candidates(), (changed, folds[1]), final, 1340)


def test_conflicting_evidence_between_individually_valid_folds_is_rejected() -> None:
    folds, final = study()
    changed_rows = []
    for row in folds[1]["fitting"]:
        document = encode_warning_row(row)
        followup = require_dict(document, "followup")
        if row["disclosure"]["borrower"] == "fit2":
            followup["first_breach_ms"] = 75
        changed_rows.append(decode_warning_row(document))
    changed = WarningFold(
        fitting=tuple(changed_rows),
        calibration=folds[1]["calibration"],
        validation=folds[1]["validation"],
    )
    with pytest.raises(ValueError, match="WARN-024"):
        run_warning_study(candidates(), (folds[0], changed), final, 1340)


def test_earlier_event_free_surveillance_can_precede_a_later_breach() -> None:
    earlier = encode_warning_row(cohort("borrower", 0)[2])
    require_dict(earlier, "followup")["first_breach_ms"] = None
    later = encode_warning_row(cohort("borrower", 30)[2])
    require_dict(later, "followup")["first_breach_ms"] = 120
    rows = (decode_warning_row(earlier), decode_warning_row(later))
    assert require_warning_history(rows) == {"borrower2": 120}
    assert require_warning_history((*reversed(rows), *rows)) == {"borrower2": 120}
