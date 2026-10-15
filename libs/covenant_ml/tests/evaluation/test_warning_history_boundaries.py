"""Independent interval boundary expectations for the first-incident contract."""

from __future__ import annotations

import pytest
from platform_core.json_utils import require_dict

from covenant_ml.evaluation.warning_alerts import evaluate_warning_alerts
from covenant_ml.evaluation.warning_history import require_warning_history
from covenant_ml.evaluation.warning_panel import decode_warning_row

from .test_warning_panel import row_document


@pytest.mark.parametrize("breach", [70, 71])
@pytest.mark.parametrize("through", [69, 70, 71, 72])
@pytest.mark.parametrize("reverse", [False, True])
def test_event_free_surveillance_boundary(breach: int, through: int, reverse: bool) -> None:
    positive = row_document("A")
    require_dict(positive, "followup")["first_breach_ms"] = breach
    negative = row_document("A")
    require_dict(negative, "followup").update(
        coverage_start_ms=10, first_breach_ms=None, observed_through_ms=through
    )
    rows = (decode_warning_row(positive), decode_warning_row(negative))
    ordered = tuple(reversed(rows)) if reverse else rows
    # The interval is inclusive at its end: a clean observation through the
    # event instant contradicts the event; ending one millisecond earlier does not.
    if through < breach:
        assert require_warning_history(ordered) == {"A": breach}
    else:
        with pytest.raises(ValueError, match=r"WARN-024.*event-free"):
            require_warning_history(ordered)


@pytest.mark.parametrize("later_breach", [None, 170])
@pytest.mark.parametrize("reverse", [False, True])
def test_reentry_is_unsupported_not_a_contradictory_coverage_interval(
    later_breach: int | None, reverse: bool
) -> None:
    earlier = decode_warning_row(row_document("A"))
    later = row_document("A", 100)
    require_dict(later, "followup")["first_breach_ms"] = later_breach
    rows = (earlier, decode_warning_row(later))
    ordered = tuple(reversed(rows)) if reverse else rows
    with pytest.raises(ValueError, match=r"WARN-030.*re-entry"):
        evaluate_warning_alerts(ordered, (0.9, 0.9), 1, 0, 210)
