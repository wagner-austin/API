"""Atomic multi-order reconciliation uses the actual SQLite transaction and reducer."""

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, require_dict

from covenant_ml.markets.execution_journal import (
    append_execution_event,
    initialize_execution_journal,
    read_execution_journal,
)
from covenant_ml.markets.execution_records import decode_receipt, encode_intent

from ._execution_fixtures import intent
from .test_execution_policy import receipt_event


@pytest.mark.parametrize("failure", ["empty", "action", "time", "identity"])
def test_account_snapshot_failure_is_atomic(tmp_path: Path, failure: str) -> None:
    path = tmp_path / "journal.sqlite"
    initialize_execution_journal(path)
    row = intent()
    before = append_execution_event(
        path, {"action": "reserve", "observed_ms": 2000, "data": encode_intent(row)}
    )
    second = receipt_event(row)
    snapshot: JSONObject = {
        "action": "account_snapshot",
        "observed_ms": 2000,
        "events": [receipt_event(row), second],
    }
    if failure == "empty":
        snapshot["events"] = []
    elif failure == "action":
        second["action"] = "halt"
    elif failure == "time":
        second["observed_ms"] = 2001
    else:
        second["client_order_id"] = "unknown"
    with pytest.raises(ValueError, match=r"EXECUTION-006|EXECUTION-012"):
        append_execution_event(path, snapshot)
    assert read_execution_journal(path) == before
    accepted = append_execution_event(
        path,
        {"action": "account_snapshot", "observed_ms": 2000, "events": [receipt_event(row)]},
    )
    assert read_execution_journal(path) == accepted
    assert accepted["orders"][0]["receipt"] == decode_receipt(
        require_dict(receipt_event(row), "data")
    )
