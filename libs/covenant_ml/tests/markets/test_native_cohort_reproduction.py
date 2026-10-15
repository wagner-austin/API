"""Reconstruct the published measurements from pinned native exchange responses.

This executes the actual archived cohort and both quote captures on every check.
No network, credentials, workstation paths or self-reported replay flags are used.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    narrow_json_to_dict,
    narrow_json_to_str,
    require_int,
    require_list,
    require_str,
)

from covenant_ml.markets.event_cohort import collect_event_cohort
from covenant_ml.markets.interval_quotes import collect_interval_quotes
from covenant_ml.markets.runner import AcquisitionMode, build_collector
from covenant_ml.markets.validation import read_document

BUNDLE_SHA = "f6eeba269313689cc4bf5ea5cb7f8e0147f9ff30f43b286da6f69ab841dbe622"
COHORT_SHA = "b1d92dbebaef11f2ab95dc3ba75281eab1c7ab5f36aff2de2c4638e83b6c7cd8"
QUOTE_SHA = "de068968848b800a391d5e7670cd65c95c611511aab2751ceb4ba9273132d87f"
WEEK_SHA = "200044456a7b9b8b4a1ce52e8af4eeeb83657eed04dafc24d8dcb7b60a5eb358"
ORIGIN = "https://external-api.kalshi.com/trade-api/v2"


def require_native_match(actual: JSONObject, expected: JSONObject) -> None:
    """Compare every native field, excluding only separately recorded run metadata.

    Args:
        actual: Reconstructed production result before persistence metadata.
        expected: Hash-pinned original published result.

    Raises:
        AssertionError: Any native value, field, identity or count changed.
    """
    metadata = {"sources", "fingerprint", "implementation_sha256", "cohort_sha256"}
    assert actual == {key: value for key, value in expected.items() if key not in metadata}


def test_published_native_cohort_and_quotes_reproduce(tmp_path: Path) -> None:
    bundle = Path(__file__).parent / "fixtures" / "kalshi-native-20261015.zip"
    assert sha256(bundle.read_bytes()).hexdigest() == BUNDLE_SHA
    with ZipFile(bundle) as evidence:
        assert len(evidence.namelist()) == 637
        assert evidence.testzip() is None
        evidence.extractall(tmp_path)
    root = tmp_path / "event-cohort"
    original = root / "results.json"
    assert sha256(original.read_bytes()).hexdigest() == COHORT_SHA
    assert (root / "replay.json").read_bytes() == original.read_bytes()
    plan = read_document(root / "plan.json")
    series = tuple(narrow_json_to_str(value) for value in require_list(plan, "series"))
    start, end = require_int(plan, "start_ts"), require_int(plan, "end_ts")
    reader, client = build_collector(root / "raw", ORIGIN, AcquisitionMode.REPLAY)
    with closing(client):
        cohort = collect_event_cohort(reader, series, start, end)
    require_native_match(cohort, read_document(original))
    families = [narrow_json_to_dict(value) for value in require_list(cohort, "families")]
    assert sum(require_int(row, "contract_count") for row in families) == 21982
    assert sum(require_int(row, "event_count") for row in families) == 21039
    assert sum(require_int(row, "zero_volume_contract_count") for row in families) == 62
    nominal = set(range(start, end, 900))
    crypto = [row for row in families if require_str(row, "series") in ("KXBTC15M", "KXETH15M")]
    assert len(crypto) == 2
    for family in crypto:
        markets = [narrow_json_to_dict(value) for value in require_list(family, "markets")]
        closes = {
            int(datetime.fromisoformat(require_str(row, "close_time")).timestamp())
            for row in markets
        }
        assert (len(nominal), len(closes), len(nominal - closes)) == (10176, 10048, 128)
        assert closes <= nominal
        assert sum(Decimal(require_str(row, "volume_fp")) == 0 for row in markets) == 6
    panels: list[JSONObject] = []
    for directory, digest, size in (
        ("interval-quotes", QUOTE_SHA, 11390),
        ("prior-eight-day", WEEK_SHA, 1520),
    ):
        quote_root = tmp_path / directory
        expected_path = quote_root / "results.json"
        assert sha256(expected_path.read_bytes()).hexdigest() == digest
        quote_plan = read_document(quote_root / "plan.json")
        collector, client = build_collector(quote_root / "raw", ORIGIN, AcquisitionMode.REPLAY)
        with closing(client):
            panel = collect_interval_quotes(collector, cohort, quote_plan)
        require_native_match(panel, read_document(expected_path))
        assert panel["contract_count"] == size
        assert panel["status_counts"] == {"two_sided": size}
        rows = [narrow_json_to_dict(value) for value in require_list(panel, "contracts")]
        assert {require_int(row, "quote_age_ms") for row in rows} == {60000}
        observed = {require_int(row, "open_ms") // 1000 + 900 for row in rows}
        expected_slots = set(
            range(require_int(quote_plan, "start_ts"), require_int(quote_plan, "end_ts"), 900)
        )
        assert observed <= expected_slots
        assert len(expected_slots - observed) == (65 if size == 11390 else 8)
        panels.append(panel)
    large = {
        require_str(row, "ticker"): row
        for value in require_list(panels[0], "contracts")
        for row in (narrow_json_to_dict(value),)
    }
    for value in require_list(panels[1], "contracts"):
        row = narrow_json_to_dict(value)
        assert large[require_str(row, "ticker")] == row
    # Real corrupted evidence must fail, even with a valid published report beside it.
    envelope_path = sorted((root / "raw").glob("*.json"))[0]
    envelope = read_document(envelope_path)
    envelope["body"] = "{}"
    envelope_path.write_text(dump_json_str(envelope), encoding="utf-8")
    reader, client = build_collector(root / "raw", ORIGIN, AcquisitionMode.REPLAY)
    with closing(client), pytest.raises(ValueError, match="MARKET-015"):
        collect_event_cohort(reader, series, start, end)
    envelope_path.unlink()
    reader, client = build_collector(root / "raw", ORIGIN, AcquisitionMode.REPLAY)
    with closing(client), pytest.raises(FileNotFoundError):
        collect_event_cohort(reader, series, start, end)
