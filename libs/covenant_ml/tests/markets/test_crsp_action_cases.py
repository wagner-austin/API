"""Independent calendar joins, event boundaries and metadata-only action diagnostics."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)

from covenant_ml.markets.crsp_action_cases import crsp_action_cases
from covenant_ml.markets.crsp_action_schema import CRSP_ACTIONS
from covenant_ml.markets.crsp_actions import check_crsp_action_rows
from covenant_ml.markets.crsp_extract import CrspSources
from covenant_ml.markets.crsp_queries import crsp_scopes
from covenant_ml.markets.crsp_rows import encode_crsp_rows

from ._crsp_action_fixtures import action_rows
from ._crsp_extraction_fixtures import NOW, limits
from ._crsp_study_fixtures import study_sources


def _case(sources: CrspSources, day: str, period: str, offset: int) -> JSONObject:
    """Anchor one explicitly chosen synthetic date, without deriving the expected window."""
    (entry,) = [row for row in sources.returns if row["permno"] == 1 and row["dlycaldt"] == day]
    return {
        "path_ordinal": 7,
        "period": period,
        "entry": {"present": True, "row_sha256": sha256(dump_json_str(entry).encode()).hexdigest()},
        "source_window": {"exit_day_offset": offset},
    }


def test_join_preserves_event_only_and_exact_entry_exit_boundaries(tmp_path: Path) -> None:
    sources = study_sources(tmp_path)
    rows = action_rows()
    rows["delist-events"] = [
        {"permno": 1, "delistingdt": day}
        for day in ("2017-01-31", "2017-02-01", "2017-02-15", "2017-03-01", "2017-03-02")
    ]
    rows["delist-events"].append({"permno": 2, "delistingdt": "2017-02-01"})
    rows["delist-details"][0].update(
        delistingdt="2017-02-15",
        deldlydt="2017-02-16",
        delret=0.0,
        deldtprcflg="PRIVATE_FLAG_12345",
        delnextdt="2017-03-02",
    )
    rows["distribution-events"][0]["disexdt"] = "2017-02-01"
    rows["distributions"][0].update(disexdt="2017-02-01", dispaydt="2017-02-28", disfacshr=2.0)
    checked = check_crsp_action_rows(
        {item.role: encode_crsp_rows(rows[item.role], item.columns) for item in CRSP_ACTIONS},
        crsp_scopes(sources.protocol)[0],
        (1, 2),
        NOW,
        limits(),
    )
    (result,) = crsp_action_cases(sources, (_case(sources, "2017-02-01", "training", 28),), checked)
    assert result["exit_day_offset"] == 28
    assert result["path_ordinal"] == 7
    events = [
        narrow_json_to_dict(item)
        for item in require_list(require_dict(result, "actions"), "delist-events")
    ]
    assert [event["event_day_offset"] for event in events] == [-1, 0, 14, 28, 29]
    assert [event["inside_window"] for event in events] == [False, True, True, False, False]
    assert [event["on_entry"] for event in events] == [False, True, False, False, False]
    assert [event["on_exit"] for event in events] == [False, False, False, True, False]
    assert [event["before_entry"] for event in events] == [True, False, False, False, False]
    assert [event["after_exit"] for event in events] == [False, False, False, False, True]
    assert events[0]["detail"] is None
    detail = require_dict(events[2], "detail")
    assert detail["any_date_after_exit"] is True
    assert require_dict(detail, "day_offsets") == {
        "delistingdt": 14,
        "delnextdt": 29,
        "delamtdt": None,
        "deldlydt": 15,
    }
    assert require_dict(detail, "zero") == {"delret": True, "deldivamt": True}
    distribution = narrow_json_to_dict(
        require_list(require_dict(result, "actions"), "distribution-events")[0]
    )
    summary = require_dict(distribution, "detail")
    assert summary["any_date_after_exit"] is False
    assert require_dict(summary, "zero") == {
        "disdivamt": True,
        "disfacpr": False,
        "disfacshr": False,
    }
    assert require_dict(summary, "present")["dispermno"] is True
    assert require_dict(summary, "present")["disrecorddt"] is False
    assert require_dict(detail, "present")["delret"] is True
    for secret in ("PRIVATE_FLAG_12345", "2017-", "2019-", '"permno"', '"delret":0.0'):
        assert secret not in dump_json_str(result)


def test_validation_case_retains_null_values_and_empty_action_roles(tmp_path: Path) -> None:
    sources = study_sources(tmp_path)
    rows = action_rows()
    rows["delist-details"][0]["delret"] = None
    rows["distribution-events"] = []
    rows["distributions"] = []
    checked = check_crsp_action_rows(
        {item.role: encode_crsp_rows(rows[item.role], item.columns) for item in CRSP_ACTIONS},
        crsp_scopes(sources.protocol)[0],
        (1,),
        NOW,
        limits(),
    )
    (result,) = crsp_action_cases(
        sources, (_case(sources, "2019-06-03", "validation", 28),), checked
    )
    assert result["period"] == "validation"
    actions = require_dict(result, "actions")
    assert actions["distribution-events"] == []
    detail = require_dict(narrow_json_to_dict(require_list(actions, "delist-events")[0]), "detail")
    assert require_dict(detail, "zero")["delret"] is None
    assert require_dict(detail, "present")["delret"] is False


@pytest.mark.parametrize(
    ("day", "period", "offset"),
    [
        ("2017-02-02", "training", 28),
        ("2019-12-02", "validation", 31),
        ("2017-02-01", "validation", 28),
        ("2017-02-01", "training", 29),
    ],
)
def test_changed_window_and_final_crossing_refuse(
    tmp_path: Path, day: str, period: str, offset: int
) -> None:
    sources = study_sources(tmp_path)
    rows = action_rows()
    checked = check_crsp_action_rows(
        {item.role: encode_crsp_rows(rows[item.role], item.columns) for item in CRSP_ACTIONS},
        crsp_scopes(sources.protocol)[0],
        (1,),
        NOW,
        limits(),
    )
    with pytest.raises(ValueError, match="CRSP_ACTION_WINDOW"):
        crsp_action_cases(sources, (_case(sources, day, period, offset),), checked)


def test_reader_requires_exact_role_inventory(tmp_path: Path) -> None:
    sources = study_sources(tmp_path)
    with pytest.raises(ValueError, match="CRSP_ACTION_KEY: require every action projection"):
        check_crsp_action_rows({}, crsp_scopes(sources.protocol)[0], (1,), NOW, limits())
