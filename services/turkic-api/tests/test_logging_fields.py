from __future__ import annotations

import logging

import pytest
from platform_core.json_utils import load_json_str
from platform_core.logging import JsonFormatter

from turkic_api.api.logging_fields import LOG_EXTRA_FIELDS


def test_log_extra_fields_cover_structured_events() -> None:
    expected = {
        "job_id",
        "language",
        "url",
        "status",
        "file_id",
        "redis_url",
        "queue_name",
        "events_channel",
        "has_url",
        "has_key",
    }
    assert set(LOG_EXTRA_FIELDS) == expected


def test_log_extra_fields_are_emitted_with_json_formatter() -> None:
    formatter = JsonFormatter(static_fields={}, extra_field_names=LOG_EXTRA_FIELDS)
    record = logging.LogRecord(
        name="turkic.test",
        level=logging.INFO,
        pathname="t.py",
        lineno=1,
        msg="upload",
        args=(),
        exc_info=None,
    )
    record.job_id = "j-1"
    record.language = "kk"
    record.url = "http://example/files"
    record.status = 201
    record.file_id = "f-1"
    record.redis_url = "redis://localhost:6379/0"
    record.queue_name = "turkic"
    record.events_channel = "turkic.events"
    record.has_url = True
    record.has_key = False

    parsed_value = load_json_str(formatter.format(record))
    if type(parsed_value) is not dict:
        pytest.fail("expected dict")
    parsed = parsed_value
    assert parsed["job_id"] == "j-1"
    assert parsed["language"] == "kk"
    assert parsed["url"] == "http://example/files"
    assert parsed["status"] == 201
    assert parsed["file_id"] == "f-1"
    assert parsed["redis_url"] == "redis://localhost:6379/0"
    assert parsed["queue_name"] == "turkic"
    assert parsed["events_channel"] == "turkic.events"
    assert parsed["has_url"] is True
    assert parsed["has_key"] is False
