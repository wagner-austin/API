"""The health journal reader: the audit's own bytes, decoded strictly."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import InvalidJsonError, JSONTypeError

from fleet_health_wake.journal import EVENT_KINDS, decode_health_event, read_health_slice
from tests.conftest import (
    REFUSED_BODY,
    REFUSED_LINE,
    TRANSITIONS_BODY,
    TRANSITIONS_LINE,
    stage_journal,
)


class TestReadHealthSlice:
    def test_decodes_the_audits_own_lines_and_reports_the_next_offset(
        self, tmp_path: pathlib.Path
    ) -> None:
        content = (TRANSITIONS_LINE + REFUSED_LINE).encode("utf-8")
        result = read_health_slice(stage_journal(tmp_path, content), 0)
        assert result == {
            "events": (
                {
                    "at": "2026-09-26T06:14:13.504Z",
                    "kind": "transitions",
                    "key": "2026-09-26T06:14:13.504Z",
                    "body": TRANSITIONS_BODY,
                },
                {
                    "at": "2026-09-26T06:20:00.000Z",
                    "kind": "refused",
                    "key": "hash-mismatch:" + "a" * 64,
                    "body": REFUSED_BODY,
                },
            ),
            "next_offset": len(content),
        }

    def test_an_absent_journal_reads_empty(self, tmp_path: pathlib.Path) -> None:
        assert read_health_slice(tmp_path / "health-events.jsonl", 0) == {
            "events": (),
            "next_offset": 0,
        }

    def test_a_complete_line_that_is_not_json_is_fatal(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(InvalidJsonError):
            read_health_slice(stage_journal(tmp_path, b"{not json\n"), 0)


class TestDecodeHealthEvent:
    def test_every_declared_kind_narrows(self) -> None:
        for kind in EVENT_KINDS:
            event = decode_health_event({"at": "a", "kind": kind, "key": "k", "body": "b"}, 1)
            assert event["kind"] == kind

    def test_an_unknown_kind_is_fatal_and_names_its_line(self) -> None:
        with pytest.raises(JSONTypeError, match="line 3 has kind 'alert'"):
            decode_health_event({"at": "a", "kind": "alert", "key": "k", "body": "b"}, 3)

    def test_a_non_object_is_fatal(self) -> None:
        with pytest.raises(JSONTypeError, match="line 2 is a list, not an object"):
            decode_health_event([1], 2)

    def test_a_missing_body_is_fatal(self) -> None:
        with pytest.raises(JSONTypeError, match="body"):
            decode_health_event({"at": "a", "kind": "baseline", "key": "k"}, 1)
