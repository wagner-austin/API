"""The journal reader: the cursor contract, the strict decode, and history.

The cases that matter are the ones the journal's writer makes real: a torn
tail mid-append, rows from before the ``agent`` field existed, and a file
that has not been created yet because no cascade has ever run.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError

from lock_wake.journal import EVENT_KINDS, decode_lock_event, read_journal_slice
from tests.conftest import journal_line, stage_journal


class TestReadJournalSlice:
    def test_reads_every_complete_line_and_reports_the_next_offset(
        self, tmp_path: pathlib.Path
    ) -> None:
        content = (journal_line(kind="requested") + journal_line(kind="acquired")).encode("utf-8")
        journal = stage_journal(tmp_path, content)

        result = read_journal_slice(journal, 0)

        assert [e["kind"] for e in result["events"]] == ["requested", "acquired"]
        assert result["next_offset"] == len(content)

    def test_starts_at_the_given_offset_rather_than_the_top(self, tmp_path: pathlib.Path) -> None:
        first = journal_line(kind="requested").encode("utf-8")
        second = journal_line(kind="released").encode("utf-8")
        journal = stage_journal(tmp_path, first + second)

        result = read_journal_slice(journal, len(first))

        assert [e["kind"] for e in result["events"]] == ["released"]
        assert result["next_offset"] == len(first) + len(second)

    def test_leaves_a_torn_tail_for_the_next_cycle(self, tmp_path: pathlib.Path) -> None:
        # The writer is mid-append: the bytes after the last newline are a
        # line that does not exist yet, not an error.
        complete = journal_line(kind="acquired").encode("utf-8")
        torn = b'{"ts":"2026-09-09T19:30:45.46'
        journal = stage_journal(tmp_path, complete + torn)

        result = read_journal_slice(journal, 0)

        assert [e["kind"] for e in result["events"]] == ["acquired"]
        assert result["next_offset"] == len(complete)

    def test_a_wholly_torn_window_yields_nothing_and_holds_position(
        self, tmp_path: pathlib.Path
    ) -> None:
        complete = journal_line(kind="acquired").encode("utf-8")
        journal = stage_journal(tmp_path, complete + b'{"ts":"2026')

        result = read_journal_slice(journal, len(complete))

        assert result["events"] == ()
        assert result["next_offset"] == len(complete)

    def test_an_absent_journal_reads_empty_at_offset_zero(self, tmp_path: pathlib.Path) -> None:
        result = read_journal_slice(tmp_path / "never-written.jsonl", 0)
        assert result == {"events": (), "next_offset": 0}

    def test_an_absent_journal_with_a_position_refuses(self, tmp_path: pathlib.Path) -> None:
        # A journal that vanished under a recorded position was deleted or
        # moved; rewinding silently would re-announce all of history.
        with pytest.raises(ValueError, match="absent"):
            read_journal_slice(tmp_path / "never-written.jsonl", 100)

    def test_a_position_past_the_end_refuses(self, tmp_path: pathlib.Path) -> None:
        journal = stage_journal(tmp_path, journal_line().encode("utf-8"))
        with pytest.raises(ValueError, match="truncated or replaced"):
            read_journal_slice(journal, 10_000)

    def test_a_complete_line_that_is_not_an_object_is_fatal(self, tmp_path: pathlib.Path) -> None:
        journal = stage_journal(tmp_path, b"[1, 2, 3]\n")
        with pytest.raises(JSONTypeError, match="not an object"):
            read_journal_slice(journal, 0)

    def test_blank_lines_are_not_transitions(self, tmp_path: pathlib.Path) -> None:
        content = (journal_line(kind="acquired") + "\n" + journal_line(kind="released")).encode(
            "utf-8"
        )
        journal = stage_journal(tmp_path, content)

        result = read_journal_slice(journal, 0)

        assert [e["kind"] for e in result["events"]] == ["acquired", "released"]
        assert result["next_offset"] == len(content)


class TestDecodeLockEvent:
    def test_reads_the_wrapper_form_field_for_field(self) -> None:
        event = decode_lock_event(
            {
                "ts": "2026-09-09T19:28:00.0608673Z",
                "kind": "step",
                "pid": 2688,
                "label": "up-transcriber",
                "op": "service-up",
                "only": "all",
                "detail": "compose up -d",
                "agent": "opus-mosh-reboot-0909",
            },
            1,
        )
        assert event == {
            "ts": "2026-09-09T19:28:00.0608673Z",
            "kind": "step",
            "holder_pid": 2688,
            "label": "up-transcriber",
            "op": "service-up",
            "only": "all",
            "detail": "compose up -d",
            "agent": "opus-mosh-reboot-0909",
        }

    def test_a_history_row_without_the_agent_key_reads_as_unlabelled(self) -> None:
        # Rows written before MCPs 66b85d32 are immutable facts without the
        # field; reading them as "" is the hpc3 ledger's own pre-field rule.
        event = decode_lock_event(
            {
                "ts": "2026-09-09T18:55:42.3761925Z",
                "kind": "released",
                "pid": 24940,
                "label": "up-cloudflared",
                "op": "cloudflared-roll",
                "only": "all",
                "detail": "",
            },
            1,
        )
        assert event["agent"] == ""

    def test_a_mistyped_agent_is_fatal_not_defaulted(self) -> None:
        with pytest.raises(JSONTypeError, match="'agent'"):
            decode_lock_event(
                {
                    "ts": "t",
                    "kind": "released",
                    "pid": 1,
                    "label": "l",
                    "op": "o",
                    "only": "all",
                    "detail": "",
                    "agent": 7,
                },
                3,
            )

    def test_an_unknown_kind_is_fatal(self) -> None:
        with pytest.raises(JSONTypeError, match="does not know"):
            decode_lock_event(
                {
                    "ts": "t",
                    "kind": "paused",
                    "pid": 1,
                    "label": "l",
                    "op": "o",
                    "only": "all",
                    "detail": "",
                    "agent": "",
                },
                2,
            )

    def test_every_declared_kind_narrows(self) -> None:
        for kind in EVENT_KINDS:
            event = decode_lock_event(
                {
                    "ts": "t",
                    "kind": kind,
                    "pid": 1,
                    "label": "l",
                    "op": "o",
                    "only": "all",
                    "detail": "",
                    "agent": "",
                },
                1,
            )
            assert event["kind"] == kind

    def test_a_non_object_is_fatal(self) -> None:
        with pytest.raises(JSONTypeError, match="not an object"):
            decode_lock_event("released", 5)
