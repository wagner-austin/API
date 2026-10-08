"""The operator asks (MCPs board task 1acbf53e): stated when they change, never otherwise.

The statement file is REAL bytes under ``tmp_path``, as fleet-mcp's
``encodeFleetAskStatement`` writes them (``JSON.stringify`` indented by two,
newline-terminated), so a decoder that drifted from the writer fails here.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest
from board_watch.config import load_credentials
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str, load_json_str
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments, tool_text_body

from fleet_health_wake import _test_hooks
from fleet_health_wake.asks import (
    OPERATOR_ASKS_FILENAME,
    decode_statement,
    state_asks,
    stated_path,
)
from fleet_health_wake.identity import BRIDGE_AGENT, HARNESS, IDENTITY
from tests.conftest import CONFIGURED_ENV, no_statement, pin_env

#: The writer's counts, as ``task_ask_sync`` answers them.
COUNTS: Final = '{"opened":1,"changed":0,"kept":0,"resolved":0}'

#: sedona's ask, as fleet-mcp's ``diskAsk`` words it.
SEDONA_SPEC: Final[JSONObject] = {
    "kind": "decision",
    "key": "disk-floor:sedona",
    "subjectTaskId": None,
    "subjectNode": "sedona",
    "urgent": False,
    "cause": None,
    "room": "coordination",
    "title": None,
    "what": "sedona's free disk is below its floor of 20GB",
    "why": (
        "a full disk stops every build, container and backup on it, and what to delete is a "
        "ruling the room's supervisor or a session makes from the survey"
    ),
    "action": (
        "see what fills sedona's disk (fleet_status, the fleet disk survey), decide what is "
        "deleted or moved, and free it"
    ),
}


def statement_bytes(at: str, *, full: bool) -> bytes:
    """A statement file as the audit writes it.

    Args:
        at: The run's capture time.
        full: Whether sedona is below its floor.

    Returns:
        The file's bytes, indented by two and newline-terminated.
    """
    asks = dump_json_str([SEDONA_SPEC] if full else [], indent=2).replace("\n", "\n  ")
    text = f'{{\n  "source": "fleet:disk",\n  "at": "{at}",\n  "asks": {asks}\n}}\n'
    return text.encode("utf-8")


def stage_statement(tmp_path: pathlib.Path, content: bytes) -> pathlib.Path:
    """Write the statement file with exact bytes.

    Args:
        tmp_path: The test's temporary directory.
        content: The file's bytes.

    Returns:
        Its path.
    """
    path = tmp_path / OPERATOR_ASKS_FILENAME
    path.write_bytes(content)
    return path


def stating_poster() -> FakeHttpPost:
    """A poster scripted for one statement: the checkin, then the writer's counts.

    Returns:
        The poster.
    """
    counts = McpHttpResponse(
        status=200, body=tool_text_body(COUNTS), content_type=EVENT_STREAM_MEDIA_TYPE
    )
    return FakeHttpPost([posted_ok(), counts])


def stated_once(tmp_path: pathlib.Path) -> pathlib.Path:
    """Stage sedona below its floor and state it, as a first cycle would.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The statement file's path.
    """
    path = stage_statement(tmp_path, statement_bytes("2026-10-05T11:00:00.000Z", full=True))
    _test_hooks.http_post = stating_poster()
    state_asks(path, load_credentials())
    return path


class TestStateAsks:
    def test_states_a_new_statement_after_the_checkin_and_records_it(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        path = stage_statement(tmp_path, statement_bytes("2026-10-05T11:00:00.000Z", full=True))
        poster = stating_poster()
        _test_hooks.http_post = poster

        state_asks(path, load_credentials())

        checkin, statement = (sent_arguments(body) for body in poster.bodies)
        assert (checkin["kind"], checkin["harness"]) == ("checkin", HARNESS)
        assert statement == {
            "agent": BRIDGE_AGENT,
            "sessionId": IDENTITY["session_id"],
            "cwd": IDENTITY["cwd"],
            "source": "fleet:disk",
            "asks": [SEDONA_SPEC],
        }
        recorded = load_json_str(stated_path(path).read_text(encoding="utf-8"))
        assert recorded == {"source": "fleet:disk", "asks": [SEDONA_SPEC]}
        assert emitted == [f"operator asks: stated 1 under fleet:disk: {COUNTS}"]

    def test_a_later_run_with_the_same_asks_states_nothing(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        path = stated_once(tmp_path)
        stage_statement(tmp_path, statement_bytes("2026-10-05T11:20:00.000Z", full=True))
        quiet = FakeHttpPost([])
        _test_hooks.http_post = quiet

        state_asks(path, load_credentials())

        assert quiet.bodies == []
        assert emitted[-1] == "operator asks: 1 open under fleet:disk, unchanged; nothing stated"

    def test_a_disk_that_recovered_is_stated_as_no_ask(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        path = stated_once(tmp_path)
        stage_statement(tmp_path, statement_bytes("2026-10-05T11:20:00.000Z", full=False))
        poster = stating_poster()
        _test_hooks.http_post = poster

        state_asks(path, load_credentials())

        assert sent_arguments(poster.bodies[1])["asks"] == []
        assert emitted[-1] == f"operator asks: stated 0 under fleet:disk: {COUNTS}"

    def test_a_refused_statement_leaves_the_marker_for_the_next_cycle(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        path = stage_statement(tmp_path, statement_bytes("2026-10-05T11:00:00.000Z", full=True))
        refused = McpHttpResponse(status=500, body="board down", content_type="text/plain")
        _test_hooks.http_post = FakeHttpPost([posted_ok(), refused])

        with pytest.raises(AppError):
            state_asks(path, load_credentials())

        assert not stated_path(path).exists()
        assert emitted == []

    def test_no_statement_yet_is_said(self, tmp_path: pathlib.Path, emitted: list[str]) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([])

        state_asks(tmp_path / OPERATOR_ASKS_FILENAME, load_credentials())

        assert emitted == [no_statement(tmp_path)]


class TestDecodeStatement:
    def test_reads_the_audit_s_bytes(self) -> None:
        raw = statement_bytes("2026-10-05T11:00:00.000Z", full=True).decode("utf-8")
        statement = decode_statement(load_json_str(raw))
        assert statement["at"] == "2026-10-05T11:00:00.000Z"
        assert [(ask["key"], ask["subjectTaskId"]) for ask in statement["asks"]] == [
            ("disk-floor:sedona", None)
        ]
        (ask,) = statement["asks"]
        assert (ask["kind"], ask["urgent"], ask["cause"], ask["room"], ask["title"]) == (
            "decision",
            False,
            None,
            "coordination",
            None,
        )

    def test_refuses_an_ask_written_before_it_declared_a_title(self) -> None:
        """An ask with no ``title`` is the pre-e6514287 audit's: refused, never read as null."""
        ask: JSONObject = {**SEDONA_SPEC}
        del ask["title"]
        value = load_json_str(dump_json_str({"source": "s", "at": "a", "asks": [ask]}))
        with pytest.raises(JSONTypeError, match="'title' is absent"):
            decode_statement(value)

    def test_refuses_an_ask_written_before_its_route_was_stated(self) -> None:
        """An ask with no ``urgent`` is the pre-db98562b audit's: refused, never read as false."""
        ask: JSONObject = {**SEDONA_SPEC}
        del ask["urgent"]
        value = load_json_str(dump_json_str({"source": "s", "at": "a", "asks": [ask]}))
        with pytest.raises(JSONTypeError, match="urgent"):
            decode_statement(value)

    def test_refuses_an_ask_missing_a_nullable_field(self) -> None:
        ask: JSONObject = {"kind": "critical", "key": "k", "subjectTaskId": None}
        value = load_json_str(dump_json_str({"source": "s", "at": "a", "asks": [ask]}))
        with pytest.raises(JSONTypeError, match="'subjectNode' is absent"):
            decode_statement(value)

    def test_refuses_a_nullable_field_of_another_type(self) -> None:
        ask: JSONObject = {"kind": "critical", "key": "k", "subjectTaskId": 7}
        value = load_json_str(dump_json_str({"source": "s", "at": "a", "asks": [ask]}))
        with pytest.raises(JSONTypeError, match="'subjectTaskId' must be a string or null, got"):
            decode_statement(value)
