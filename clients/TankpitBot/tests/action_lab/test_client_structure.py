"""End-to-end tests for the one-shot client structure survey.

Tests drive the REAL pipeline: fake CDP value -> real capture/validation
-> real ``emit_diagnostic`` -> JSONL via
:class:`tests.conftest.FakeFileSystem`. Nothing is mocked.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict
from tests.conftest import FakeFileSystem

from tankpit_bot.browser.client_structure import ClientStructureSurveyor
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.runtime_logging import configure_bot_runtime_logging
from tankpit_bot.runtime_records import require_str_field

_SURVEY_VALUE: JSONObject = {
    "kind": "object",
    "key_count": 2,
    "keys": ["h", "i"],
    "children": {
        "h": {"kind": "object", "key_count": 1, "keys": ["tanks"], "children": {}},
        "i": {"kind": "object", "key_count": 1, "keys": ["id"], "children": {}},
    },
}


class _SurveyCDPSession:
    """Fake CDP session serving a structure survey for the walk expression."""

    def __init__(self, value: JSONObject | None) -> None:
        """Store the survey value to serve (``None`` = client absent)."""
        self._value = value
        self.evaluate_count = 0

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Serve the survey value for the structure-walk expression.

        Args:
            method: CDP method name.
            params: CDP call parameters.

        Returns:
            CDP-style result object.
        """
        if method == "Runtime.evaluate" and params is not None:
            expression = str(params.get("expression", ""))
            if "MAX_DEPTH" in expression:
                self.evaluate_count += 1
                value: JSONObject | None = self._value
                return {"result": {"value": value}}
        return {"result": {"value": ""}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Unused protocol member."""

    def detach(self) -> None:
        """Unused protocol member."""


def test_survey_emits_once_and_lands_in_artifact(fake_fs: FakeFileSystem) -> None:
    """The first healthy capture emits the survey; later calls are gated off."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    cdp = _SurveyCDPSession(_SURVEY_VALUE)

    surveyor = ClientStructureSurveyor()

    first = surveyor.maybe_emit(cdp)
    second = surveyor.maybe_emit(cdp)

    assert first is True
    assert second is False
    assert cdp.evaluate_count == 1
    records = [
        record
        for record in load_event_records(Path(artifacts["latest_events_path"]))
        if record["fields"].get("diagnostic_kind") == "client_structure_survey"
    ]
    assert len(records) == 1
    survey = narrow_json_to_dict(
        load_json_str(require_str_field(records[0]["fields"], "survey_json"))
    )
    assert survey == _SURVEY_VALUE


def test_survey_retries_while_client_object_is_absent(fake_fs: FakeFileSystem) -> None:
    """A null client object keeps the gate open for the next tick."""
    configure_bot_runtime_logging("20260610-120000")
    absent = _SurveyCDPSession(None)
    present = _SurveyCDPSession(_SURVEY_VALUE)

    surveyor = ClientStructureSurveyor()

    assert surveyor.maybe_emit(absent) is False
    assert surveyor.maybe_emit(present) is True


def test_each_surveyor_owes_its_own_survey(fake_fs: FakeFileSystem) -> None:
    """A second session surveys again -- the gate is instance state.

    This is the property that replaced ``reset_client_structure_survey``:
    "once per session" must mean once per SESSION, not once per process
    ([[session-state-deglobalisation]] step 5).
    """
    configure_bot_runtime_logging("20260610-120000")
    cdp = _SurveyCDPSession(_SURVEY_VALUE)
    assert ClientStructureSurveyor().maybe_emit(cdp) is True

    assert ClientStructureSurveyor().maybe_emit(cdp) is True
    assert cdp.evaluate_count == 2
