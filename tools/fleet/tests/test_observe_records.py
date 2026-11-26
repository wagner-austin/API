"""Reading a node's observe document and copying each record into the wire shape.

Every refusal here is a VALUE (MCPs board task 56fee128): the decode returns
a :class:`fleet.core.observe_records.RecordRefusal` naming every fault, which
the node pass makes that node's outcome. Until 2026-11-25 each of these
raised, and one demure record whose ``messagingSocketPath`` was null ended
every hub tick from 01:09Z. The expected sentences are worked by hand from
the field shapes the harness writes (the fixture is the 2.1.270 record
measured on austinpc), not copied from what the code printed.
"""

from __future__ import annotations

from itertools import combinations

import pytest
from platform_core.errors import FleetErrorCode
from platform_core.json_utils import InvalidJsonError, JSONObject

from fleet.core import observe_records
from tests._observe_fixtures import MACHINE, SESSION, UPDATED_MS, document, harness_record

UNREADABLE = "colossus reported a session record this runner cannot read"

#: The fields a record must carry, in the order the decode reads them, each
#: with the kind it must be: the 2.1.270 harness record's own shape.
REQUIRED_FIELDS: tuple[tuple[str, str], ...] = (
    ("sessionId", "not a non-empty string"),
    ("pid", "not an integer"),
    ("startedAt", "not an integer"),
    ("name", "not a non-empty string"),
    ("nameSource", "not a non-empty string"),
    ("cwd", "not a non-empty string"),
    ("messagingSocketPath", "not a non-empty string"),
    ("version", "not a non-empty string"),
    ("status", "not a non-empty string"),
    ("updatedAt", "not an integer"),
)

#: The pidDomain Claude Code 2.1.294 wrote on ziggurat, measured 2026-10-16.
ZIGGURAT_PID_DOMAIN = "linux:f191c99155c14a719632bdc88e77f914:pid:[4026531836]"


def decoded(output: str) -> observe_records.NodeSessions:
    """Decode a document a case expects to read.

    Args:
        output: The script's output.

    Returns:
        The decoded document; a refusal fails the case, naming it.
    """
    sessions = observe_records.decode_node_sessions("colossus", output)
    assert not observe_records.is_refusal(sessions), sessions
    return sessions


def observed(
    pid_domain: str, record: JSONObject, host: str = "colossus"
) -> observe_records.SessionObservation:
    """Copy a record a case expects to read.

    Args:
        pid_domain: The pidDomain its node reports.
        record: The record.
        host: The node.

    Returns:
        The observation; a refusal fails the case, naming it.
    """
    observation = observe_records.to_observation(host, pid_domain, record)
    assert not observe_records.is_refusal(observation), observation
    return observation


def test_iso_from_millis_keeps_milliseconds_and_names_utc() -> None:
    assert observe_records.iso_from_millis(UPDATED_MS) == "2026-09-10T00:27:10.500+00:00"
    assert observe_records.iso_from_millis(0) == "1970-01-01T00:00:00.000+00:00"


def test_decode_node_sessions_reads_the_document() -> None:
    sessions = decoded(document(harness_record()))
    assert sessions["platform"] == "linux"
    assert sessions["hostname"] == "colossus"
    assert sessions["pid_domain"] == MACHINE
    assert len(sessions["records"]) == 1
    assert sessions["records"][0]["sessionId"] == SESSION
    assert observe_records.machine_of(sessions) == MACHINE


def test_decode_node_sessions_reads_an_empty_directory_as_zero_records() -> None:
    assert decoded(document())["records"] == ()


NO_STRING = "not a non-empty string"


@pytest.mark.parametrize(
    ("output", "detail"),
    [
        ("[]", "the document is list, not an object"),
        ('{"platform":"linux","hostname":"x","records":{}}', "'records' is dict, not an array"),
        (
            '{"platform":"linux","hostname":"x","pidDomain":"linux:x","records":[1]}',
            "a record is int, not an object",
        ),
        (
            '{"platform":"linux","records":[]}',
            f"'hostname' is NoneType, {NO_STRING}; 'pidDomain' is NoneType, {NO_STRING}",
        ),
        (
            '{"platform":"","hostname":"x","pidDomain":"linux:x","records":[]}',
            f"'platform' is str, {NO_STRING}",
        ),
        (
            '{"platform":"linux","hostname":"x","records":[]}',
            f"'pidDomain' is NoneType, {NO_STRING}",
        ),
        (
            '{"records":[null,2]}',
            "a record is NoneType, not an object; a record is int, not an object; "
            f"'platform' is NoneType, {NO_STRING}; 'hostname' is NoneType, {NO_STRING}; "
            f"'pidDomain' is NoneType, {NO_STRING}",
        ),
    ],
)
def test_decode_node_sessions_refuses_a_document_it_cannot_read(output: str, detail: str) -> None:
    """Every fault of the document is named in one refusal, returned, not raised."""
    assert observe_records.decode_node_sessions("colossus", output) == (
        observe_records.RecordRefusal(
            code=FleetErrorCode.SESSION_RECORD_UNREADABLE, message=f"{UNREADABLE}: {detail}"
        )
    )


def test_decode_node_sessions_raises_on_output_that_is_not_json() -> None:
    """The script prints one json.dumps document or exits nonzero; text that
    is not JSON at all is this runner's own script broken, and raises."""
    with pytest.raises(InvalidJsonError):
        observe_records.decode_node_sessions("colossus", "Welcome to colossus")


def test_to_observation_copies_every_field_and_converts_the_instants() -> None:
    observation = observed(MACHINE, harness_record())
    assert observation == observe_records.SessionObservation(
        sessionId=SESSION,
        pid=4242,
        processStartedAt="2026-09-10T00:26:40.000+00:00",
        name="api-7e",
        nameSource="derived",
        cwd="/home/corvis/PROJECTS/API",
        socketPath="/run/user/1000/claude-cc-msg-abc",
        harnessVersion="2.1.270",
        status="idle",
        heartbeatAt="2026-09-10T00:27:10.500+00:00",
    )
    assert observe_records.observation_json(observation) == {
        "sessionId": SESSION,
        "pid": 4242,
        "processStartedAt": "2026-09-10T00:26:40.000+00:00",
        "name": "api-7e",
        "nameSource": "derived",
        "cwd": "/home/corvis/PROJECTS/API",
        "socketPath": "/run/user/1000/claude-cc-msg-abc",
        "harnessVersion": "2.1.270",
        "status": "idle",
        "heartbeatAt": "2026-09-10T00:27:10.500+00:00",
    }


def test_to_observation_accepts_a_record_without_pid_domain() -> None:
    """2.1.236 writes no ``pidDomain``; the node it was read from is the machine."""
    assert observed(MACHINE, harness_record(without=("pidDomain",)))["sessionId"] == SESSION


def test_to_observation_refuses_a_record_from_another_machine() -> None:
    assert observe_records.to_observation(
        "colossus", MACHINE, harness_record(pidDomain="linux:lacrymal")
    ) == observe_records.RecordRefusal(
        code=FleetErrorCode.SESSION_MACHINE_MISMATCH,
        message=(
            f"colossus: record for session {SESSION} names pidDomain 'linux:lacrymal' "
            f"but its node reports 'linux:colossus'"
        ),
    )


def test_to_observation_accepts_the_2_1_294_spelling_its_node_reports() -> None:
    """Board task 5cee2f65: every hub tick from 06:30Z failed here, because
    the record was compared to ``linux:ziggurat``. The node now reports the
    spelling its harness writes, and the record is compared to that."""
    record = harness_record(pidDomain=ZIGGURAT_PID_DOMAIN, version="2.1.294")

    observation = observed(ZIGGURAT_PID_DOMAIN, record, host="ziggurat")

    assert observation["harnessVersion"] == "2.1.294"


def test_to_observation_refuses_a_record_from_another_machine_id() -> None:
    other = "linux:b016a0cfb2c942b0a2cc55b43e865711:pid:[4026531836]"
    assert observe_records.to_observation(
        "ziggurat", ZIGGURAT_PID_DOMAIN, harness_record(pidDomain=other)
    ) == observe_records.RecordRefusal(
        code=FleetErrorCode.SESSION_MACHINE_MISMATCH,
        message=(
            f"ziggurat: record for session {SESSION} names pidDomain {other!r} "
            f"but its node reports {ZIGGURAT_PID_DOMAIN!r}"
        ),
    )


@pytest.mark.parametrize(
    ("record", "detail"),
    [
        (harness_record(pidDomain=7), "'pidDomain' is int, not a string"),
        (harness_record(pid="4242"), "'pid' is str, not an integer"),
        (harness_record(pid=True), "'pid' is bool, not an integer"),
        (harness_record(without=("startedAt",)), "'startedAt' is NoneType, not an integer"),
        (harness_record(without=("name",)), f"'name' is NoneType, {NO_STRING}"),
        (harness_record(version=""), f"'version' is str, {NO_STRING}"),
        (harness_record(status=3), f"'status' is int, {NO_STRING}"),
        # The demure record of 2026-11-25T01:09Z (Claude Code 2.1.223).
        (
            harness_record(messagingSocketPath=None),
            f"'messagingSocketPath' is NoneType, {NO_STRING}",
        ),
    ],
)
def test_to_observation_refuses_a_record_it_cannot_read(record: JSONObject, detail: str) -> None:
    assert observe_records.to_observation("colossus", MACHINE, record) == (
        observe_records.RecordRefusal(
            code=FleetErrorCode.SESSION_RECORD_UNREADABLE,
            message=f"{UNREADABLE} for session {SESSION}: {detail}",
        )
    )


def test_to_observation_names_every_fault_and_no_session_it_cannot_read() -> None:
    record = harness_record(without=("sessionId",), pid="1", status="")
    assert observe_records.to_observation("colossus", MACHINE, record) == (
        observe_records.RecordRefusal(
            code=FleetErrorCode.SESSION_RECORD_UNREADABLE,
            message=(
                f"{UNREADABLE}: 'sessionId' is NoneType, {NO_STRING}; "
                f"'pid' is str, not an integer; 'status' is str, {NO_STRING}"
            ),
        )
    )


def test_to_observation_refuses_an_unreadable_record_before_comparing_its_machine() -> None:
    """A record that is both unreadable and from another machine is refused as
    unreadable: the mismatch sentence would name a session it cannot copy."""
    record = harness_record(pidDomain="linux:lacrymal", name=None)
    refusal = observe_records.to_observation("colossus", MACHINE, record)
    assert observe_records.is_refusal(refusal)
    assert refusal["code"] is FleetErrorCode.SESSION_RECORD_UNREADABLE


@pytest.mark.parametrize("size", range(len(REQUIRED_FIELDS) + 1))
def test_to_observation_names_exactly_the_missing_fields_of_every_subset(size: int) -> None:
    """Property over every subset of the required fields (1024 records): a
    record lacking those fields is refused naming exactly them, in reading
    order, and a record lacking none is copied."""
    for missing in combinations(REQUIRED_FIELDS, size):
        keys = tuple(key for key, _ in missing)
        result = observe_records.to_observation("colossus", MACHINE, harness_record(without=keys))
        if not missing:
            assert not observe_records.is_refusal(result)
            assert result["sessionId"] == SESSION
            continue
        named = "" if "sessionId" in keys else f" for session {SESSION}"
        faults = "; ".join(f"{key!r} is NoneType, {kind}" for key, kind in missing)
        assert result == observe_records.RecordRefusal(
            code=FleetErrorCode.SESSION_RECORD_UNREADABLE,
            message=f"{UNREADABLE}{named}: {faults}",
        )
