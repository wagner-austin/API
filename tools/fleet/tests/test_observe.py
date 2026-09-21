"""Observing a node's Claude Code sessions into the board's ledger.

The node speaks through the same command hook every other remote test uses
and the board through the same JSON-RPC-over-SSE fake, so what is exercised
is the real script send, the real decode of a record in the shape the
harness writes, and the real request the runner builds. The record fixture
is the field set measured on austinpc on 2026-09-12 (2.1.270, with
``pidDomain``) with the identifiers changed.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str
from platform_core.mcp_client import McpCredentials

from fleet.contracts.node import NodePlatform
from fleet.core import _test_hooks, dialect_linux, dialect_windows, observe, remote
from fleet.core.registry import RegistryNode
from tests._queue_fakes import FakeQueue
from tests.conftest import FakeRun, failed, ok, timed_out

BOARD = McpCredentials(url="http://127.0.0.1:8033/mcp", api_key="k", tenant_id="t")
IDENTITY: JSONObject = {
    "agent": "fleet-runner-austinpc",
    "sessionId": "33333333-cccc-4ccc-8ccc-333333333333",
    "cwd": "C:/Users/Test/PROJECTS/API",
}
SESSION = "d31e5228-3269-4a22-a27f-e535cbef1894"
MACHINE = "win32:serendipity"
STARTED_MS = 1_789_000_000_000
UPDATED_MS = 1_789_000_030_500


def harness_record(*, without: tuple[str, ...] = (), **overrides: JSONValue) -> JSONObject:
    """Build one registration document as the 2.1.270 harness writes it.

    Args:
        without: Fields to leave out, so a test can exercise absence without
            a second fixture.
        **overrides: Fields to vary.

    Returns:
        The record.
    """
    record: JSONObject = {
        "sessionId": SESSION,
        "pid": 4242,
        "startedAt": STARTED_MS,
        "updatedAt": UPDATED_MS,
        "name": "api-7e",
        "nameSource": "derived",
        "cwd": "C:\\Users\\serendipity\\PROJECTS\\API",
        "messagingSocketPath": "\\\\.\\pipe\\LOCAL\\cc-msg-abc",
        "version": "2.1.270",
        "status": "idle",
        "pidDomain": MACHINE,
        "permissionMode": "default",
    }
    for key in without:
        del record[key]
    record.update(overrides)
    return record


def node(
    name: str,
    role: str,
    *,
    enabled: bool = True,
    user: str | None = "austin",
    platform: NodePlatform = "windows",
) -> RegistryNode:
    """Build one identity-registry node.

    Args:
        name: The node name.
        role: hub, worker, vpn-jump or client.
        enabled: Whether the registry says it should answer.
        user: The provisioned account, or None.
        platform: What the registry says it runs.

    Returns:
        The node.
    """
    return RegistryNode(name=name, enabled=enabled, role=role, user=user, platform=platform)


def document(*records: JSONObject, hostname: str = "serendipity", platform: str = "win32") -> str:
    """Render what the observe script prints for these records.

    Args:
        *records: The registration documents.
        hostname: The lowercased hostname the script reports.
        platform: The harness platform tag the script reports.

    Returns:
        The compact JSON the script emits.
    """
    return dump_json_str({"platform": platform, "hostname": hostname, "records": list(records)})


def linux_record() -> JSONObject:
    """One registration document as a Linux node's harness writes it.

    Returns:
        The record, with a POSIX cwd and socket and a ``linux:`` pidDomain.
    """
    return harness_record(
        pidDomain="linux:diphtheria",
        cwd="/home/corvis/PROJECTS/API",
        messagingSocketPath="/run/user/1000/claude-cc-msg-abc",
    )


def test_script_path_is_literal_absolute_and_in_the_platforms_dialect() -> None:
    """The write command on the far side expands nothing, so the home is
    spelled out; the extension is the dialect's (board task cd5010c4)."""
    assert observe.script_path("austin", "windows") == (
        "C:/Users/austin/.fleet/observe-sessions.ps1"
    )
    assert observe.script_path("corvis", "linux") == "/home/corvis/.fleet/observe-sessions.sh"


def test_iso_from_millis_keeps_milliseconds_and_names_utc() -> None:
    assert observe.iso_from_millis(UPDATED_MS) == "2026-09-10T00:27:10.500+00:00"
    assert observe.iso_from_millis(0) == "1970-01-01T00:00:00.000+00:00"


def test_decode_node_sessions_reads_the_document() -> None:
    sessions = observe.decode_node_sessions("serendipity", document(harness_record()))
    assert sessions["platform"] == "win32"
    assert sessions["hostname"] == "serendipity"
    assert len(sessions["records"]) == 1
    assert sessions["records"][0]["sessionId"] == SESSION
    assert observe.machine_of(sessions) == MACHINE


def test_decode_node_sessions_reads_an_empty_directory_as_zero_records() -> None:
    sessions = observe.decode_node_sessions("serendipity", document())
    assert sessions["records"] == ()


@pytest.mark.parametrize(
    ("output", "detail"),
    [
        ("[]", "the document is list, not an object"),
        ('{"platform":"win32","hostname":"x","records":{}}', "'records' is dict, not an array"),
        ('{"platform":"win32","hostname":"x","records":[1]}', "a record is int, not an object"),
        ('{"platform":"win32","records":[]}', "'hostname' is NoneType, not a non-empty string"),
        (
            '{"platform":"","hostname":"x","records":[]}',
            "'platform' is str, not a non-empty string",
        ),
    ],
)
def test_decode_node_sessions_refuses_a_document_it_cannot_read(output: str, detail: str) -> None:
    with pytest.raises(AppError) as caught:
        observe.decode_node_sessions("serendipity", output)
    assert caught.value.code is FleetErrorCode.SESSION_RECORD_UNREADABLE
    assert caught.value.message == (
        f"serendipity reported a session record this runner cannot read: {detail}"
    )


def test_to_observation_copies_every_field_and_converts_the_instants() -> None:
    observation = observe.to_observation("serendipity", MACHINE, harness_record())
    assert observation == observe.SessionObservation(
        sessionId=SESSION,
        pid=4242,
        processStartedAt="2026-09-10T00:26:40.000+00:00",
        name="api-7e",
        nameSource="derived",
        cwd="C:\\Users\\serendipity\\PROJECTS\\API",
        socketPath="\\\\.\\pipe\\LOCAL\\cc-msg-abc",
        harnessVersion="2.1.270",
        status="idle",
        heartbeatAt="2026-09-10T00:27:10.500+00:00",
    )
    assert observe.observation_json(observation) == {
        "sessionId": SESSION,
        "pid": 4242,
        "processStartedAt": "2026-09-10T00:26:40.000+00:00",
        "name": "api-7e",
        "nameSource": "derived",
        "cwd": "C:\\Users\\serendipity\\PROJECTS\\API",
        "socketPath": "\\\\.\\pipe\\LOCAL\\cc-msg-abc",
        "harnessVersion": "2.1.270",
        "status": "idle",
        "heartbeatAt": "2026-09-10T00:27:10.500+00:00",
    }


def test_to_observation_accepts_a_record_without_pid_domain() -> None:
    """2.1.236 writes no ``pidDomain``; the node it was read from is the machine."""
    observation = observe.to_observation(
        "serendipity", MACHINE, harness_record(without=("pidDomain",))
    )
    assert observation["sessionId"] == SESSION


def test_to_observation_refuses_a_record_from_another_machine() -> None:
    with pytest.raises(AppError) as caught:
        observe.to_observation("serendipity", MACHINE, harness_record(pidDomain="win32:austinpc"))
    assert caught.value.code is FleetErrorCode.SESSION_MACHINE_MISMATCH
    assert caught.value.message == (
        f"serendipity: record for session {SESSION} names pidDomain 'win32:austinpc' "
        f"but was read from 'win32:serendipity'"
    )


@pytest.mark.parametrize(
    ("record", "detail"),
    [
        (harness_record(pidDomain=7), "'pidDomain' is int, not a string"),
        (harness_record(pid="4242"), "'pid' is str, not an integer"),
        (harness_record(pid=True), "'pid' is bool, not an integer"),
        (harness_record(without=("startedAt",)), "'startedAt' is NoneType, not an integer"),
        (harness_record(without=("name",)), "'name' is NoneType, not a non-empty string"),
        (harness_record(version=""), "'version' is str, not a non-empty string"),
        (harness_record(status=3), "'status' is int, not a non-empty string"),
    ],
)
def test_to_observation_refuses_a_record_it_cannot_read(record: JSONObject, detail: str) -> None:
    with pytest.raises(AppError) as caught:
        observe.to_observation("serendipity", MACHINE, record)
    assert caught.value.code is FleetErrorCode.SESSION_RECORD_UNREADABLE
    assert caught.value.message == (
        f"serendipity reported a session record this runner cannot read: {detail}"
    )


def test_observe_node_sends_the_script_runs_it_by_path_and_records_on_the_board() -> None:
    run = FakeRun([ok(""), ok(document(harness_record()))])
    _test_hooks.run = run
    board = FakeQueue(["observed 1 session(s) for win32:serendipity (1 new row(s)) at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("serendipity", "worker"), IDENTITY)

    assert outcome == observe.NodeOutcome(
        node="serendipity",
        recorded=True,
        detail="observed 1 session(s) for win32:serendipity (1 new row(s)) at t",
    )
    # The script body went over stdin to the literal path, then ran by path.
    assert run.stdin[0] == dialect_windows.OBSERVE_SESSIONS_SCRIPT.encode("utf-8")
    assert "C:/Users/austin/.fleet/observe-sessions.ps1" in run.calls[0][-1]
    assert run.calls[1] == (
        "ssh",
        *remote.SSH_OPTIONS,
        "serendipity",
        *dialect_windows.POWERSHELL_INVOCATION,
        "C:/Users/austin/.fleet/observe-sessions.ps1",
    )
    assert board.tools == ["task_session_observe"]
    expected_observation: JSONObject = {
        "sessionId": SESSION,
        "pid": 4242,
        "processStartedAt": "2026-09-10T00:26:40.000+00:00",
        "name": "api-7e",
        "nameSource": "derived",
        "cwd": "C:\\Users\\serendipity\\PROJECTS\\API",
        "socketPath": "\\\\.\\pipe\\LOCAL\\cc-msg-abc",
        "harnessVersion": "2.1.270",
        "status": "idle",
        "heartbeatAt": "2026-09-10T00:27:10.500+00:00",
    }
    expected_arguments: JSONObject = {
        "machine": MACHINE,
        "observations": [expected_observation],
        **IDENTITY,
    }
    assert board.arguments == [expected_arguments]


def test_observe_node_records_an_empty_pass_for_a_node_with_no_sessions() -> None:
    """Zero records is a fact worth recording: the machine answered and is idle."""
    _test_hooks.run = FakeRun([ok(""), ok(document(hostname="loki"))])
    board = FakeQueue(["observed 0 session(s) for win32:loki (0 new row(s)) at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("loki", "worker", user="loki"), IDENTITY)

    assert outcome["recorded"] is True
    assert board.arguments[0]["machine"] == "win32:loki"
    assert board.arguments[0]["observations"] == []


def test_observe_node_speaks_sh_to_a_linux_node_and_records_its_machine() -> None:
    """Board task cd5010c4, acceptance 2: a Linux node in the registry gets
    the sh script at the sh path, run by /bin/sh, and its records land on
    the board under the ``linux:`` machine its own document spells."""
    run = FakeRun([ok(""), ok(document(linux_record(), hostname="diphtheria", platform="linux"))])
    _test_hooks.run = run
    board = FakeQueue(["observed 1 session(s) for linux:diphtheria (1 new row(s)) at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(
        BOARD, node("diphtheria", "worker", user="corvis", platform="linux"), IDENTITY
    )

    assert outcome == observe.NodeOutcome(
        node="diphtheria",
        recorded=True,
        detail="observed 1 session(s) for linux:diphtheria (1 new row(s)) at t",
    )
    path = "/home/corvis/.fleet/observe-sessions.sh"
    assert run.stdin[0] == dialect_linux.OBSERVE_SESSIONS_SCRIPT.encode("utf-8")
    assert run.calls[0] == (
        "ssh",
        *remote.SSH_OPTIONS,
        "diphtheria",
        dialect_linux.WRITE_COMMAND.format(path=path),
    )
    assert run.calls[1] == (
        "ssh",
        *remote.SSH_OPTIONS,
        "diphtheria",
        *dialect_linux.SH_INVOCATION,
        path,
    )
    assert board.tools == ["task_session_observe"]
    assert board.arguments[0]["machine"] == "linux:diphtheria"
    assert board.arguments[0]["observations"] == [
        {
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
    ]


def test_observe_node_records_a_linux_node_that_has_never_run_claude_code() -> None:
    """diphtheria as measured on 2026-09-21: no ``~/.claude`` at all. The
    honest line is zero sessions for ``linux:diphtheria``, and it is recorded,
    which is what closes the ledger coverage gap the task was filed for."""
    _test_hooks.run = FakeRun([ok(""), ok(document(hostname="diphtheria", platform="linux"))])
    board = FakeQueue(["observed 0 session(s) for linux:diphtheria (0 new row(s)) at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(
        BOARD, node("diphtheria", "worker", user="corvis", platform="linux"), IDENTITY
    )

    assert observe.render_outcome(outcome) == (
        "diphtheria: observed 0 session(s) for linux:diphtheria (0 new row(s)) at t"
    )
    assert board.arguments == [{"machine": "linux:diphtheria", "observations": [], **IDENTITY}]


def test_observe_node_reports_an_unreachable_node_as_an_outcome() -> None:
    _test_hooks.run = FakeRun([failed(255, "ssh: connect to host loki port 22: timed out")])
    board = FakeQueue([])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("loki", "worker"), IDENTITY)

    assert outcome == observe.NodeOutcome(
        node="loki",
        recorded=False,
        detail=(
            "ssh to loki failed while sending C:/Users/austin/.fleet/observe-sessions.ps1: "
            "ssh: connect to host loki port 22: timed out"
        ),
    )
    assert board.tools == []


def test_observe_pass_records_every_other_node_when_one_times_out() -> None:
    """Board task 41ac6ed2, acceptance 2: the ssh to one node outlives its
    deadline (the shape of the 2026-09-17 pendragon wedge), the pass reports
    that node as not observed with the elapsed bound, and still records the
    node after it instead of holding the tick."""
    run = FakeRun([timed_out(120), ok(""), ok(document(harness_record()))])
    _test_hooks.run = run
    board = FakeQueue(["observed 1 session(s) for win32:serendipity (0 new row(s)) at t"])
    _test_hooks.http_post = board
    registry = {
        "pendragon": node("pendragon", "worker", user="austi"),
        "serendipity": node("serendipity", "worker"),
    }

    outcomes = observe.observe_pass(BOARD, registry, IDENTITY)

    assert [observe.render_outcome(o) for o in outcomes] == [
        "pendragon: not observed -- ssh to pendragon timed out while sending "
        "C:/Users/austi/.fleet/observe-sessions.ps1: timed out after 120 s",
        "serendipity: observed 1 session(s) for win32:serendipity (0 new row(s)) at t",
    ]
    assert [a["machine"] for a in board.arguments] == ["win32:serendipity"]
    # Every ssh the pass made carried the deadline that ended the first one.
    assert run.timeouts == [remote.SSH_TIMEOUT_SECONDS] * 3


def test_observe_node_reports_a_node_without_a_provisioned_user() -> None:
    _test_hooks.run = FakeRun([])
    outcome = observe.observe_node(BOARD, node("ghost", "worker", user=None), IDENTITY)
    assert outcome == observe.NodeOutcome(
        node="ghost", recorded=False, detail="no provisioned user in the registry"
    )


def test_observe_node_raises_on_a_record_from_another_machine() -> None:
    _test_hooks.run = FakeRun([ok(""), ok(document(harness_record(pidDomain="win32:austinpc")))])
    board = FakeQueue([])
    _test_hooks.http_post = board
    with pytest.raises(AppError) as caught:
        observe.observe_node(BOARD, node("serendipity", "worker"), IDENTITY)
    assert caught.value.code is FleetErrorCode.SESSION_MACHINE_MISMATCH
    assert board.tools == []


def test_observe_pass_visits_enabled_workers_and_jump_hosts_in_name_order() -> None:
    _test_hooks.run = FakeRun(
        [
            failed(255, "loki is off"),
            ok(""),
            ok(document(harness_record())),
            ok(""),
            ok(document(hostname="viper")),
        ]
    )
    board = FakeQueue(
        [
            "observed 1 session(s) for win32:serendipity (0 new row(s)) at t",
            "observed 0 session(s) for win32:viper (0 new row(s)) at t",
        ]
    )
    _test_hooks.http_post = board
    registry = {
        "viper": node("viper", "vpn-jump", user="viper"),
        "austinpc": node("austinpc", "hub"),
        "serendipity": node("serendipity", "worker"),
        "phone": node("phone", "client", user=None),
        "lavender": node("lavender", "worker", enabled=False),
        "loki": node("loki", "worker", user="loki"),
    }

    outcomes = observe.observe_pass(BOARD, registry, IDENTITY)

    assert [o["node"] for o in outcomes] == ["loki", "serendipity", "viper"]
    assert [o["recorded"] for o in outcomes] == [False, True, True]
    assert [a["machine"] for a in board.arguments] == ["win32:serendipity", "win32:viper"]


def test_render_outcome_distinguishes_a_quiet_node_from_a_recorded_one() -> None:
    assert (
        observe.render_outcome(observe.NodeOutcome(node="s", recorded=True, detail="observed 2"))
        == "s: observed 2"
    )
    assert (
        observe.render_outcome(observe.NodeOutcome(node="loki", recorded=False, detail="off"))
        == "loki: not observed -- off"
    )
