"""Observing a node's Claude Code sessions into the board's ledger.

The node speaks through the same command hook every other remote test uses
and the board through the same JSON-RPC-over-SSE fake, so what is exercised
is the real script send, the real decode of a record in the shape the
harness writes, and the real request the runner builds. The record fixture
is the field set measured on austinpc on 2026-09-12 (2.1.270, with
``pidDomain``) with the identifiers and paths changed to a Linux node's.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str
from platform_core.mcp_client import McpCredentials

from fleet.contracts.machine import LedgerObserver, Machine, MachineRole
from fleet.core import _test_hooks, dialect, observe, remote
from fleet.core.ssh_options import SSH_OPTIONS
from tests._queue_fakes import FakeQueue
from tests.conftest import FakeRun, failed, ok, timed_out

BOARD = McpCredentials(url="http://127.0.0.1:8033/mcp", api_key="k", tenant_id="t")
IDENTITY: JSONObject = {
    "agent": "fleet-node-colossus",
    "sessionId": "33333333-cccc-4ccc-8ccc-333333333333",
    "cwd": "/home/corvis/PROJECTS/API",
}
SESSION = "d31e5228-3269-4a22-a27f-e535cbef1894"
MACHINE = "linux:colossus"
STARTED_MS = 1_789_000_000_000
UPDATED_MS = 1_789_000_030_500

#: Where the observe script lands under the fleet's account on every node.
SCRIPT_PATH = "/home/corvis/.fleet/observe-sessions.sh"


def harness_record(*, without: tuple[str, ...] = (), **overrides: JSONValue) -> JSONObject:
    """Build one registration document as the 2.1.270 harness writes it.

    Args:
        without: Fields to leave out, so a test can exercise absence without
            a second fixture.
        **overrides: Fields to vary.

    Returns:
        The record, whose ``pidDomain`` is the ``<platform>:<hostname>``
        spelling every harness wrote before 2.1.294.
    """
    record: JSONObject = {
        "sessionId": SESSION,
        "pid": 4242,
        "startedAt": STARTED_MS,
        "updatedAt": UPDATED_MS,
        "name": "api-7e",
        "nameSource": "derived",
        "cwd": "/home/corvis/PROJECTS/API",
        "messagingSocketPath": "/run/user/1000/claude-cc-msg-abc",
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
    user: str | None = "corvis",
    observer: LedgerObserver = LedgerObserver.FLEET_ARM,
) -> Machine:
    """Build one roster machine that declares no dispatch capacity.

    Args:
        name: The machine name.
        role: hub, worker, vpn-jump, session-host or client.
        enabled: Whether the roster says it should answer.
        user: The provisioned account, or None.
        observer: The arm the roster says records its sessions.

    Returns:
        The machine.
    """
    return Machine(
        name=name,
        role=MachineRole(role),
        user=user,
        enabled=enabled,
        ledger_observer=observer,
        dispatch=None,
        held="observed, never dispatched to",
    )


def document(
    *records: JSONObject,
    hostname: str = "colossus",
    platform: str = "linux",
    pid_domain: str | None = None,
) -> str:
    """Render what the observe script prints for these records.

    Args:
        *records: The registration documents.
        hostname: The lowercased hostname the script reports.
        platform: The harness platform tag the script reports.
        pid_domain: The pidDomain the script reports; None spells it
            ``<platform>:<hostname>``, as every harness before 2.1.294 did.

    Returns:
        The compact JSON the script emits.
    """
    return dump_json_str(
        {
            "platform": platform,
            "hostname": hostname,
            "pidDomain": f"{platform}:{hostname}" if pid_domain is None else pid_domain,
            "records": list(records),
        }
    )


#: The pidDomain Claude Code 2.1.294 writes on diphtheria: its
#: ``/etc/machine-id`` and pid namespace, measured 2026-10-16 (board task
#: 5cee2f65). Its ledger machine key stays ``linux:diphtheria``.
DIPHTHERIA_PID_DOMAIN = "linux:212dd22057a34d6587f019c387979599:pid:[4026531836]"


def linux_record() -> JSONObject:
    """One registration document as a node's 2.1.294 harness writes it.

    Returns:
        The record, with the machine-id pidDomain.
    """
    return harness_record(pidDomain=DIPHTHERIA_PID_DOMAIN)


def test_script_path_is_literal_and_absolute() -> None:
    """The write command on the far side expands nothing, so the home is
    spelled out (board task cd5010c4)."""
    assert observe.script_path("corvis") == SCRIPT_PATH
    assert observe.script_path("austin") == "/home/austin/.fleet/observe-sessions.sh"


def test_iso_from_millis_keeps_milliseconds_and_names_utc() -> None:
    assert observe.iso_from_millis(UPDATED_MS) == "2026-09-10T00:27:10.500+00:00"
    assert observe.iso_from_millis(0) == "1970-01-01T00:00:00.000+00:00"


def test_decode_node_sessions_reads_the_document() -> None:
    sessions = observe.decode_node_sessions("colossus", document(harness_record()))
    assert sessions["platform"] == "linux"
    assert sessions["hostname"] == "colossus"
    assert sessions["pid_domain"] == MACHINE
    assert len(sessions["records"]) == 1
    assert sessions["records"][0]["sessionId"] == SESSION
    assert observe.machine_of(sessions) == MACHINE


def test_decode_node_sessions_reads_an_empty_directory_as_zero_records() -> None:
    sessions = observe.decode_node_sessions("colossus", document())
    assert sessions["records"] == ()


@pytest.mark.parametrize(
    ("output", "detail"),
    [
        ("[]", "the document is list, not an object"),
        ('{"platform":"linux","hostname":"x","records":{}}', "'records' is dict, not an array"),
        ('{"platform":"linux","hostname":"x","records":[1]}', "a record is int, not an object"),
        ('{"platform":"linux","records":[]}', "'hostname' is NoneType, not a non-empty string"),
        (
            '{"platform":"","hostname":"x","records":[]}',
            "'platform' is str, not a non-empty string",
        ),
        (
            '{"platform":"linux","hostname":"x","records":[]}',
            "'pidDomain' is NoneType, not a non-empty string",
        ),
    ],
)
def test_decode_node_sessions_refuses_a_document_it_cannot_read(output: str, detail: str) -> None:
    with pytest.raises(AppError) as caught:
        observe.decode_node_sessions("colossus", output)
    assert caught.value.code is FleetErrorCode.SESSION_RECORD_UNREADABLE
    assert caught.value.message == (
        f"colossus reported a session record this runner cannot read: {detail}"
    )


def test_to_observation_copies_every_field_and_converts_the_instants() -> None:
    observation = observe.to_observation("colossus", MACHINE, harness_record())
    assert observation == observe.SessionObservation(
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
    assert observe.observation_json(observation) == {
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
    observation = observe.to_observation(
        "colossus", MACHINE, harness_record(without=("pidDomain",))
    )
    assert observation["sessionId"] == SESSION


def test_to_observation_refuses_a_record_from_another_machine() -> None:
    with pytest.raises(AppError) as caught:
        observe.to_observation("colossus", MACHINE, harness_record(pidDomain="linux:lacrymal"))
    assert caught.value.code is FleetErrorCode.SESSION_MACHINE_MISMATCH
    assert caught.value.message == (
        f"colossus: record for session {SESSION} names pidDomain 'linux:lacrymal' "
        f"but its node reports 'linux:colossus'"
    )


#: The pidDomain Claude Code 2.1.294 wrote on ziggurat, measured 2026-10-16.
ZIGGURAT_PID_DOMAIN = "linux:f191c99155c14a719632bdc88e77f914:pid:[4026531836]"


def test_to_observation_accepts_the_2_1_294_spelling_its_node_reports() -> None:
    """Board task 5cee2f65: every hub tick from 06:30Z failed here, because
    the record was compared to ``linux:ziggurat``. The node now reports the
    spelling its harness writes, and the record is compared to that."""
    record = harness_record(pidDomain=ZIGGURAT_PID_DOMAIN, version="2.1.294")

    observation = observe.to_observation("ziggurat", ZIGGURAT_PID_DOMAIN, record)

    assert observation["harnessVersion"] == "2.1.294"


def test_to_observation_refuses_a_record_from_another_machine_id() -> None:
    other = "linux:b016a0cfb2c942b0a2cc55b43e865711:pid:[4026531836]"
    with pytest.raises(AppError) as caught:
        observe.to_observation("ziggurat", ZIGGURAT_PID_DOMAIN, harness_record(pidDomain=other))
    assert caught.value.code is FleetErrorCode.SESSION_MACHINE_MISMATCH
    assert caught.value.message == (
        f"ziggurat: record for session {SESSION} names pidDomain {other!r} "
        f"but its node reports {ZIGGURAT_PID_DOMAIN!r}"
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
        observe.to_observation("colossus", MACHINE, record)
    assert caught.value.code is FleetErrorCode.SESSION_RECORD_UNREADABLE
    assert caught.value.message == (
        f"colossus reported a session record this runner cannot read: {detail}"
    )


def test_observe_node_sends_the_script_runs_it_by_path_and_records_on_the_board() -> None:
    """Board task cd5010c4, acceptance 2: the sh script at the sh path, run
    by /bin/sh, and its records land on the board under the machine its own
    document spells."""
    run = FakeRun([ok(""), ok(document(harness_record()))])
    _test_hooks.run = run
    board = FakeQueue(["observed 1 session(s) for linux:colossus (1 new row(s)) at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("colossus", "worker"), IDENTITY)

    assert outcome == observe.NodeOutcome(
        node="colossus",
        recorded=True,
        detail="observed 1 session(s) for linux:colossus (1 new row(s)) at t",
    )
    # The script body went over stdin to the literal path, then ran by path.
    assert run.stdin[0] == dialect.OBSERVE_SESSIONS_SCRIPT.encode("utf-8")
    assert run.calls[0] == (
        "ssh",
        *SSH_OPTIONS,
        "colossus",
        dialect.WRITE_COMMAND.format(path=SCRIPT_PATH),
    )
    assert run.calls[1] == ("ssh", *SSH_OPTIONS, "colossus", *dialect.SH_INVOCATION, SCRIPT_PATH)
    assert board.tools == ["task_session_observe"]
    expected_observation: JSONObject = {
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
    expected_arguments: JSONObject = {
        "machine": MACHINE,
        "observations": [expected_observation],
        **IDENTITY,
    }
    assert board.arguments == [expected_arguments]


def test_observe_node_records_an_empty_pass_for_a_node_with_no_sessions() -> None:
    """Zero records is a fact worth recording: the machine answered and is idle."""
    _test_hooks.run = FakeRun([ok(""), ok(document(hostname="lacrymal"))])
    board = FakeQueue(["observed 0 session(s) for linux:lacrymal (0 new row(s)) at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("lacrymal", "worker"), IDENTITY)

    assert outcome["recorded"] is True
    assert board.arguments[0]["machine"] == "linux:lacrymal"
    assert board.arguments[0]["observations"] == []


def test_observe_node_keys_a_2_1_294_node_by_its_hostname() -> None:
    """The record carries the 2.1.294 pidDomain, which is checked against the
    one the node reports, while the ledger keys it ``linux:diphtheria``
    (board task 5cee2f65)."""
    run = FakeRun(
        [
            ok(""),
            ok(document(linux_record(), hostname="diphtheria", pid_domain=DIPHTHERIA_PID_DOMAIN)),
        ]
    )
    _test_hooks.run = run
    board = FakeQueue(["observed 1 session(s) for linux:diphtheria (1 new row(s)) at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("diphtheria", "worker"), IDENTITY)

    assert outcome == observe.NodeOutcome(
        node="diphtheria",
        recorded=True,
        detail="observed 1 session(s) for linux:diphtheria (1 new row(s)) at t",
    )
    assert board.arguments[0]["machine"] == "linux:diphtheria"


def test_observe_node_records_a_node_that_has_never_run_claude_code() -> None:
    """diphtheria as measured on 2026-09-21: no ``~/.claude`` at all. The
    honest line is zero sessions for ``linux:diphtheria``, and it is recorded,
    which is what closes the ledger coverage gap the task was filed for."""
    _test_hooks.run = FakeRun(
        [ok(""), ok(document(hostname="diphtheria", pid_domain=DIPHTHERIA_PID_DOMAIN))]
    )
    board = FakeQueue(["observed 0 session(s) for linux:diphtheria (0 new row(s)) at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("diphtheria", "worker"), IDENTITY)

    assert observe.render_outcome(outcome) == (
        "diphtheria: observed 0 session(s) for linux:diphtheria (0 new row(s)) at t"
    )
    assert board.arguments == [{"machine": "linux:diphtheria", "observations": [], **IDENTITY}]


def test_observe_node_reports_an_unreachable_node_as_an_outcome() -> None:
    _test_hooks.run = FakeRun([failed(255, "ssh: connect to host lacrymal port 22: timed out")])
    board = FakeQueue([])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("lacrymal", "worker"), IDENTITY)

    assert outcome == observe.NodeOutcome(
        node="lacrymal",
        recorded=False,
        detail=(
            f"ssh to lacrymal failed while sending {SCRIPT_PATH}: "
            "ssh: connect to host lacrymal port 22: timed out"
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
    board = FakeQueue(["observed 1 session(s) for linux:colossus (0 new row(s)) at t"])
    _test_hooks.http_post = board
    machines = (node("pendragon", "worker"), node("colossus", "worker"))

    outcomes = observe.observe_pass(BOARD, machines, IDENTITY)

    assert [observe.render_outcome(o) for o in outcomes] == [
        "pendragon: not observed -- ssh to pendragon timed out while sending "
        f"{SCRIPT_PATH}: timed out after 120 s",
        "colossus: observed 1 session(s) for linux:colossus (0 new row(s)) at t",
    ]
    assert [a["machine"] for a in board.arguments] == ["linux:colossus"]
    # Every ssh the pass made carried the deadline that ended the first one.
    assert run.timeouts == [remote.SSH_TIMEOUT_SECONDS] * 3


def test_observe_node_reports_a_node_without_a_provisioned_user() -> None:
    _test_hooks.run = FakeRun([])
    outcome = observe.observe_node(BOARD, node("ghost", "worker", user=None), IDENTITY)
    assert outcome == observe.NodeOutcome(
        node="ghost", recorded=False, detail="no provisioned user in the registry"
    )


def test_observe_node_raises_on_a_record_from_another_machine() -> None:
    _test_hooks.run = FakeRun([ok(""), ok(document(harness_record(pidDomain="linux:lacrymal")))])
    board = FakeQueue([])
    _test_hooks.http_post = board
    with pytest.raises(AppError) as caught:
        observe.observe_node(BOARD, node("colossus", "worker"), IDENTITY)
    assert caught.value.code is FleetErrorCode.SESSION_MACHINE_MISMATCH
    assert board.tools == []


def test_observe_pass_visits_enabled_fleet_arm_machines_in_name_order() -> None:
    _test_hooks.run = FakeRun(
        [
            ok(""),
            ok(document(harness_record())),
            failed(255, "lacrymal is off"),
            ok(""),
            ok(document(hostname="viper")),
        ]
    )
    board = FakeQueue(
        [
            "observed 1 session(s) for linux:colossus (0 new row(s)) at t",
            "observed 0 session(s) for linux:viper (0 new row(s)) at t",
        ]
    )
    _test_hooks.http_post = board
    registry = {
        "viper": node("viper", "vpn-jump"),
        "hub": node("hub", "hub", observer=LedgerObserver.PCSESSION),
        "colossus": node("colossus", "worker"),
        "phone": node("phone", "client", user=None, observer=LedgerObserver.NONE),
        "lavender": node("lavender", "worker", enabled=False),
        "lacrymal": node("lacrymal", "worker"),
    }

    outcomes = observe.observe_pass(BOARD, observe.observable(registry), IDENTITY)

    assert [o["node"] for o in outcomes] == ["colossus", "lacrymal", "viper"]
    assert [o["recorded"] for o in outcomes] == [True, False, True]
    assert [a["machine"] for a in board.arguments] == ["linux:colossus", "linux:viper"]


def test_observe_pass_visits_a_session_host() -> None:
    # MCPs board task 701151b4: the node the operator's sessions run on.
    _test_hooks.run = FakeRun([ok(""), ok(document(hostname="ziggurat"))])
    board = FakeQueue(["observed 0 session(s) for linux:ziggurat (0 new row(s)) at t"])
    _test_hooks.http_post = board

    registry = {
        "ziggurat": node("ziggurat", "session-host"),
        "hub": node("hub", "hub", observer=LedgerObserver.PCSESSION),
    }

    outcomes = observe.observe_pass(BOARD, observe.observable(registry), IDENTITY)

    assert [(o["node"], o["recorded"]) for o in outcomes] == [("ziggurat", True)]
    assert [a["machine"] for a in board.arguments] == ["linux:ziggurat"]


def test_observable_follows_the_declared_observer_not_the_role() -> None:
    # MCPs board task 911b0303: diphtheria became a hub on 2026-10-18 while
    # still declaring fleet-arm, and a role filter dropped it from every pass.
    registry = {
        "diphtheria": node("diphtheria", "hub"),
        "hub": node("hub", "hub", observer=LedgerObserver.PCSESSION),
        "relative": node("relative", "worker", observer=LedgerObserver.NONE),
        "ziggurat": node("ziggurat", "session-host"),
    }

    assert [m["name"] for m in observe.observable(registry)] == ["diphtheria", "ziggurat"]


def test_render_outcome_distinguishes_a_quiet_node_from_a_recorded_one() -> None:
    assert (
        observe.render_outcome(observe.NodeOutcome(node="s", recorded=True, detail="observed 2"))
        == "s: observed 2"
    )
    assert (
        observe.render_outcome(observe.NodeOutcome(node="lacrymal", recorded=False, detail="off"))
        == "lacrymal: not observed -- off"
    )
