"""Observing a node's Claude Code sessions into the board's ledger.

The node speaks through the same command hook every other remote test uses
and the board through the same JSON-RPC-over-SSE fake, so what is exercised
is the real script send, the real decode of a record in the shape the
harness writes, and the real request the runner builds. The fixtures are
:mod:`tests._observe_fixtures`; the decode's own cases are
``test_observe_records``.

A node that cannot be observed is that node's OUTCOME (MCPs board task
56fee128): ssh not reaching it, or a record it reports that this runner
cannot read, is recorded on the board as the node's refusal in place of a
pass, and every later node is still observed.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject

from fleet.contracts.machine import LedgerObserver
from fleet.core import _test_hooks, dialect, observe, remote
from fleet.core.ssh_options import SSH_OPTIONS
from tests._observe_fixtures import (
    BOARD,
    DIPHTHERIA_PID_DOMAIN,
    EMPTY_IDENTITIES,
    EVIDENCE,
    IDENTITY,
    MACHINE,
    SCRIPT_PATH,
    SESSION,
    document,
    harness_record,
    linux_record,
    node,
)
from tests._queue_fakes import FakeQueue
from tests.conftest import FakeRun, failed, ok, timed_out

#: A second session on the same node, for a case that reads two records.
OTHER_SESSION = "5916b503-0e9a-4d0c-9d3e-1f0a6c2b7e41"


def refusal_arguments(machine: str, code: str, detail: str) -> JSONObject:
    """The request a node's refusal sends to ``task_session_observe``.

    Args:
        machine: The ledger machine key.
        code: The refusal's code.
        detail: The refusal's sentence.

    Returns:
        The tool's arguments: no records, the refusal, the runner's identity.
    """
    refusal: JSONObject = {"code": code, "detail": detail}
    return {"machine": machine, "observations": [], "refusal": refusal, **IDENTITY}


def test_script_path_is_literal_and_absolute() -> None:
    """The write command on the far side expands nothing, so the home is
    spelled out (board task cd5010c4)."""
    assert observe.script_path("corvis") == SCRIPT_PATH
    assert observe.script_path("austin") == "/home/austin/.fleet/observe-sessions.sh"


def test_observe_node_sends_the_script_runs_it_by_path_and_records_on_the_board() -> None:
    """Board task cd5010c4, acceptance 2: the sh script at the sh path, run
    by /bin/sh, and its records land on the board under the machine its own
    document spells."""
    run = FakeRun([ok(""), ok(document(harness_record()))])
    _test_hooks.run = run
    board = FakeQueue(
        [EMPTY_IDENTITIES, "observed 1 session(s) for linux:colossus (1 new row(s)) at t"]
    )
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("colossus", "worker"), IDENTITY, EVIDENCE)

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
    assert board.tools == ["task_sessions_list", "task_session_observe"]
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
        "nativeObservations": [],
        **IDENTITY,
    }
    assert board.arguments == [
        {**IDENTITY, "machine": MACHINE, "format": "json", "limit": 200, "offset": 0},
        expected_arguments,
    ]


def test_observe_node_records_an_empty_pass_for_a_node_with_no_sessions() -> None:
    """Zero records is a fact worth recording: the machine answered and is idle."""
    _test_hooks.run = FakeRun([ok(""), ok(document(hostname="lacrymal"))])
    board = FakeQueue(
        [EMPTY_IDENTITIES, "observed 0 session(s) for linux:lacrymal (0 new row(s)) at t"]
    )
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("lacrymal", "worker"), IDENTITY, EVIDENCE)

    assert outcome["recorded"] is True
    assert board.arguments[0]["machine"] == "linux:lacrymal"
    assert board.arguments[1]["observations"] == []


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
    board = FakeQueue(
        [EMPTY_IDENTITIES, "observed 1 session(s) for linux:diphtheria (1 new row(s)) at t"]
    )
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("diphtheria", "worker"), IDENTITY, EVIDENCE)

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
    board = FakeQueue(
        [EMPTY_IDENTITIES, "observed 0 session(s) for linux:diphtheria (0 new row(s)) at t"]
    )
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("diphtheria", "worker"), IDENTITY, EVIDENCE)

    assert observe.render_outcome(outcome) == (
        "diphtheria: observed 0 session(s) for linux:diphtheria (0 new row(s)) at t"
    )
    assert board.arguments == [
        {**IDENTITY, "machine": "linux:diphtheria", "format": "json", "limit": 200, "offset": 0},
        {"machine": "linux:diphtheria", "observations": [], "nativeObservations": [], **IDENTITY},
    ]


def test_observe_node_reports_an_unreachable_node_as_an_outcome() -> None:
    """The node ssh does not reach is recorded on the board as its refusal,
    under the roster's machine key, so the ledger reads it as not observed
    with ssh's own reason; its outcome is the board's line for that."""
    _test_hooks.run = FakeRun([failed(255, "ssh: connect to host lacrymal port 22: timed out")])
    board = FakeQueue(["not observed: linux:lacrymal refused at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("lacrymal", "worker"), IDENTITY, EVIDENCE)

    assert outcome == observe.NodeOutcome(
        node="lacrymal", recorded=False, detail="not observed: linux:lacrymal refused at t"
    )
    assert board.tools == ["task_session_observe"]
    assert board.arguments == [
        refusal_arguments(
            "linux:lacrymal",
            "NODE_UNREACHABLE",
            f"ssh to lacrymal failed while sending {SCRIPT_PATH}: "
            "ssh: connect to host lacrymal port 22: timed out",
        )
    ]


def test_observe_node_reports_a_script_that_exits_nonzero_as_its_refusal() -> None:
    """A record file the script cannot even parse stops the script, which
    exits nonzero: the node is refused ``DISPATCH_FAILED`` with the script's
    own last words, on the board, and nothing is recorded for it."""
    stderr = "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
    _test_hooks.run = FakeRun([ok(""), failed(1, stderr)])
    board = FakeQueue(["not observed: linux:demure refused at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("demure", "server"), IDENTITY, EVIDENCE)

    assert outcome["recorded"] is False
    assert board.arguments == [
        refusal_arguments(
            "linux:demure",
            "DISPATCH_FAILED",
            f"running `/bin/sh {SCRIPT_PATH}` on demure exited 1: {stderr}",
        )
    ]


def test_observe_pass_records_every_other_node_when_one_times_out() -> None:
    """Board task 41ac6ed2, acceptance 2: the ssh to one node outlives its
    deadline (the shape of the 2026-09-17 pendragon wedge), the pass reports
    that node as not observed with the elapsed bound, and still records the
    node after it instead of holding the tick."""
    run = FakeRun([timed_out(120), ok(""), ok(document(harness_record()))])
    _test_hooks.run = run
    board = FakeQueue(
        [
            "not observed: linux:pendragon refused at t",
            EMPTY_IDENTITIES,
            "observed 1 session(s) for linux:colossus (0 new row(s)) at t",
        ]
    )
    _test_hooks.http_post = board
    machines = (node("pendragon", "worker"), node("colossus", "worker"))

    outcomes = observe.observe_pass(BOARD, machines, IDENTITY, EVIDENCE)

    assert [observe.render_outcome(o) for o in outcomes] == [
        "pendragon: not observed -- not observed: linux:pendragon refused at t",
        "colossus: observed 1 session(s) for linux:colossus (0 new row(s)) at t",
    ]
    assert board.arguments[0] == refusal_arguments(
        "linux:pendragon",
        "NODE_UNREACHABLE",
        f"ssh to pendragon timed out while sending {SCRIPT_PATH}: timed out after 120 s",
    )
    assert [a["machine"] for a in board.arguments] == [
        "linux:pendragon",
        "linux:colossus",
        "linux:colossus",
    ]
    # Every ssh the pass made carried the deadline that ended the first one.
    assert run.timeouts == [remote.SSH_TIMEOUT_SECONDS] * 3


def test_observe_pass_records_every_other_node_when_one_reports_an_unreadable_record() -> None:
    """Board task 56fee128, acceptance 1, in the shape of 2026-11-25T01:09Z:
    demure holds one readable record and one whose ``messagingSocketPath``
    is null. The nodes before and after it are recorded; demure records no
    pass -- its readable record alone would depart the other -- and its
    refusal goes to the board naming the session and the field."""
    unreadable = harness_record(sessionId=OTHER_SESSION, messagingSocketPath=None)
    _test_hooks.run = FakeRun(
        [
            ok(""),
            ok(document(hostname="citrine")),
            ok(""),
            ok(document(harness_record(pidDomain="linux:demure"), unreadable, hostname="demure")),
            ok(""),
            ok(document(hostname="ziggurat")),
        ]
    )
    board = FakeQueue(
        [
            EMPTY_IDENTITIES,
            "observed 0 session(s) for linux:citrine (0 new row(s)) at t",
            "not observed: linux:demure refused at t",
            EMPTY_IDENTITIES,
            "observed 0 session(s) for linux:ziggurat (0 new row(s)) at t",
        ]
    )
    _test_hooks.http_post = board
    machines = (node("citrine", "worker"), node("demure", "server"), node("ziggurat", "worker"))

    outcomes = observe.observe_pass(BOARD, machines, IDENTITY, EVIDENCE)

    assert [(o["node"], o["recorded"]) for o in outcomes] == [
        ("citrine", True),
        ("demure", False),
        ("ziggurat", True),
    ]
    assert observe.render_outcome(outcomes[1]) == (
        "demure: not observed -- not observed: linux:demure refused at t"
    )
    assert board.tools == [
        "task_sessions_list",
        "task_session_observe",
        "task_session_observe",
        "task_sessions_list",
        "task_session_observe",
    ]
    assert board.arguments[2] == refusal_arguments(
        "linux:demure",
        "SESSION_RECORD_UNREADABLE",
        "refused 1 of 2 session record(s): demure reported a session record this runner "
        f"cannot read for session {OTHER_SESSION}: 'messagingSocketPath' is NoneType, "
        "not a non-empty string",
    )


def test_observe_node_names_every_refused_record_under_the_first_ones_code() -> None:
    mismatched = harness_record(pidDomain="linux:lacrymal")
    unreadable = harness_record(sessionId=OTHER_SESSION, pid="4242")
    _test_hooks.run = FakeRun([ok(""), ok(document(mismatched, harness_record(), unreadable))])
    board = FakeQueue(["not observed: linux:colossus refused at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("colossus", "worker"), IDENTITY, EVIDENCE)

    assert outcome == observe.NodeOutcome(
        node="colossus", recorded=False, detail="not observed: linux:colossus refused at t"
    )
    assert board.arguments == [
        refusal_arguments(
            MACHINE,
            "SESSION_MACHINE_MISMATCH",
            f"refused 2 of 3 session record(s): colossus: record for session {SESSION} "
            "names pidDomain 'linux:lacrymal' but its node reports 'linux:colossus'; "
            "colossus reported a session record this runner cannot read for session "
            f"{OTHER_SESSION}: 'pid' is str, not an integer",
        )
    ]


def test_observe_node_reports_a_document_it_cannot_read_under_the_roster_key() -> None:
    """A document that does not decode spells no machine of its own, so the
    refusal is keyed ``linux:<name>``, the key the fleet audit holds."""
    _test_hooks.run = FakeRun([ok(""), ok('{"records":{}}')])
    board = FakeQueue(["not observed: linux:demure refused at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("demure", "server"), IDENTITY, EVIDENCE)

    assert outcome["recorded"] is False
    assert board.arguments == [
        refusal_arguments(
            "linux:demure",
            "SESSION_RECORD_UNREADABLE",
            "demure reported a session record this runner cannot read: "
            "'records' is dict, not an array",
        )
    ]


def test_observe_node_reports_a_node_without_a_provisioned_user() -> None:
    _test_hooks.run = FakeRun([])
    outcome = observe.observe_node(BOARD, node("ghost", "worker", user=None), IDENTITY, EVIDENCE)
    assert outcome == observe.NodeOutcome(
        node="ghost", recorded=False, detail="no provisioned user in the registry"
    )


def test_observe_node_reports_a_record_from_another_machine_as_its_outcome() -> None:
    _test_hooks.run = FakeRun([ok(""), ok(document(harness_record(pidDomain="linux:lacrymal")))])
    board = FakeQueue(["not observed: linux:colossus refused at t"])
    _test_hooks.http_post = board

    outcome = observe.observe_node(BOARD, node("colossus", "worker"), IDENTITY, EVIDENCE)

    assert outcome["recorded"] is False
    assert board.tools == ["task_session_observe"]
    assert board.arguments[0]["refusal"] == {
        "code": "SESSION_MACHINE_MISMATCH",
        "detail": (
            f"refused 1 of 1 session record(s): colossus: record for session {SESSION} "
            "names pidDomain 'linux:lacrymal' but its node reports 'linux:colossus'"
        ),
    }


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
            EMPTY_IDENTITIES,
            "observed 1 session(s) for linux:colossus (0 new row(s)) at t",
            "not observed: linux:lacrymal refused at t",
            EMPTY_IDENTITIES,
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

    outcomes = observe.observe_pass(BOARD, observe.observable(registry), IDENTITY, EVIDENCE)

    assert [o["node"] for o in outcomes] == ["colossus", "lacrymal", "viper"]
    assert [o["recorded"] for o in outcomes] == [True, False, True]
    assert [a["machine"] for a in board.arguments] == [
        "linux:colossus",
        "linux:colossus",
        "linux:lacrymal",
        "linux:viper",
        "linux:viper",
    ]


def test_observe_pass_visits_a_session_host() -> None:
    # MCPs board task 701151b4: the node the operator's sessions run on.
    _test_hooks.run = FakeRun([ok(""), ok(document(hostname="ziggurat"))])
    board = FakeQueue(
        [EMPTY_IDENTITIES, "observed 0 session(s) for linux:ziggurat (0 new row(s)) at t"]
    )
    _test_hooks.http_post = board

    registry = {
        "ziggurat": node("ziggurat", "session-host"),
        "hub": node("hub", "hub", observer=LedgerObserver.PCSESSION),
    }

    outcomes = observe.observe_pass(BOARD, observe.observable(registry), IDENTITY, EVIDENCE)

    assert [(o["node"], o["recorded"]) for o in outcomes] == [("ziggurat", True)]
    assert [a["machine"] for a in board.arguments] == ["linux:ziggurat", "linux:ziggurat"]


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
