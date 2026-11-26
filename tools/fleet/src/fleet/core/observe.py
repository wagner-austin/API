"""Observing the fleet's Claude Code sessions into the board's session ledger.

THE HUB CAN MOUNT ITS OWN SESSION RECORDS AND NOBODY ELSE'S. ``pcsession-mcp``
(repo MCPs) reads austinpc's ``~/.claude/sessions/<pid>.json`` through a
read-only bind mount and records every process there in ``agent_session_ledger``
(MCPs board task 5a3865bf). serendipity's directory is on serendipity, and no
container on the compose network has a route to it. This runner does: it is
the process on the hub that already reaches every node over ssh with the keys
the server deliberately does not hold, so it reads the same records there and
hands them to ``task_session_observe``, which records them through the same
canonical writer, keyed by the machine they came from.

THE SCRIPT IS SENT AND RUN BY PATH. Like every remote command in this
package (:mod:`fleet.core.remote` carries the measured reason) it is a file
the node runs by path, :data:`fleet.core.dialect.OBSERVE_SESSIONS_SCRIPT`.
Until board task cd5010c4 a PowerShell text went to every node, and
diphtheria answered ``powershell: command not found`` on every tick, logged
as a node that did not answer. The script emits ONE JSON document: the
node's platform and lowercased hostname (which together spell the ledger's
machine key, ``linux:diphtheria``), the
``pidDomain`` the node's harness writes, and every record in the directory,
verbatim. An absent directory is a node that has never run Claude Code,
reported as zero records rather than as a fault. Reading that document,
the machine key, and what each record is checked for, is
:mod:`fleet.core.observe_records`.

EVERY FAILURE OF ONE NODE IS THAT NODE'S OUTCOME (MCPs board task
56fee128). A node that is off, or refuses ssh, is an OUTCOME of the pass --
reported per node and the pass continues, exactly as
:func:`fleet.core.probe.attempt_probe` treats an unreachable node -- because
one laptop asleep must not blind the ledger to the rest of the fleet. A
record that cannot be read, or that names another machine's pidDomain, is
the same: decoding returns a :class:`RecordRefusal` VALUE carrying its code
and message, and the node's outcome is not observed, naming every refused
record. Until 2026-11-25 such a record raised, and one demure record
lacking ``messagingSocketPath`` ended every hub tick from 01:09Z: no later
node was observed and the reap pass never ran. Nothing is recorded around a
refused record -- its node gets no pass at all, because the readable records
alone would be derived as the node's whole set and depart the refused
session -- and the refusal itself goes to the board
(:func:`fleet.core.queue.observe_refusal`), which reads the node as not
observed with this reason and moves none of its sessions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypedDict

from platform_core.error_codes_fleet import FleetErrorCode
from platform_core.json_utils import JSONObject, JSONValue
from platform_core.mcp_client import McpCredentials

from fleet.contracts.machine import PLATFORM, LedgerObserver, Machine
from fleet.core import dialect, native_observe, native_wake, queue, remote
from fleet.core.observe_records import (
    RecordRefusal,
    decode_node_sessions,
    is_refusal,
    machine_of,
    observation_json,
    to_observation,
)

#: What the script is called on the node, before the dialect's extension,
#: under the provisioned account's ``.fleet`` directory -- the same
#: place-by-path convention the probe and dispatch scripts use, and a name
#: that cannot collide with either.
OBSERVE_STEM: Final = "observe-sessions"


class NodeOutcome(TypedDict):
    """What one node's pass came to.

    Attributes:
        node: The registry name.
        recorded: Whether the board accepted a pass for it. False means it
            did not: the node was not asked (no provisioned user), or it
            was asked and could not be observed -- ssh did not reach it, or
            it reported a record this pass cannot read -- and the board
            recorded that refusal in place of a pass.
        detail: The board's rendered line, for a pass or for a refusal, or
            why the node was not asked.
    """

    node: str
    recorded: bool
    detail: str


def script_path(user: str) -> str:
    """Where the script lands on a node.

    Args:
        user: The provisioned account, which spells its home.

    Returns:
        An absolute, literal path (``/home/corvis/.fleet/observe-sessions.sh``)
        -- the write command on the far side expands nothing, so this must
        be spelled out.
    """
    return dialect.script_path(dialect.fleet_directory(user), OBSERVE_STEM)


def roster_machine(node: Machine) -> str:
    """The machine key the roster spells for a node it could not read.

    Args:
        node: The machine, from the roster.

    Returns:
        ``linux:<name>``: the key the fleet audit holds the node's
        watermark to, used when the node's own document did not decode.
    """
    return f"{PLATFORM}:{node['name']}"


def refused_records_detail(refusals: list[RecordRefusal], total: int) -> str:
    """Name every record of one node this pass refused.

    Args:
        refusals: The refused records' refusals, in the node's order.
        total: How many records the node reported.

    Returns:
        ``refused <k> of <n> session record(s): <message>; <message>``.
    """
    messages = "; ".join(refusal["message"] for refusal in refusals)
    return f"refused {len(refusals)} of {total} session record(s): {messages}"


def _not_observed(
    board: McpCredentials,
    node: Machine,
    machine: str,
    code: FleetErrorCode,
    detail: str,
    identity: JSONObject,
) -> NodeOutcome:
    """Record on the board that a node could not be observed, and why.

    Args:
        board: The taskboard's endpoint and headers.
        node: The machine, from the roster.
        machine: Its ledger machine key.
        code: The refusal's code.
        detail: The refusal's sentence.
        identity: This runner's identity arguments.

    Returns:
        The node's outcome: not recorded, with the board's line for the
        refusal, which names the code, the reason and the pass that stands.

    Raises:
        AppError: Any transport or contract failure from the board call.
    """
    reply = queue.observe_refusal(
        board, machine=machine, code=code, detail=detail, identity=identity
    )
    return NodeOutcome(node=node["name"], recorded=False, detail=reply)


def observe_node(
    board: McpCredentials, node: Machine, identity: JSONObject, evidence_seconds: int
) -> NodeOutcome:
    """Read one node's records and record them on the board.

    Args:
        board: The taskboard's endpoint and headers.
        node: The machine, from the roster.
        identity: This runner's identity arguments.
        evidence_seconds: The workspace's ``native_evidence_seconds``, the
            lifetime of each native Codex observation
            (:func:`fleet.core.native_observe.observe_codex`).

    Returns:
        The outcome. Recorded; or not asked, for a node with no provisioned
        user; or not observed, with the refusal the board recorded in place
        of a pass: ssh's failure for a node it did not reach, or every
        record the node reported that this pass cannot read. A node with
        any refused record records no pass, so none of its sessions is
        departed by the refused one's absence.

    Raises:
        AppError: Any transport or contract failure from a board call.
            Never for the node: whatever it reported is its outcome.
    """
    user = node["user"]
    if user is None:
        return NodeOutcome(
            node=node["name"], recorded=False, detail="no provisioned user in the registry"
        )
    outcome = remote.attempt_script(
        node["name"], script_path(user), dialect.OBSERVE_SESSIONS_SCRIPT
    )
    failure = outcome["failure"]
    if failure is not None:
        return _not_observed(
            board, node, roster_machine(node), failure["code"], failure["message"], identity
        )
    sessions = decode_node_sessions(node["name"], outcome["output"])
    if is_refusal(sessions):
        return _not_observed(
            board, node, roster_machine(node), sessions["code"], sessions["message"], identity
        )
    machine = machine_of(sessions)
    observations: list[JSONValue] = []
    refusals: list[RecordRefusal] = []
    for record in sessions["records"]:
        read = to_observation(node["name"], sessions["pid_domain"], record)
        if is_refusal(read):
            refusals.append(read)
        else:
            observations.append(observation_json(read))
    if refusals:
        detail = refused_records_detail(refusals, len(sessions["records"]))
        return _not_observed(board, node, machine, refusals[0]["code"], detail, identity)
    native = native_observe.observe_codex(
        board, node["name"], user, machine, identity, evidence_seconds
    )
    detail = queue.observe_sessions(
        board,
        machine=machine,
        observations=observations,
        identity=identity,
        native_observations=native,
    )
    # After the ledger holds this pass's runtime reading, so the idle state
    # each wake acts on is the one the board already shows (MCPs board task
    # 562cfa7a).
    woken = native_wake.wake_idle(board, identity, node["name"], user, native)
    wakes = "".join(f"; {line}" for line in woken)
    return NodeOutcome(
        node=node["name"],
        recorded=True,
        detail=f"{native_observe.evidence_note(native, evidence_seconds)}{detail}{wakes}",
    )


def observable(machines: Mapping[str, Machine]) -> tuple[Machine, ...]:
    """The machines the roster says are live and whose sessions this pass records.

    The roster's ``ledger_observer`` decides, not the role: it is what the
    fleet audit holds each machine's watermark to. Until MCPs board task
    911b0303 the pass took workers, jump hosts and session hosts, and when
    diphtheria became a hub on 2026-10-18 while still declaring
    ``fleet-arm`` it dropped out of every pass.

    Args:
        machines: The roster's machines, by name.

    Returns:
        Each enabled machine that declares ``fleet-arm``, in name order.
        Disabled machines, a machine ``pcsession-mcp`` reads on its own
        mount and one nobody observes (a client) are left out: they are
        never asked, which is a different fact from not answering.
    """
    return tuple(
        machines[name]
        for name in sorted(machines)
        if machines[name]["enabled"]
        and machines[name]["ledger_observer"] is LedgerObserver.FLEET_ARM
    )


def observe_pass(
    board: McpCredentials,
    machines: tuple[Machine, ...],
    identity: JSONObject,
    evidence_seconds: int,
) -> tuple[NodeOutcome, ...]:
    """Observe each machine :func:`observable` chose.

    Args:
        board: The taskboard's endpoint and headers.
        machines: The machines to visit, in order.
        identity: This runner's identity arguments.
        evidence_seconds: See :func:`observe_node`.

    Returns:
        One outcome per machine, in order: one node that cannot be observed
        is that node's outcome, and every machine after it is still visited.

    Raises:
        AppError: See :func:`observe_node`.
    """
    return tuple(observe_node(board, machine, identity, evidence_seconds) for machine in machines)


def render_outcome(outcome: NodeOutcome) -> str:
    """One log line per node.

    Args:
        outcome: The node's outcome.

    Returns:
        ``<node>: <detail>``, prefixed ``not observed`` when no pass was
        recorded for it, so a refused or unasked node and a recorded one
        never read alike.
    """
    if outcome["recorded"]:
        return f"{outcome['node']}: {outcome['detail']}"
    return f"{outcome['node']}: not observed -- {outcome['detail']}"


__all__ = [
    "OBSERVE_STEM",
    "NodeOutcome",
    "observable",
    "observe_node",
    "observe_pass",
    "refused_records_detail",
    "render_outcome",
    "roster_machine",
    "script_path",
]
