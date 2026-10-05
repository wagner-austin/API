"""CLI: one serve of a node's runner on the queue's node lane.

Usage:
    fleet-node-agent --config fleet.json --node sedona
    fleet-node-agent --config fleet.json --node sedona --announce
    fleet-node-agent --config fleet.json --node serendipity --elevated

A run without ``--announce`` SERVES the node until it hands over before a
fire boundary (:mod:`fleet.cli.node_serve`, MCPs board task 8993c306),
reading ``refs/fleet/rolled`` from the git checkout its records directory
is in when it starts and at each boundary, so it hands over once a roll
moves the ref (:func:`fleet.core.rolled.rolled_state`).

ONE RUNNER PER ENABLED NODE, ALL OF THEM ON THE HUB (MCPs board task
fd5cabfa, A1 and A5). Until this command the queue had one runner, the hub's
``fleet-agent``, claiming the oldest job once per tick whatever it was, so
nine nodes behaved as one slow node and a revive waited behind a queue of
kills. Now each enabled node in ``fleet.json`` has a scheduled task of its own
(``scripts/register-node-agents.ps1``) running this command with ``--node``
set to it, claiming from the queue's NODE lane the jobs that name it or name
no node, and only those whose required tags the node carries
(:func:`fleet.contracts.tags.node_tags`, plus the tag of every tool its
toolchain probe found this tick, :mod:`fleet.cli.node_ready`). The hub's
``fleet-agent`` keeps the
HUB lane, so the two never hold each other's work. The runners live on the
hub rather than on the nodes because the hub holds the git credentials and
the ssh keys and the tailnet policy lets nothing else reach it.

THE NODE IS PROBED BEFORE ANYTHING IS CLAIMED. A tick asks its node what it
has free first, and a node that does not answer or has room for nothing
(:func:`fleet.core.capacity.room_for_any`) claims nothing, so the job stays
in the lane for a node that can run it; the first tick of these runners
(2026-09-21T10:00:02Z) had a sleeping node take the oldest job and refuse
it while a live one found the lane empty. It then asks the node's build
toolchain (:func:`fleet.core.toolchain.readiness_gap`, MCPs board task
bad56f65), and a node missing a tool or on the wrong Python claims nothing
either, naming the tool and the command that would install it there.

WHAT A CLAIMED JOB BECOMES. The job names a project and a commit. The runner
re-checks the job's tags against the registry's declaration for the project
and judges the project's fit on the probe already taken, and hands the rest
to a launch thread beside the next claim (:mod:`fleet.cli.node_launch`, MCPs
board task 8993c306): it fetches the commit from the project's declared
remote into a bare mirror (:mod:`fleet.core.export`) BEFORE any lease is
taken, so a sha the remote has never seen is refused with nothing held, then
takes the project's lease on this node, stages ``git archive`` of the commit
through the same verified transport every dispatch uses, sends the build
script with the project's install steps and the node's caches, launches it
detached, and reports the run started. The
serve's watch, or a later serve's collect pass, collects it
(:mod:`fleet.cli.node_watch`, :mod:`fleet.cli.node_collect`): reads the result,
reads the tail of the transcript, composes the verdict
(:mod:`fleet.core.verdict`), posts it to the submitter's feed when the job
names no task (the queue's close posts a task's row, MCPs board task
2fecad69), and closes the job on both sides; or stops the build, when
it runs past its lease or its queue job was cancelled under it.

THE IDENTITY IS DERIVED, NOT CONFIGURED. The label is ``fleet-node-<alias>``
and the session id is the version-5 UUID of that label, so every tick of one
node's runner is the same session on the board's ledger and ``held_by`` sees
its own claims across ticks; a fresh UUID per tick would make every tick a
stranger to the last. ``--announce`` posts the check-in that registers that
session on the ledger (MCPs mig 530), once, from the registration script.

A NODE THAT DECLARES ``elevated`` HAS A SECOND RUNNER (MCPs board task
a98d7083), this command with ``--elevated``: a separate identity,
``fleet-node-<alias>-elevated``, so it collects and stops only what it
claimed; the ``elevated`` tag, which the queue's exclusive rule makes the
only jobs it takes and the one kind the ordinary runner never gets
(:func:`fleet.contracts.tags.runner_tags`); and one more gate before the
claim, the ssh session's token as this tick's probe measured it
(:func:`fleet.contracts.elevation.elevation_gap`), because every build it
launches registers at RunLevel Highest. The two share the node's capacity,
so the ORDINARY runner claims nothing while an elevated job waits for the
node (:mod:`fleet.core.elevated_yield`); it would take every freed slot.

Exits 0 whenever the agent itself worked, refused jobs and failed suites
included, for the reason :mod:`fleet.cli.agent` gives: the status is whether
THE AGENT worked, and a loop that stopped on a red build would stop on the
one condition it exists to keep reporting.
"""

from __future__ import annotations

import functools
import sys
import uuid
from collections.abc import Sequence

from board_watch import config as board_config
from platform_core import cli_args
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config, node_serve, node_watch
from fleet.cli.node_claim import ask_queue, refuse
from fleet.cli.node_collect import collect_pass, require_sha
from fleet.cli.node_launch import Launcher
from fleet.cli.node_prepare import admit
from fleet.cli.node_ready import ready_state
from fleet.contracts.node import NodeConfig
from fleet.contracts.tags import runner_tags
from fleet.contracts.workspace import require_node
from fleet.core import _test_hooks, names, queue, rolled, tick_report

_log = get_logger(__name__)

NODE_FLAG = "--node"
ELEVATED_FLAG = "--elevated"

_FLAGS = (_config.CONFIG_FLAG, _config.RECORDS_FLAG, NODE_FLAG)

#: The namespace the runner's session UUID is derived in.
IDENTITY_NAMESPACE = uuid.NAMESPACE_URL


def node_identity(alias: str, *, elevated: bool) -> tuple[str, str]:
    """The label and session id a node's runner acts as.

    Args:
        alias: The node's workspace name.
        elevated: Whether this is the node's elevated runner.

    Returns:
        ``fleet-node-<alias>`` and the version-5 UUID of
        ``fleet-node-agent/<alias>`` in :data:`IDENTITY_NAMESPACE`, lowercase
        hyphenated as the board wants it; for the elevated runner
        ``fleet-node-<alias>-elevated`` and the UUID of
        ``fleet-node-agent/<alias>/elevated``, a session of its own, since
        each runner collects and stops only the jobs its label holds.
    """
    label = f"fleet-node-{names.runner_name(alias, elevated=elevated)}"
    if elevated:
        return label, str(uuid.uuid5(IDENTITY_NAMESPACE, f"fleet-node-agent/{alias}/elevated"))
    return label, str(uuid.uuid5(IDENTITY_NAMESPACE, f"fleet-node-agent/{alias}"))


def claim_pass(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    identity: JSONObject,
    *,
    alias: str,
    node: NodeConfig,
    elevated: bool,
    launcher: Launcher,
) -> str | None:
    """Take one job for this node and hand its launch off, or report why it could not.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        alias: This node's workspace name.
        node: Its declaration.
        elevated: Whether this is the node's elevated runner, which claims
            with the ``elevated`` tag and only once the probe has read an
            administrator's token for its ssh session.
        launcher: The serve's launches, whose grants and projects the gate
            charges to the node and which the claimed job's launch joins
            (:mod:`fleet.cli.node_launch`, MCPs board task 8993c306).

    THE NODE IS ASKED BEFORE THE QUEUE IS. A runner that claimed first and
    probed second took the oldest job off the lane and refused it while a
    live node beside it found nothing (:func:`fleet.core.capacity.room_for_any`
    carries the measurement), so a node that does not answer, or has room
    for nothing, claims nothing this tick and the job stays for one that
    can run it. The same holds for its TOOLCHAIN, asked before the room
    (:func:`fleet.core.toolchain.attempt_toolchain`, over the ssh account
    whose SID the build's scheduled task is registered for): lavender
    claimed slime jobs d515d038 and 7235d4c4, staged a whole export and ran
    npm ci before dying at the Makefile's first python call on the Store
    alias stub (MCPs board task e62c8120), so a node missing a tool now logs
    the code, the tool and this node's install command and claims nothing.
    A TAGGED tool it lacks closes nothing: the runner claims without that
    tool's tag (:func:`fleet.cli.node_ready.ready_state`), so the queue keeps
    the jobs that need it for another node (MCPs board task 939ec5c7).

    Returns:
        The job id of the job claimed and handed to its launch; None when
        this node could take nothing, nothing in the lane matched it, or
        the job it claimed was refused, here or by a launch that has
        already finished, which :func:`fill_pass` reads as the end of the
        pass.

    Raises:
        AppError: Only from the queue calls themselves, or a launch that has
            already finished raising one. A LOCAL refusal (an unknown
            project, tags that disagree, a project with no remote, a sha the
            remote lacks, too little capacity for the project, a held
            lease) is reported to the queue as ``refused`` with its code and
            message verbatim and does not propagate: transport, not
            recovery.
    """
    launching = launcher.launching()
    gate = ready_state(
        loaded,
        alias=alias,
        node=node,
        elevated=elevated,
        launching=launching["load"],
        launching_projects=launching["projects"],
    )
    ready = gate["ready"]
    if ready is None:
        tick_report.record_tick(credentials, gate["tick"], identity=identity)
        return None
    tick = gate["tick"]
    job = ask_queue(
        loaded, credentials, identity, tick, ready, alias=alias, node=node, elevated=elevated
    )
    if job is None:
        return None
    sha = require_sha(job)
    try:
        admitted = admit(loaded, job, node=node, ready=ready)
    except AppError as refusal:
        refuse(credentials, job, identity, detail=f"{refusal.code}: {refusal.message}")
        return None
    if isinstance(admitted, str):
        refuse(credentials, job, identity, detail=admitted)
        return None
    launch = launcher.start(job, admitted=admitted, sha=sha)
    if launch.done() and launch.result() is None:
        return None
    return job["job_id"]


def fill_pass(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    identity: JSONObject,
    *,
    alias: str,
    node: NodeConfig,
    elevated: bool,
    launcher: Launcher,
) -> tuple[str, ...]:
    """Claim and launch until this node has no room or the lane nothing it fits.

    ONE CLAIM PER TICK CAPPED THE FLEET BY THE CLOCK, NOT BY MEMORY (MCPs
    board task 48842bfd): once a node's room counted every live run
    (:func:`fleet.core.capacity.assess`), a node could hold several, but a
    runner launched at most one job per three-minute tick. With 14 checks
    queued on 2026-10-03, each node took exactly one per tick, each finished
    within about a tick, and lavender-wsl ran one job at a time with 18.6 GB
    free. So after every claim the runner claims again, and every pass
    re-runs the whole gate: :func:`fleet.cli.node_ready.ready_state` re-reads
    the ledger, which already charges every run launched, adds what the
    launches still under way were granted (:mod:`fleet.cli.node_launch`),
    re-probes the node, and leaves out the projects a lease or a launch
    holds. The claim does not wait for its launch (MCPs board task 8993c306).

    A CLAIM THAT LAUNCHES NOTHING ENDS THE PASS, a refusal included, once
    the claim knows it: a refusal the claim reports, or one its launch has
    already reported. A node that stops answering while one job is being
    staged fails the next claim's probe, which closes the gate.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        alias: This node's workspace name.
        node: Its declaration.
        elevated: Whether this is the node's elevated runner.
        launcher: The serve's launches, which hand each run to the serve's
            watch once its start is reported.

    Returns:
        The job ids this pass claimed and handed to a launch, in order.

    Raises:
        AppError: From :func:`claim_pass`, or from a launch of an earlier
            pass that raised (:meth:`fleet.cli.node_launch.Launcher.raise_failed`).
    """
    launcher.raise_failed()
    launched: list[str] = []
    while (
        job_id := claim_pass(
            loaded,
            credentials,
            identity,
            alias=alias,
            node=node,
            elevated=elevated,
            launcher=launcher,
        )
    ) is not None:
        launched.append(job_id)
    _log.info("%s launched %d job(s) this pass", alias, len(launched))
    return tuple(launched)


def main(argv: Sequence[str] | None = None) -> int:
    """Serve one node: collect what finished, fill its room, watch what it
    runs, and claim what arrives, until the serve hands over
    (:mod:`fleet.cli.node_serve`).

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 whenever the agent itself worked. See the module docstring.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            a required one is absent, or when ``--elevated`` names a node
            that declares no elevated runner.
        AppError: When the queue or the board cannot be reached or answered
            a shape this runner cannot read, when the node is not declared,
            or when this machine's records and the fleet disagree about a
            run.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    announce = queue.ANNOUNCE_FLAG in tokens
    elevated = ELEVATED_FLAG in tokens
    remaining = [token for token in tokens if token not in (queue.ANNOUNCE_FLAG, ELEVATED_FLAG)]
    parsed = cli_args.parse_single_flags(remaining, _FLAGS)
    loaded = _config.load_workspace(parsed)
    alias = cli_args.require_flag(parsed, NODE_FLAG)
    node = require_node(loaded.workspace, alias)
    tags = runner_tags(node, elevated=elevated)
    agent, session_id = node_identity(alias, elevated=elevated)
    identity = queue.identity_arguments(agent, session_id, str(loaded.directory))
    board = board_config.load_credentials()
    if announce:
        _log.info(
            "%s",
            queue.announce(
                board,
                machine=f"{sys.platform}:{_test_hooks.hostname()}",
                body=(
                    f"{agent} for {alias}: claims the queue's node lane for jobs "
                    f"naming {alias} or no node, carrying {', '.join(sorted(tags))}"
                ),
                identity=identity,
            ),
        )
        return 0
    credentials = queue.load_credentials()

    settle = functools.partial(
        node_watch.collect_ended, loaded, credentials, board, identity, agent=agent
    )
    watch = node_watch.RunWatch(loaded, alias=alias, node=node, settle=settle)
    launcher = Launcher(
        loaded, credentials, identity, alias=alias, node=node, agent=agent, hold=watch.hold
    )

    def collect() -> None:
        collect_pass(
            loaded,
            credentials,
            board,
            identity,
            agent=agent,
            alias=alias,
            hold=watch.hold,
            launching=launcher.launching()["jobs"],
        )

    def fill() -> None:
        fill_pass(
            loaded,
            credentials,
            identity,
            alias=alias,
            node=node,
            elevated=elevated,
            launcher=launcher,
        )

    def queued() -> frozenset[str]:
        return node_serve.queued_here(credentials, alias=alias)

    def rolled_now() -> str:
        return rolled.rolled_state(loaded.directory)

    with launcher:
        node_serve.serve(
            watch,
            alias=alias,
            started=_test_hooks.now(),
            serve_seconds=loaded.workspace["node_serve_seconds"],
            poll_seconds=loaded.workspace["node_poll_seconds"],
            steps=node_serve.ServeSteps(
                collect=collect, fill=fill, queued=queued, rolled=rolled_now
            ),
        )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-node-agent",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


# Without this, `python -m fleet.cli.node_agent` imports the module, runs
# nothing and exits 0, which reads as a tick that found nothing to do.
if __name__ == "__main__":
    entrypoint()


__all__ = [
    "ELEVATED_FLAG",
    "IDENTITY_NAMESPACE",
    "NODE_FLAG",
    "claim_pass",
    "entrypoint",
    "fill_pass",
    "main",
    "node_identity",
]
