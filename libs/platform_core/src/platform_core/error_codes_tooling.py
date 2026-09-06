"""What the operator's TOOLS refuse, as opposed to what the services return.

SPLIT OUT OF :mod:`platform_core.error_codes` ON 2026-09-06, when that module
hit the same 600-line ceiling that had produced it two days earlier. The first
split separated VOCABULARY from MACHINERY; this one separates two vocabularies
that were never the same thing and had simply never been told apart.

THE BOUNDARY IS OBSERVABLE, NOT A JUDGEMENT CALL, and it holds in two
independent ways:

1. Every enum here belongs to a ``tools/*`` package -- the cluster submitter,
   the fleet dispatcher, the board reader, the two announce bridges, and the
   MCP transport all five speak. Every enum left next door belongs to a
   service or to the platform itself.

2. NONE of these is ever rendered as an HTTP status. Look at
   :mod:`platform_core.errors`: it carries ``_ERROR_CODE_STATUS``,
   ``_HANDWRITING_STATUS`` and ``_MODEL_TRAINER_STATUS``, and no status map
   for ``Hpc3ErrorCode`` or ``FleetErrorCode`` -- because those are a
   command line's refusal to a person, not a response to a request. That
   asymmetry existed before this file did; the split only wrote it down.

So the rule for the next code, and it needs no taste: does a ``tools/*``
package raise it, and does it reach a human through a terminal rather than a
response body? Then it belongs here.

NOTHING WAS LEFT BEHIND TO REDIRECT OLD IMPORTS. ``platform_core.errors``
already re-exports ``Hpc3ErrorCode`` and ``FleetErrorCode`` and is the
package's public surface, so the ~100 files importing through it did not move.
The nineteen lines that imported these enums from ``error_codes`` directly now
name this module. A re-export that exists only so stale imports keep resolving
is not an interface, and this workspace does not keep one.
"""

from __future__ import annotations

from platform_core.error_codes import ErrorCodeBase


class Hpc3ErrorCode(ErrorCodeBase):
    """Slurm cluster submission and staging error codes.

    Each names one invariant of submitting work to HPC3. There is
    deliberately no generic member: a code that covers everything identifies
    nothing, and the first thing a caller does with one is re-parse the
    message string it was supposed to replace.

    These carry no meaningful HTTP status. They surface through a CLI, and
    ``_default_status_for`` returning 500 for them is correct in the sense
    that nothing consults it.
    """

    # Submission rules -- each maps to one refusal in decode_job_spec.
    PROJECT_UNIMAGED = "PROJECT_UNIMAGED"
    WHEEL_TAG_UNKNOWN = "WHEEL_TAG_UNKNOWN"
    GPU_TYPE_UNPINNED = "GPU_TYPE_UNPINNED"
    IMAGED_COMMAND_NEEDS_A_SHELL = "IMAGED_COMMAND_NEEDS_A_SHELL"
    RUN_REMOVES_IMAGE = "RUN_REMOVES_IMAGE"
    PARTITION_BILLS = "PARTITION_BILLS"
    PARTITION_GPU_MISMATCH = "PARTITION_GPU_MISMATCH"
    GPU_MODEL_EXHAUSTED = "GPU_MODEL_EXHAUSTED"
    PREEMPTIBLE_RUN_UNPROTECTED = "PREEMPTIBLE_RUN_UNPROTECTED"
    TIME_LIMIT_EXCEEDS_PARTITION = "TIME_LIMIT_EXCEEDS_PARTITION"

    # Sweeps -- many jobs from one template.
    SWEEP_EXCEEDS_GPU_CEILING = "SWEEP_EXCEEDS_GPU_CEILING"
    SWEEP_EXCEEDS_CPU_CEILING = "SWEEP_EXCEEDS_CPU_CEILING"
    SWEEP_EXCEEDS_JOB_CEILING = "SWEEP_EXCEEDS_JOB_CEILING"

    # Job arrays -- one sbatch call carrying a whole sweep. There is no
    # members-diverge code, deliberately: the array renderer takes the sweep
    # document itself, whose members share the template by construction, so
    # divergence is unrepresentable rather than checked.
    ARRAY_ID_UNPARSABLE = "ARRAY_ID_UNPARSABLE"
    ARRAY_INDICES_EMPTY = "ARRAY_INDICES_EMPTY"

    # Concurrency -- two jobs that would write one file.
    ARTIFACT_ALREADY_IN_FLIGHT = "ARTIFACT_ALREADY_IN_FLIGHT"

    # Campaigns -- a set of runs converging on a declared end state.
    CAMPAIGN_MEMBER_HAS_NO_ARTIFACT = "CAMPAIGN_MEMBER_HAS_NO_ARTIFACT"

    # Image builds -- the one job that is submitted from an already-rendered
    # script rather than from a run document.
    IMAGE_BUILD_SCRIPT_UNREADABLE = "IMAGE_BUILD_SCRIPT_UNREADABLE"
    IMAGE_BUILD_NAME_MISMATCH = "IMAGE_BUILD_NAME_MISMATCH"

    # Staging -- the bytes a run is entitled to read.
    DIGEST_MISMATCH = "DIGEST_MISMATCH"
    MANIFEST_FILE_MISSING = "MANIFEST_FILE_MISSING"
    STAGED_DIGEST_UNEXPECTED = "STAGED_DIGEST_UNEXPECTED"

    # Budget -- our own share of a shared machine, capped before and during.
    BUDGET_PROJECTION_EXCEEDED = "BUDGET_PROJECTION_EXCEEDED"
    BUDGET_CONSUMPTION_EXCEEDED = "BUDGET_CONSUMPTION_EXCEEDED"

    # Bootstrap -- creating the FIRST environment, which capture then probes.
    #
    # These refuse on the creating path, about what the command itself just
    # built, rather than gating a project that already runs. That distinction
    # is the point: every other code here can only ever fire at somebody who
    # has finished, which is how a system accumulates refusals and no
    # on-ramps.
    BOOTSTRAP_ENV_EXISTS = "BOOTSTRAP_ENV_EXISTS"
    BOOTSTRAP_PYTHON_MISMATCH = "BOOTSTRAP_PYTHON_MISMATCH"
    BOOTSTRAP_ENV_NOT_SELF_CONTAINED = "BOOTSTRAP_ENV_NOT_SELF_CONTAINED"

    # Preflight -- validating a job against the live scheduler before running it.
    PREFLIGHT_REJECTED = "PREFLIGHT_REJECTED"
    PREFLIGHT_UNPARSABLE = "PREFLIGHT_UNPARSABLE"
    ENV_PATH_MISSING = "ENV_PATH_MISSING"
    ENV_PACKAGE_MISMATCH = "ENV_PACKAGE_MISMATCH"
    ENV_PROBE_UNREADABLE = "ENV_PROBE_UNREADABLE"

    # Workspace configuration -- the one document every command reads.
    WORKSPACE_PROJECT_UNKNOWN = "WORKSPACE_PROJECT_UNKNOWN"
    RUN_FIELD_UNKNOWN = "RUN_FIELD_UNKNOWN"

    # Cluster selection -- which measured machine the rules come from.
    CLUSTER_UNKNOWN = "CLUSTER_UNKNOWN"
    PARTITION_UNKNOWN = "PARTITION_UNKNOWN"

    # Cluster interaction.
    REMOTE_COMMAND_FAILED = "REMOTE_COMMAND_FAILED"
    SACCT_FIELD_UNPARSABLE = "SACCT_FIELD_UNPARSABLE"


class FleetErrorCode(ErrorCodeBase):
    """Dispatching work to the machines on the tailnet.

    A sibling of :class:`Hpc3ErrorCode` rather than a section of it, because
    the two answer to different schedulers. HPC3 has Slurm: partitions, QOS
    ceilings, preemption, service units. A workstation on the tailnet has none
    of those and is bounded instead by the thing Slurm never has to think
    about -- somebody is sitting at it. Folding these into the Slurm enum
    would put codes there that can never fire on a cluster, and the first
    reader to meet ``NODE_OWNER_RESERVED`` in a Slurm traceback would have to
    work out that it cannot happen.

    Same discipline as its sibling: no generic member. A code that covers
    everything identifies nothing.
    """

    # Capacity -- the local analogue of GPU_MODEL_EXHAUSTED, and it exists for
    # the same measured reason. Admissible and going-to-finish are different
    # questions: on 2026-09-04 two overlapping suites on one box held 66
    # processes and 77.9 GB of commit while doing no work at all.
    NODE_MEMORY_EXHAUSTED = "NODE_MEMORY_EXHAUSTED"
    NODE_DISK_EXHAUSTED = "NODE_DISK_EXHAUSTED"
    NODE_OWNER_RESERVED = "NODE_OWNER_RESERVED"
    NODE_UNREACHABLE = "NODE_UNREACHABLE"
    # Distinct from NODE_UNREACHABLE, and the distinction is the whole point:
    # "was never asked" is not "did not answer". A node the workspace declares
    # disabled is deliberately off -- travelling, unprovisioned, retired --
    # and dispatching to it is a request that should never have left the
    # ground, not a fault to investigate on the tailnet.
    NODE_DISABLED = "NODE_DISABLED"
    # The identity registry named by `fleet-nodes --registry` could not be
    # read. Drift itself has NO code, deliberately: it is reported as one
    # printed line per disagreement plus a non-zero exit, the same shape as an
    # unreachable node, because raising on the first disagreement would hide
    # the rest. An unreadable registry is the opposite case -- nothing was
    # compared at all, and reporting that as "no drift" would be a lie.
    NODE_REGISTRY_UNREADABLE = "NODE_REGISTRY_UNREADABLE"

    # The lease -- one project's environment, mutated by one dispatch at a
    # time. This is the code that answers the incident the package was written
    # for: `poetry sync` reinstalling a package under a live interpreter.
    LEASE_HELD = "LEASE_HELD"
    LEASE_NOT_HELD = "LEASE_NOT_HELD"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    # Distinct from LEASE_HELD, because the two send a reader in opposite
    # directions. A held environment is per node and the answer is another
    # node; a held fleet-wide resource -- the one `corvis_test` every MCPs
    # database suite migrates -- has no second copy anywhere, so the only
    # answer is to wait. One code would send half its readers hunting for
    # capacity that could not have helped.
    RESOURCE_HELD = "RESOURCE_HELD"

    # Toolchain -- what a node must already have to be dispatched to.
    # Measured 2026-09-04: `make` was present on one node of three.
    NODE_TOOL_MISSING = "NODE_TOOL_MISSING"
    NODE_PYTHON_MISMATCH = "NODE_PYTHON_MISMATCH"

    # Staging -- the bytes the node is entitled to run.
    STAGE_DIGEST_MISMATCH = "STAGE_DIGEST_MISMATCH"
    STAGE_ARCHIVE_UNREADABLE = "STAGE_ARCHIVE_UNREADABLE"

    # The tree a dispatch has to carry. A project is not self-contained here:
    # its pyproject names path dependencies, and its Makefile calls a launcher
    # at the repository root. Measured 2026-09-04, the first dispatch that
    # reached a node staged the project alone and could not have resolved its
    # own lockfile.
    PROJECT_MANIFEST_MISSING = "PROJECT_MANIFEST_MISSING"
    PROJECT_MANIFEST_UNREADABLE = "PROJECT_MANIFEST_UNREADABLE"
    PROJECT_DEPENDENCY_ESCAPES_ROOT = "PROJECT_DEPENDENCY_ESCAPES_ROOT"

    # Dispatch and its record.
    DISPATCH_FAILED = "DISPATCH_FAILED"
    # Distinct from DISPATCH_FAILED, which is work that ran and exited
    # non-zero. This is work that never started: on 2026-09-04 a scheduled
    # task registered cleanly, `Start-ScheduledTask` failed with a
    # non-terminating error, PowerShell exited 0, and the ledger recorded a
    # run that did not exist.
    DISPATCH_NOT_LAUNCHED = "DISPATCH_NOT_LAUNCHED"
    RUN_RESULT_UNREADABLE = "RUN_RESULT_UNREADABLE"
    RUN_UNKNOWN = "RUN_UNKNOWN"
    LEDGER_ROW_UNPARSABLE = "LEDGER_ROW_UNPARSABLE"
    FEED_EVENT_UNPARSABLE = "FEED_EVENT_UNPARSABLE"

    # Workspace configuration -- the one document every command reads.
    WORKSPACE_NODE_UNKNOWN = "WORKSPACE_NODE_UNKNOWN"
    WORKSPACE_PROJECT_UNKNOWN = "WORKSPACE_PROJECT_UNKNOWN"

    # The corvis dispatch queue, which a runner on the hub claims work from.
    # Two codes, split where a reader's next action splits: a malformed
    # ANSWER means the tool's contract moved and the fix is in the MCP
    # package, while a queue job naming something this workspace does not
    # declare means the fleet.json here is behind and the fix is a config
    # line. One code would send half its readers to the wrong repository.
    QUEUE_ANSWER_MALFORMED = "QUEUE_ANSWER_MALFORMED"
    QUEUE_CREDENTIALS_MISSING = "QUEUE_CREDENTIALS_MISSING"


class McpClientErrorCode(ErrorCodeBase):
    """Calling one MCP tool over HTTP -- the transport, and nothing above it.

    Split out of :class:`BoardWatchErrorCode` on 2026-09-05, when a second
    package started calling MCP tools and the transport half of that enum
    turned out never to have been board-specific. These three describe the
    EXCHANGE: the endpoint refusing, the endpoint accepting and the tool
    failing, and the endpoint answering in a shape no MCP client can read.
    What a particular tool's ANSWER should look like stays with whoever parses
    it -- the board's rendered-prose grammar codes are still next door,
    because a fleet-dispatch caller reading JSON cannot hit them.

    Same discipline as its siblings: no generic member.
    """

    # HTTP_STATUS is the endpoint refusing (401 on a rotated key is the case
    # that actually happens); RPC_ERROR is the endpoint accepting and the tool
    # itself failing. Merging them would send a reader to the wrong layer.
    HTTP_STATUS = "HTTP_STATUS"
    RPC_ERROR = "RPC_ERROR"
    RESPONSE_NOT_EVENT_STREAM = "RESPONSE_NOT_EVENT_STREAM"


class BoardWatchErrorCode(ErrorCodeBase):
    """Subscribing a shell to the corvis agent board's change feed.

    A sibling of :class:`FleetErrorCode` rather than a section of it, because
    the two fail at different layers. Fleet's codes are about a machine having
    or not having a resource. These are about a RENDERED TEXT CONTRACT holding
    or not holding: ``task_events`` answers in prose built for a model to
    read, so every field this package needs is recovered by parsing rather
    than by reading a JSON key. When that parse fails the useful question is
    which part of the grammar moved, and a code per part is what answers it.

    Same discipline as its siblings: no generic member. A code that covers
    everything identifies nothing.
    """

    # Configuration -- what the shell must be given before it can call at all.
    # Separate members rather than one CONFIG_ERROR because the operator fixes
    # them in different places: one is a container's environment, the other is
    # a row in the tenants table.
    API_KEY_MISSING = "API_KEY_MISSING"
    TENANT_ID_MISSING = "TENANT_ID_MISSING"

    # Transport codes used to live here. They moved to
    # :class:`McpClientErrorCode` on 2026-09-05 with the client itself: an
    # endpoint refusing and a tool erroring are properties of MCP-over-HTTP,
    # not of this board, and a second caller needed the same three.

    # The rendered contract -- one code per element of the grammar, because
    # each is produced by a different function on the server and so can move
    # independently. Recorded 2026-09-05: a watcher scraping the cursor with a
    # regex for ``nextCursor`` never matched the real footer, which spells it
    # ``next cursor:``. It advanced no cursor and silently replayed the same
    # events forever, and a single MALFORMED_RESPONSE code would not have said
    # which half was wrong.
    EVENT_LINE_MALFORMED = "EVENT_LINE_MALFORMED"
    FOOTER_MISSING = "FOOTER_MISSING"
    FOOTER_MALFORMED = "FOOTER_MALFORMED"


class BoardBridgeErrorCode(ErrorCodeBase):
    """What ANY bridge announcing terminal work on the board can get wrong.

    Shared by ``tools/hpc-wake`` and ``tools/fleet-wake`` through
    :mod:`platform_core.board`, so the two cannot describe the same condition
    differently. A bridge's own domain failures stay in its own enum below --
    this holds only what the shared code raises.
    """

    # Configuration: the standing task announcements land in. Required, not
    # discovered -- an announcement posted to a guessed task is one nobody
    # subscribed to, which reads exactly like the bridge working.
    TASK_ID_MISSING = "TASK_ID_MISSING"


class HpcWakeErrorCode(ErrorCodeBase):
    """Announcing Slurm terminal states on the agent board.

    A sibling of :class:`BoardWatchErrorCode` for the same reason that is a
    sibling of :class:`FleetErrorCode`: the failures live at this package's
    own layer. Transport is :class:`McpClientErrorCode`'s, credentials are
    :class:`BoardWatchErrorCode`'s (the bridge loads them through the same
    reader), the cluster's are :class:`Hpc3ErrorCode`'s, and everything a
    bridge shares with the fleet bridge is :class:`BoardBridgeErrorCode`'s.
    What is left is what only this package can get wrong.
    """

    # A terminal accounting row for a job the ledger never recorded. The
    # query is BUILT from the ledger, so this is a contract violation --
    # answering it with a skip would hide whichever expansion bug produced
    # it behind a quiet cycle.
    JOB_UNKNOWN_TO_LEDGER = "JOB_UNKNOWN_TO_LEDGER"


__all__ = [
    "BoardBridgeErrorCode",
    "BoardWatchErrorCode",
    "FleetErrorCode",
    "Hpc3ErrorCode",
    "HpcWakeErrorCode",
    "McpClientErrorCode",
]
