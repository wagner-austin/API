"""What the operator's TOOLS refuse, as opposed to what the services return.

SPLIT OUT OF :mod:`platform_core.error_codes` ON 2026-09-06, when that module
hit the same 600-line ceiling that had produced it two days earlier. The first
split separated VOCABULARY from MACHINERY; this one separates two vocabularies
that were never the same thing and had simply never been told apart.

THE BOUNDARY IS OBSERVABLE, NOT A JUDGEMENT CALL, and it holds in two
independent ways:

1. Every enum here belongs to a ``tools/*`` package -- the cluster submitter,
   the board reader, the two announce bridges, and the MCP transport all of
   them speak; the fleet dispatcher's, the largest, moved to
   :mod:`platform_core.error_codes_fleet` on 2026-09-29 under the same rule.
   Every enum left next door belongs to a service or to the platform itself.

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
    RUN_INPUT_MISSING = "RUN_INPUT_MISSING"
    RUN_INPUT_UNCERTIFIED = "RUN_INPUT_UNCERTIFIED"
    PARTITION_BILLS = "PARTITION_BILLS"
    PARTITION_GPU_MISMATCH = "PARTITION_GPU_MISMATCH"
    GPU_MODEL_EXHAUSTED = "GPU_MODEL_EXHAUSTED"
    PREEMPTIBLE_RUN_UNPROTECTED = "PREEMPTIBLE_RUN_UNPROTECTED"
    REPLAY_EXCEEDS_AFFORDABLE_LOSS = "REPLAY_EXCEEDS_AFFORDABLE_LOSS"
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
    # A manifest records the digest of the bytes ON DISK. When those differ
    # from the repository's copy of the same tracked path, the record is
    # accurate about what was staged and cannot be reproduced by anyone on
    # another checkout. Measured 2026-09-09 across every staging manifest in
    # tools/hpc3/runs: three files, all with Windows CRLF digests against LF
    # blobs. --expect-from cannot see it, because the manifest and the
    # expected-digest record are both written from the same working tree and
    # agree with each other by construction.
    STAGE_SOURCE_NOT_REPRODUCIBLE = "STAGE_SOURCE_NOT_REPRODUCIBLE"

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

    # Added 2026-09-07, when the server contract this package reads CHANGED.
    # Until ``edfa06ec`` a cursor came only on a FULL page, so a short page's
    # events were re-served on every poll forever; the fix makes every
    # NON-EMPTY page carry its last row's cursor. This package now walks on
    # that guarantee, which means a non-empty page arriving WITHOUT a cursor
    # is the server contradicting itself. Raising is deliberate: returning
    # early there would silently reinstate the original replay bug, and a
    # watcher that hides a regression reports silence -- indistinguishable
    # from a quiet board, which is the failure this package exists to remove.
    PAGE_WITHOUT_CURSOR = "PAGE_WITHOUT_CURSOR"


class CommitScopeErrorCode(ErrorCodeBase):
    """Checking that a commit carries only the paths its author declared.

    The git index is shared mutable state with no lock, and ``git commit``
    takes ALL of it -- so anything another session stages between your ``git
    add`` and your ``git commit`` ships under your message and your
    authorship. Measured twice in three hours on 2026-09-04/05: ``9d945451``
    swept another session's staged deletions, and ``09d2a04b`` did the same
    through ``--amend``.

    These codes cover only what can FAIL, which is a smaller set than what
    can be REFUSED. A commit carrying undeclared paths is a decision this
    package returns, not an exception it raises -- the caller is a git hook
    and its answer is an exit code. What raises is the environment being
    unable to answer the question at all.

    Same discipline as its siblings: no generic member. A code that covers
    everything identifies nothing.
    """

    # The two ways git itself cannot answer. Separate members because the
    # operator fixes them in different places: one is a broken or absent git,
    # the other is running the hook somewhere that is not a work tree.
    GIT_INDEX_UNREADABLE = "GIT_INDEX_UNREADABLE"
    GIT_REPO_ROOT_UNRESOLVED = "GIT_REPO_ROOT_UNRESOLVED"

    # The two ways a DECLARATION cannot mean anything, both of which would
    # otherwise fail open. git reports staged paths repo-relative with forward
    # slashes, so an absolute entry and an entry climbing out of the tree can
    # never match one -- and an entry that matches nothing silently widens
    # nothing while looking like protection. Refusing at decode is the only
    # point where the author is still present to fix it.
    SCOPE_ENTRY_NOT_RELATIVE = "SCOPE_ENTRY_NOT_RELATIVE"
    SCOPE_ENTRY_ESCAPES_REPO = "SCOPE_ENTRY_ESCAPES_REPO"


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


class CiWakeErrorCode(ErrorCodeBase):
    """Announcing GitHub Actions verdicts on the agent board.

    The third bridge, and the first whose upstream is neither a cluster nor
    this machine. It shares the board layer with the other two -- transport
    is :class:`McpClientErrorCode`'s, credentials
    :class:`BoardWatchErrorCode`'s, the standing task
    :class:`BoardBridgeErrorCode`'s -- so what is left here is the two things
    only a CI bridge touches: the ``gh`` boundary, and the enrolment record a
    ``pre-push`` hook writes.
    """

    # ``gh`` exited non-zero, or is not installed, or is not logged in. NOT
    # narrowed into three codes: the CLI reports all three on stderr and the
    # operator's fix is the same first step (`gh auth status`), so splitting
    # them would be a taxonomy the message already carries better.
    GH_COMMAND_FAILED = "GH_COMMAND_FAILED"

    # A push was enrolled under a value that cannot address anything --
    # a repository that is not ``owner/name``, or a sha that is not 40 hex
    # characters. Refused at ENROLMENT rather than at announcement time, so
    # the push that typed it hears about it instead of a bridge three
    # minutes later that cannot say whose push it was.
    ENROLMENT_FIELD_MALFORMED = "ENROLMENT_FIELD_MALFORMED"


class SessionLabelErrorCode(ErrorCodeBase):
    """Resolving the acting session's board label against the ledger.

    Shared by every enrolling tool that records who asked for work --
    ``hpc3`` at submit, ``ci-wake`` at ``pre-push`` -- through
    :mod:`platform_core.session_label`, so the two cannot describe the same
    refusal differently. Transport is :class:`McpClientErrorCode`'s; what is
    here is what the resolution itself can find wrong.
    """

    # ``BOARD_AGENT_LABEL`` names a label other than the one the board bound
    # to the acting session. The twin of the board's own
    # ``TASK_IDENTITY_MISMATCH``, raised at enrolment so the wrong name is
    # refused before a bridge posts under it (MCPs board task 3843d29f).
    LABEL_MISMATCH = "SESSION_LABEL_MISMATCH"

    # ``CLAUDE_CODE_SESSION_ID`` is set to something the harness could not
    # have written, so the board cannot be asked about it.
    SESSION_ID_MALFORMED = "SESSION_ID_MALFORMED"

    # The stack's ``.env`` carries no taskboard key or no operator tenant, so
    # the board cannot be asked at all. Named separately from
    # :class:`BoardWatchErrorCode`'s pair because the fix is in a different
    # place: that file, not the shell that runs a poller.
    CREDENTIALS_MISSING = "SESSION_LABEL_CREDENTIALS_MISSING"


class StackEndpointErrorCode(ErrorCodeBase):
    """Reading where the MCPs stack's services are reached from this machine.

    Shared by every tool here that calls the stack -- the session-label
    resolution and the board pollers -- through
    :mod:`platform_core.stack_endpoints`, the one reader of the MCPs
    repository's ``scripts/fleet/stack-endpoints.json``.
    """

    # The declaration carries no non-empty url for the service asked about,
    # so there is nowhere to call. The declaration is the one place the
    # stack's addresses are written since the hub stopped forwarding loopback
    # ports (MCPs board task c6fc4882), so no address is assumed instead.
    UNDECLARED = "STACK_ENDPOINT_UNDECLARED"


class MaketoolsErrorCode(ErrorCodeBase):
    """What the Makefile recipes' launcher (``tools/maketools``) refuses.

    The package runs under the SYSTEM interpreter before any venv exists,
    which is why its codes live here beside the other tooling enums rather
    than in a module of its own: ``platform_core.errors`` is the one error
    module the monorepo permits, and it is stdlib-importable by path.
    """

    # The command line a recipe passed is not one the launcher recognises,
    # or carries arguments the command does not take. Named separately from
    # a command's own refusals because the fix is in the Makefile, not in
    # the machine.
    USAGE = "MAKETOOLS_USAGE"

    # The process snapshot could not be taken or did not have the shape the
    # reaper reads (PowerShell's CIM query exited non-zero, or ``/proc/stat``
    # carried no ``btime``). The sweep that needed it does not run rather
    # than running over an empty table and reporting nothing stale.
    PROCESS_TABLE = "MAKETOOLS_PROCESS_TABLE"

    # A recipe's prerequisite executable (``poetry``, ``uv``) is not on the
    # PATH. Refused before the recipe runs a dozen lines that would each
    # fail with "not recognized", and the message carries the install hint
    # the Makefile used to print.
    TOOL_MISSING = "MAKETOOLS_TOOL_MISSING"

    # A first-party wheel or a virtual environment the recipe was told to
    # use is not where it was told it would be: no wheel under the crate's
    # ``target/wheels``, no ``.venv`` executable of the requested name.
    ARTIFACT_MISSING = "MAKETOOLS_ARTIFACT_MISSING"


__all__ = [
    "BoardBridgeErrorCode",
    "BoardWatchErrorCode",
    "CiWakeErrorCode",
    "Hpc3ErrorCode",
    "HpcWakeErrorCode",
    "MaketoolsErrorCode",
    "McpClientErrorCode",
    "SessionLabelErrorCode",
]
