"""What the fleet dispatcher refuses: :class:`FleetErrorCode`, alone.

SPLIT OUT OF :mod:`platform_core.error_codes_tooling` ON 2026-09-29, when a
new member (``NODE_STACK_MISMATCH``, MCPs board task 554bffc1) took that
module past the 600-line ceiling. The fleet's vocabulary was already the
largest there, about a third of the file, and it is the one that grows with
every capability a node can declare, so it is the one that leaves. The rule
:mod:`platform_core.error_codes_tooling` states still decides where a code
belongs; this module is that rule's fleet half.

``platform_core.errors`` re-exports the class as the package's public
surface, as it did before; the few modules that imported it from
``error_codes_tooling`` directly now name this module, and nothing is left
behind to redirect them.
"""

from __future__ import annotations

from platform_core.error_codes import ErrorCodeBase


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
    # The node lacks a tag the project requires (MCPs board task 41f45bd7):
    # its platform is the other dialect, or the project needs a CUDA device
    # and the node declares none. Distinct from the three capacity codes
    # because no amount of waiting changes it, and from NODE_DISABLED because
    # the node is on and fine -- it is the wrong kind of machine for this
    # suite, and the answer is another node, never this one later.
    NODE_LACKS_TAG = "NODE_LACKS_TAG"
    # The identity registry named by `fleet-nodes --registry` could not be
    # read. Drift itself has NO code, deliberately: it is reported as one
    # printed line per disagreement plus a non-zero exit, the same shape as an
    # unreachable node, because raising on the first disagreement would hide
    # the rest. An unreadable registry is the opposite case -- nothing was
    # compared at all, and reporting that as "no drift" would be a lie.
    NODE_REGISTRY_UNREADABLE = "NODE_REGISTRY_UNREADABLE"
    # The session ledger's fleet arm (MCPs board task 5a3865bf). The agent
    # reads a node's ~/.claude/sessions/<pid>.json records over ssh and hands
    # them to `task_session_observe`. A record that is not the shape the
    # harness writes is refused by name rather than recorded with a field
    # invented for it -- the ledger's whole value is that every row is a
    # fact somebody can trace to a file.
    SESSION_RECORD_UNREADABLE = "SESSION_RECORD_UNREADABLE"
    # A record that names its own machine (`pidDomain`, 2.1.270 onward) and
    # names a different one than the node it was read from. That is
    # corruption, not a remote session, and recording it under the node's
    # name would be the ledger's first lie -- the same refusal pcsession-mcp
    # makes for the hub's own directory.
    SESSION_MACHINE_MISMATCH = "SESSION_MACHINE_MISMATCH"

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
    # Node.js older than the projects' floor. Measured 2026-09-25: serendipity
    # carried v18.13.0 and was dispatched to, because node was judged on
    # presence alone.
    NODE_NODEJS_MISMATCH = "NODE_NODEJS_MISMATCH"
    # A node's elevated runner whose ssh session does not hold an
    # administrator's token: every build it launched at RunLevel Highest
    # would fail to register, so it claims nothing (MCPs board task
    # a98d7083).
    NODE_NOT_ELEVATED = "NODE_NOT_ELEVATED"

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

    # Exporting one commit of a project for the queue's node lane (MCPs board
    # task fd5cabfa, A2). Five codes, split where the reader's next action
    # splits: a project the registry declares without a remote is a
    # fleet.json line to write; a sha the remote does not have is a push the
    # submitter has not made; a fetch or archive that failed for any other
    # reason is git's own words, a network or credential question on the
    # hub; and a job whose tags disagree with the registry's declaration for
    # its project is a submitter that copied the wrong line, refused rather
    # than run under the wrong capability.
    PROJECT_REMOTE_MISSING = "PROJECT_REMOTE_MISSING"
    SHA_NOT_ON_REMOTE = "SHA_NOT_ON_REMOTE"
    EXPORT_FAILED = "EXPORT_FAILED"
    PROJECT_TAGS_MISMATCH = "PROJECT_TAGS_MISMATCH"
    # The fifth, added for the companion repositories a project's export
    # carries beside it (MCPs board task 0515040d): a declared ref the
    # companion's remote does not serve. Distinct from SHA_NOT_ON_REMOTE,
    # and the split is the same one the four above are split on. A sha the remote lacks is
    # the SUBMITTER's unpushed commit, one `git push` from checkable by the
    # session that submitted it; a ref the remote lacks was written into
    # fleet.json and is wrong for every job of that project until somebody
    # edits that line. Sending the second reader to look for their own
    # unpushed work would waste the one fact the refusal has.
    COMPANION_REF_NOT_ON_REMOTE = "COMPANION_REF_NOT_ON_REMOTE"
    # The sixth (MCPs board task 8454b6a9): an install step fleet.json
    # declares today names a path the exported commit does not contain,
    # because the step was declared after the commit was made. Measured
    # 2026-09-26: packages/db at a 2026-09-21 commit ran 'bash
    # scripts/testdb-setup.sh' and closed exit 127, which read as a broken
    # commit. It is neither the commit nor the submitter: today's registry
    # cannot build that commit, so it is refused before any lease.
    INSTALL_PATH_NOT_IN_COMMIT = "INSTALL_PATH_NOT_IN_COMMIT"

    # Dispatch and its record.
    DISPATCH_FAILED = "DISPATCH_FAILED"
    # Distinct from DISPATCH_FAILED, which is work that ran and exited
    # non-zero. This is work that never started: on 2026-09-04 a scheduled
    # task registered cleanly, `Start-ScheduledTask` failed with a
    # non-terminating error, PowerShell exited 0, and the ledger recorded a
    # run that did not exist.
    DISPATCH_NOT_LAUNCHED = "DISPATCH_NOT_LAUNCHED"
    # An unstarted claim two live runs could be; refused, never guessed (MCPs task 5a4f9b3e).
    DISPATCH_CLAIM_AMBIGUOUS = "DISPATCH_CLAIM_AMBIGUOUS"
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

    # The CI runner roster (`tools/fleet/runners.json`) -- which GitHub
    # Actions runner installs each machine carries, and the host assets their
    # job classes honestly require. Written 2026-09-08, the day that state
    # was still folklore: a keepalive scheduled task nobody had recorded took
    # six runners offline when a WSL restart missed it, and every host asset
    # (licensed game tree, llama.cpp checkout, /data) had been placed by hand.
    #
    # Same split as the registry codes above. An unreadable spec raises --
    # an audit that could not read its own roster has established nothing.
    # Drift itself carries NO code: it is one printed line per disagreement
    # plus a non-zero exit, because raising on the first would hide the rest.
    RUNNER_SPEC_UNREADABLE = "RUNNER_SPEC_UNREADABLE"
    # A host named on the command line that the roster does not declare.
    # Distinct from drift: the operator asked about a machine this document
    # has never heard of, and auditing nothing while exiting 0 would read as
    # a healthy host.
    RUNNER_HOST_UNKNOWN = "RUNNER_HOST_UNKNOWN"
    # The audit script's output did not parse. The script promises one
    # CHECK line per declared item; anything else on stdout means the
    # transport or the script is corrupt, and scoring a corrupt transcript
    # would let a mangled line read as a passing check.
    RUNNER_AUDIT_UNPARSABLE = "RUNNER_AUDIT_UNPARSABLE"
    # Onboarding a repository onto the self-hosted fleet (2026-09-09, the
    # operator's mandate after the last hand-rolled install). Two codes,
    # split where the reader's next action splits: an unmintable token is a
    # gh/auth/permissions question on the LOCAL machine, while an
    # already-onboarded repo means the roster says this work is done and
    # re-running it would register duplicate runners.
    RUNNER_TOKEN_UNAVAILABLE = "RUNNER_TOKEN_UNAVAILABLE"
    RUNNER_ALREADY_ONBOARDED = "RUNNER_ALREADY_ONBOARDED"


__all__ = ["FleetErrorCode"]
