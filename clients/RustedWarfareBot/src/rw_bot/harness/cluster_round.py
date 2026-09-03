"""One doctrine-search round, played by the cluster instead of the local fleet.

The search driver proposes candidates and scores rounds; what varies is who
plays the matches. The workstation fleet plays through the match-service
queue; this runner plays through HPC3, and it drives EXACTLY the canonical
tools an operator drives by hand -- freeze (``scripts.stage_payload``),
stage (``hpc3.cli.stage``), document (``scripts.campaign_doc``), converge
(``hpc3.cli.campaign``) -- because a second submission path would be the
parallel write path this workspace bans. Every child process goes through
the ``run_capture`` hook and every wait through ``sleep``, so a test scripts
the whole cluster conversation without a mock.

What a round leaves behind is the same thing the queue path leaves behind:
one scorecard per job under ``<sweeps_root>/<batch>/``, which is all the
margin scorer reads. The cluster's own copies stay on the cluster.

Failure is loud and typed, and the loudness discriminates. A COMMAND that
fails -- non-zero from the thing the command ran -- raises immediately
with the command and its output, because that is a defect surfacing and
retrying it would hide it. A TRANSPORT loss is different: ``ssh`` and
``scp`` exit 255 when the connection itself died, which says nothing about
the command they were carrying, and every cluster command in this chain is
idempotent by design (probes read, extraction recreates, staging is
digest-verified, convergence is the documented idempotent resubmit). Those
are retried up to a bounded consecutive budget with each drop reported --
because a search's drain loop issues hundreds of sequential probes over
hours, and a driver that dies on any single drop statistically cannot
finish its own runtime (vhsearch3 died twice in 30 minutes on 2026-09-02,
once at the round-0 pull and once at a drain probe, both on the
Cloudflare-websocket leg, with all cluster work intact both times). A
round that cannot reach its expected scorecard count within the
convergence budget still raises naming the gap; the one resubmission loop
is ``hpc3-campaign``'s own convergence, re-invoked only after the queue
drains with members still missing.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot import RwBotError
from rw_bot.harness import _test_hooks

#: How many converge-then-drain passes a round may spend before the gap is
#: an error. Each pass resubmits only what is missing, so the budget bounds
#: casualties, not throughput; the measured boot-casualty rate was 4 of 96
#: with shared-filesystem clones and 0 of 96 with node-local ones.
CONVERGE_PASSES = 6

#: The exit status OpenSSH's ssh and scp reserve for "the connection
#: failed", as opposed to any status the remote command returned. Both
#: 2026-09-02 websocket-leg drops surfaced as exactly this.
TRANSPORT_EXIT = 255

#: How many CONSECUTIVE transport losses one command may ride out before
#: the route itself is the finding. Resets on any success; five drops in a
#: row spaced a poll apart is an outage, not a blip, and pretending
#: otherwise would spend hours discovering what the error says in seconds.
TRANSPORT_BUDGET = 5


class ClusterRoundError(RwBotError):
    """A cluster round could not deliver its scorecards.

    Args:
        code: Stable machine-readable identifier -- ``RW-CROUND-001`` for a
            child command that failed, ``RW-CROUND-002`` for a round still
            short of its expected scorecards after the convergence budget,
            ``RW-CROUND-003`` for a transport route that stayed down
            through the whole retry budget.
        message: Human-readable description carrying the command or the gap.
    """


class ClusterRound:
    """Plays one round's jobs on HPC3 and files the scorecards locally.

    Attributes:
        config: The hpc3 workspace document, resolved for every tool call.
        host: SSH destination, the workspace's own host.
        cluster_root: The project's cluster directory, absolute.
        map_path: The map every job plays.
        difficulty: The AI difficulty every job plays at.
        fast_forward: The pace multiple every job runs under.
        scratch: Local directory for the round's frozen tree, archive and
            campaign document -- run artifacts, not repository content.
        sweeps_root: Where the round's scorecards land locally.
        jobs_dir: Repository directory the round's job file is written to.
            Must sit inside the payload freeze's ``sweeps`` source so the
            members read the same file the driver wrote.
        poll_seconds: Seconds between drain polls.
    """

    def __init__(
        self,
        *,
        config: str,
        host: str,
        cluster_root: str,
        map_path: str,
        difficulty: int,
        fast_forward: int,
        scratch: Path,
        sweeps_root: Path,
        jobs_dir: Path,
        poll_seconds: float,
    ) -> None:
        """Bind one round runner to its workspace and its experiment knobs.

        Args:
            config: Path of the hpc3 workspace document.
            host: SSH destination.
            cluster_root: The project's cluster directory, absolute.
            map_path: The map every job plays.
            difficulty: The AI difficulty every job plays at.
            fast_forward: The pace multiple every job runs under.
            scratch: Local directory for per-round run artifacts.
            sweeps_root: Where scorecards land locally.
            jobs_dir: Repository directory for the round's job file.
            poll_seconds: Seconds between drain polls.
        """
        self.config = config
        self.host = host
        self.cluster_root = cluster_root
        self.map_path = map_path
        self.difficulty = difficulty
        self.fast_forward = fast_forward
        self.scratch = scratch
        self.sweeps_root = sweeps_root
        self.jobs_dir = jobs_dir
        self.poll_seconds = poll_seconds

    def _capture(self, argv: Sequence[str]) -> tuple[str, ...]:
        """Run one child command, raising on failure.

        ``ssh`` and ``scp`` commands whose CONNECTION died (exit
        :data:`TRANSPORT_EXIT`) are re-run up to :data:`TRANSPORT_BUDGET`
        consecutive times, a poll interval apart, each drop reported --
        every cluster command this runner issues is idempotent by design,
        so re-carrying one after a dropped connection repeats no work and
        hides no defect. Any other non-zero status is the command itself
        failing and raises immediately.

        Args:
            argv: Argument vector, program first.

        Returns:
            The command's output lines.

        Raises:
            ClusterRoundError: With ``RW-CROUND-001`` when the command exits
                non-zero, carrying the command and everything it printed --
                the child's own words are the diagnostic, and swallowing
                them would leave a code with nothing behind it. With
                ``RW-CROUND-003`` when an ssh/scp route stays down through
                the whole transport budget.
        """
        transported = argv[0] in ("ssh", "scp")
        drops = 0
        while True:
            status, lines = _test_hooks.run_capture(argv)
            if status == 0:
                return lines
            printed = "\n".join(lines)
            if not (transported and status == TRANSPORT_EXIT):
                raise ClusterRoundError(
                    "RW-CROUND-001",
                    f"command failed ({status}): {' '.join(argv)}\n{printed}",
                )
            drops += 1
            if drops >= TRANSPORT_BUDGET:
                raise ClusterRoundError(
                    "RW-CROUND-003",
                    f"transport down through {TRANSPORT_BUDGET} consecutive "
                    f"attempts: {' '.join(argv)}\n{printed}",
                )
            _test_hooks.write_line(
                f"# transport dropped ({drops}/{TRANSPORT_BUDGET}), retrying: {' '.join(argv)}"
            )
            _test_hooks.sleep(self.poll_seconds)

    def _remote(self, command: str) -> tuple[str, ...]:
        """Run one command on the cluster.

        Args:
            command: The shell command.

        Returns:
            Its output lines.

        Raises:
            ClusterRoundError: Through :meth:`_capture`.
        """
        return self._capture(["ssh", "-o", "BatchMode=yes", self.host, command])

    def _freeze_and_stage(self, batch: str) -> None:
        """Freeze the working tree into a payload and stage it as this batch's.

        The round's variant doctrines and job file are already in the
        repository directories the freeze copies, so the tree the members
        import is the tree the driver just wrote.

        Args:
            batch: The round's batch name; the cluster payload directory is
                ``payload-<batch>``.

        Raises:
            ClusterRoundError: Through :meth:`_capture`, on any failing step.
        """
        commit = self._capture(["git", "rev-parse", "HEAD"])[-1].strip()
        tree = self.scratch / batch / "rw-payload"
        # The archive carries the BATCH'S OWN NAME. Two drivers staging
        # concurrently under one shared name raced on 2026-09-03:
        # impsearch1's tar landed between vhsearch4's upload and verify,
        # and the stage's digest check refused the substituted bytes --
        # loudly, correctly, and fatally for the round. Distinct names
        # make the race impossible instead of merely detected.
        archive = self.scratch / batch / f"rw-payload-{batch}.tar"
        self._capture(
            [
                sys.executable,
                "-m",
                "scripts.stage_payload",
                "--tree",
                str(tree),
                "--commit",
                commit,
                "--archive",
                str(archive),
                "--out",
                str(self.scratch / batch / "payload-tree.json"),
                "--digests",
                str(self.scratch / batch / "payload-tree-digests.txt"),
                "--manifest",
                str(self.scratch / batch / "stage-payload-tree.json"),
                "--destination",
                f"{self.cluster_root}/staging",
            ]
        )
        self._capture(
            [
                sys.executable,
                "-m",
                "hpc3.cli.stage",
                "--config",
                self.config,
                "--manifest",
                str(self.scratch / batch / "stage-payload-tree.json"),
                "--source-dir",
                str(self.scratch / batch),
                "--expect-from",
                str(self.scratch / batch / "payload-tree-digests.txt"),
            ]
        )
        payload = f"{self.cluster_root}/payload-{batch}"
        self._remote(
            f"set -e; rm -rf {payload}; mkdir -p {payload}; "
            f"tar -xf {self.cluster_root}/staging/rw-payload-{batch}.tar -C {payload}"
        )

    def _write_document(self, batch: str, jobs_file: Path) -> Path:
        """Emit the round's campaign document.

        Args:
            batch: The round's batch name.
            jobs_file: The job file the members read, repository-relative.

        Returns:
            The document's path.

        Raises:
            ClusterRoundError: Through :meth:`_capture`.
        """
        document = self.scratch / batch / "campaign.json"
        self._capture(
            [
                sys.executable,
                "-m",
                "scripts.campaign_doc",
                "--config",
                self.config,
                "--jobs",
                jobs_file.as_posix(),
                "--batch",
                batch,
                "--map",
                self.map_path,
                "--difficulty",
                str(self.difficulty),
                "--fast-forward",
                str(self.fast_forward),
                "--payload",
                f"payload-{batch}",
                "--out",
                str(document),
            ]
        )
        return document

    def _filed(self, batch: str) -> int:
        """Count the scorecards the cluster has filed for this batch.

        Args:
            batch: The round's batch name.

        Returns:
            How many scorecards exist.
        """
        lines = self._remote(
            f'ls {self.cluster_root}/runs/sweeps/{batch} 2>/dev/null | grep -c "\\.txt$" || echo 0'
        )
        return int(lines[-1].strip())

    def _in_queue(self, batch: str) -> int:
        """Count this batch's jobs still queued or running.

        End-anchored on the QUALIFIED name, because an array's tasks all
        carry the sweep's own name -- ``rusted.vhsearch2-r0``, no member
        suffix. The first drain matched ``<batch>-`` expecting per-member
        names, saw zero rows while 136 tasks ran, and burned every
        convergence pass in seconds (vhsearch2-r0, 2026-09-02). The anchor
        also keeps ``-r1`` from counting ``-r10``.

        Args:
            batch: The round's batch name.

        Returns:
            How many jobs the cluster still holds.
        """
        lines = self._remote(f'squeue --me -h -o "%j" | grep -c "\\.{batch}$" || echo 0')
        return int(lines[-1].strip())

    def run(self, batch: str, job_lines: Sequence[str]) -> None:
        """Play one round to completion on the cluster.

        Args:
            batch: The round's batch name.
            job_lines: The round's job file content, one member per line
                plus comments -- written verbatim, frozen into the payload,
                and read back by every member.

        Raises:
            ClusterRoundError: With ``RW-CROUND-001`` on a failed command,
                or ``RW-CROUND-002`` when the round is still short of its
                scorecards after :data:`CONVERGE_PASSES` convergence passes
                -- at that point the casualties are not the measured
                boot-contention shape, and resubmitting harder would spend
                the cluster hiding a real defect.
        """
        expected = sum(1 for line in job_lines if line.strip() != "" and not line.startswith("#"))
        jobs_file = self.jobs_dir / f"{batch}.txt"
        _test_hooks.make_dirs(self.jobs_dir)
        _test_hooks.write_text_lines(jobs_file, list(job_lines))

        self._freeze_and_stage(batch)
        document = self._write_document(batch, jobs_file)

        filed = 0
        for _ in range(CONVERGE_PASSES):
            converge = self._capture(
                [
                    sys.executable,
                    "-m",
                    "hpc3.cli.campaign",
                    "--config",
                    self.config,
                    "--run",
                    str(document),
                ]
            )
            _test_hooks.write_line(f"# {batch}: {converge[-1]}")
            while True:
                filed = self._filed(batch)
                if filed >= expected:
                    break
                if self._in_queue(batch) == 0:
                    # Drained short: casualties. The next convergence pass
                    # resubmits exactly the missing members.
                    break
                _test_hooks.sleep(self.poll_seconds)
            if filed >= expected:
                break
        if filed < expected:
            raise ClusterRoundError(
                "RW-CROUND-002",
                f"{batch}: {filed} of {expected} scorecards after "
                f"{CONVERGE_PASSES} convergence passes; the missing members "
                "are not converging and resubmitting harder would spend the "
                "cluster hiding a real defect.",
            )

        local = self.sweeps_root / batch
        _test_hooks.make_dirs(local)
        self._capture(
            [
                "scp",
                "-q",
                f"{self.host}:{self.cluster_root}/runs/sweeps/{batch}/*.txt",
                str(local),
            ]
        )


__all__ = [
    "CONVERGE_PASSES",
    "TRANSPORT_BUDGET",
    "TRANSPORT_EXIT",
    "ClusterRound",
    "ClusterRoundError",
]
