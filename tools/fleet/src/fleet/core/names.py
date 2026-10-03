"""What a dispatch's directory on a node is called, file by file.

ONE SPELLING, SHARED BY EVERY DIALECT AND EVERY LIFECYCLE STEP. The build
writes the result file, the collector reads it, the stager lands the archive
and the digest script reads it; the dispatch registers a task or unit
by name and the cancel stops it by the same name. Each of those pairs used to
spell its name in two places, and the task name in particular was separately
spelled by the launch and the cancel -- one rename away from a cancel that
reported success having stopped nothing. These are the names, once.

Extensions are NOT here: a script is ``.ps1`` on one platform and ``.sh`` on
the other, and :meth:`fleet.core.dialect.Dialect.script_path` adds the right
one to a stem. The names below that are scripts are therefore stems.
"""

from __future__ import annotations

#: Where a dispatch's result is left on the node, under its own directory.
#:
#: The build's last act writes the recipe's exit status here, so the file's
#: ABSENCE means the run is still going and its presence means it is over,
#: with no third state to disambiguate. The build's transcript sits beside
#: it as ``<RESULT_NAME>.log``.
RESULT_NAME = "result.txt"

#: Where a Windows build records its own process id, under its directory.
#:
#: THE STOP NEEDS THE BUILD'S PROCESS, NOT ONLY ITS TASK. Stopping the
#: scheduled task ends the process the task started and nothing below it:
#: measured on sedona 2026-09-23, a task whose ``build.ps1`` ran a native
#: child read parent alive=False, child alive=True after
#: ``Stop-ScheduledTask``, which is how a cancelled slime run's vitest held
#: sedona for four hours. The build writes ``$PID`` here as its first act so
#: the stop can end that tree by id. A Linux unit needs no such file: stopping
#: it stops its whole control group.
PID_NAME = "build.pid"

#: The payload as scp lands it, under a dispatch's directory or a companion's
#: staging directory: an export's ``git archive``, or since MCPs board task
#: 2026dfbc a companion's ``git bundle``. One name for both because both
#: dialects' digest scripts read it; git reads a bundle by its contents, so
#: the ``.tgz`` spelling misleads a reader but no tool.
ARCHIVE_NAME = "tree.tgz"

#: The script that runs the suite, which is all it does.
#:
#: THE BUILD IS A SEPARATE FILE FROM THE THING THAT LAUNCHES IT, and that is
#: the fix for a real bug rather than a tidiness preference. The first version
#: passed the whole build as ``-Argument '-Command "cd ''{path}''; ..."'``, and
#: PowerShell ended the single-quoted argument at the first inner quote: the
#: registered task carried ``-Command "cd`` as its arguments and the remaining
#: two hundred characters as its WORKING DIRECTORY. Measured on sedona
#: 2026-09-04, the resulting task could not be started at all -- ``Element not
#: found``, because that working directory does not exist. Sending the script
#: and naming it by path is the same rule :mod:`fleet.core.remote` follows for
#: ssh, applied one layer further in; the launch then interpolates one path
#: and no code.
BUILD_STEM = "build"

#: The script that detaches the build from the connection and proves it began.
LAUNCH_STEM = "launch"

#: The script the collector runs to ask whether the build has finished.
COLLECT_STEM = "collect"

#: The script that prints the landed archive's digest.
DIGEST_STEM = "digest"

#: The script that unpacks a verified archive.
EXTRACT_STEM = "extract"

#: The script that makes the staged tree a git repository.
INIT_REPOSITORY_STEM = "init-repo"

#: The script that clones a staged companion from its bundle.
COMPANION_REPOSITORY_STEM = "companion-repo"

#: What a staging directory is called: the name of the directory it stages
#: and this suffix, beside it under the node's stage root.
#:
#: THE TRANSPORT FILES ARE KEPT OUT OF EVERY STAGED TREE, a dispatch's export
#: as well as a companion. Each is read as a repository: an export's claim is
#: that it is a commit and nothing else, and a companion's that it is a clone
#: of its ref. A dispatch's archive and step scripts once landed inside
#: its export and ``git add --all`` committed them, so MCPs' maketools
#: host-code on hardware-wiki read ``digest.sh``, ``extract.sh`` and
#: ``init-repo.sh`` as tracked host scripts the repository does not have
#: (FLEET-CHECK 8c1abde3, MCPs board task a8ee9b21).
STAGE_SUFFIX = ".stage"

#: The install script, under a node's stage root.
INSTALL_STEM = "fleet-install"

#: The script the collector runs to read the tail of a build's transcript,
#: the lines the verdict's counts are parsed from (MCPs board task fd5cabfa,
#: A3).
LOG_TAIL_STEM = "log-tail"

#: The directory under a node's stage root that holds the package managers'
#: caches for every export run on that node: npm's, poetry's and
#: playwright's, pointed at by the build script's environment. A clean export
#: has no dependencies of its own; these are where they are restored from,
#: and they outlive every run directory beside them. Never a run id: run ids
#: are ``<project>-<node>-<epoch>`` and the project grammar cannot spell this
#: name alone.
CACHE_DIRECTORY = "cache"

#: The directory under a node's stage root that keeps every retired run's
#: transcript, as ``<run_id>.log`` (MCPs board task bfca20e6). A run's own
#: directory is removed once its runner has settled or stopped it, and the
#: verdict line names the transcript's path here, so a reader following that
#: line still finds it. Kept because it is small and the export is not:
#: measured on diphtheria 2026-09-27, 126 transcripts came to 29 MB while the
#: exports beside them came to 293 GB. Never a run id, for the reason
#: :data:`CACHE_DIRECTORY` gives.
LOGS_DIRECTORY = "logs"


def cache_root(stage_root: str) -> str:
    """Where a node keeps the dependency caches its export runs share.

    Args:
        stage_root: The node's declared stage root.

    Returns:
        ``<stage_root>/cache``.
    """
    return f"{stage_root}/{CACHE_DIRECTORY}"


def dispatch_directory(stage_root: str, run_id: str) -> str:
    """Where one dispatch's export, scripts and result live on its node.

    Args:
        stage_root: The node's declared stage root.
        run_id: The dispatch.

    Returns:
        ``<stage_root>/<run_id>``.
    """
    return f"{stage_root}/{run_id}"


def log_path(target: str) -> str:
    """Where a build's transcript is written under its dispatch directory.

    Args:
        target: The dispatch's absolute remote directory.

    Returns:
        ``<target>/<RESULT_NAME>.log``, the one spelling the build writes
        and the collector reads.
    """
    return f"{target}/{RESULT_NAME}.log"


def retained_log_path(stage_root: str, run_id: str) -> str:
    """Where a retired run's transcript is kept on its node.

    Args:
        stage_root: The node's declared stage root.
        run_id: The dispatch.

    Returns:
        ``<stage_root>/<LOGS_DIRECTORY>/<run_id>.log``.
    """
    return f"{stage_root}/{LOGS_DIRECTORY}/{run_id}.log"


def companion_directory(stage_root: str, directory: str) -> str:
    """Where one companion's repository lands on a node.

    BESIDE THE EXPORTS, NOT INSIDE ONE, and never named after a run. A
    recipe reaches its companion as ``../<directory>`` from its own root,
    which is where a workstation keeps the same checkout, so the project's
    Makefile names one path and neither end has to know which machine it is
    on. The cost is that a companion is shared by every run on that node and
    replaced by each of them, rather than copied per run: nothing sweeps a
    stage root, and a per-run copy of a workspace this size would spend a
    node's whole disk budget in a handful of runs.

    Args:
        stage_root: The node's declared stage root.
        directory: The companion's declared directory name, one segment.

    Returns:
        ``<stage_root>/<directory>``.
    """
    return f"{stage_root}/{directory}"


def stage_name(name: str) -> str:
    """What the staging directory of one directory under the stage root is called.

    The string the script that makes it is named after, as
    :func:`make_directory_stem` names every directory maker.

    Args:
        name: The staged directory's name under the stage root: a run id, or
            a companion's declared directory name.

    Returns:
        ``<name>.stage``.
    """
    return f"{name}{STAGE_SUFFIX}"


def staging_directory(directory: str) -> str:
    """Where a staged directory's archive and scripts land, beside it.

    Args:
        directory: The absolute remote directory being staged: a dispatch's
            export or a companion.

    Returns:
        ``<directory>.stage``, beside it under the same stage root and never
        inside it.
    """
    return f"{directory}{STAGE_SUFFIX}"


def reset_directory_stem(directory: str) -> str:
    """Name the script that replaces one companion's directory.

    It lives under the stage ROOT, like the script that makes a dispatch's
    directory, because the directory it replaces is what it is about to
    delete.

    Args:
        directory: The companion's declared directory name.

    Returns:
        The stem.
    """
    return f"reset-{directory}"


def recipe_directory(target: str, path: str) -> str:
    """Where a project's recipe runs inside its export.

    Args:
        target: The dispatch's absolute remote directory, the export root.
        path: The project's directory inside its repository, ``""`` for a
            project that is the repository.

    Returns:
        ``<target>/<path>``, or ``target`` itself for the root.
    """
    return target if path == "" else f"{target}/{path}"


def task_name(run_id: str) -> str:
    """Name the scheduled task or transient unit a dispatch owns.

    Derived from the run id rather than from the stage path, so the name a
    dispatch registers and the name the cancel stops are the same string
    produced by the same function.

    Args:
        run_id: The dispatch.

    Returns:
        The name, valid as a Task Scheduler task name and as a systemd unit
        name (a run id is a project path with its slashes replaced, a node
        name and an epoch second, so it carries only word characters and
        hyphens).
    """
    return f"fleet-{run_id}"


def make_directory_stem(name: str) -> str:
    """Name the script that creates one staged directory.

    It lives under the stage ROOT rather than the directory it creates,
    which does not exist until it has run; named after that directory so two
    stagings at once cannot overwrite each other's script.

    Args:
        name: The directory's name under the stage root: a run id for a
            dispatch's export, a companion's staging directory otherwise.

    Returns:
        The stem.
    """
    return f"mkdir-{name}"


def runner_name(alias: str, *, elevated: bool) -> str:
    """Name one of a node's runners: the node itself, or its elevated lane.

    The name a runner's tick log, its scheduled task's stem and its probe
    scripts carry, so the two runners of a node that declares ``elevated``
    never share one of them.

    Args:
        alias: The node's workspace name.
        elevated: Whether this is the node's elevated runner.

    Returns:
        ``<alias>`` or ``<alias>-elevated``.
    """
    return f"{alias}-elevated" if elevated else alias


def capacity_probe_stem(writer: str) -> str:
    """Name the capacity probe one writer sends to a node's stage root.

    ONE PATH PER WRITER, never one per node. Every caller that asks a node
    what it has free writes the same constant script and runs it, and two
    callers asking the same node at once both wrote one file: on 2026-10-03
    serendipity's node and elevated runners, scheduled on the same minute,
    collided on it every tick (``The process cannot access the file ...
    because it is being used by another process``; ``Set-Content : Stream
    was not readable``), so neither read the node and it claimed nothing
    with 11.5 GB free (MCPs board task 939ec5c7).

    Args:
        writer: Who sends it: a runner's :func:`runner_name`, or the command
            that probes (``fleet-run``, ``fleet-nodes``, ``fleet-preflight``).

    Returns:
        The stem.
    """
    return f"fleet-capacity-{writer}"


def toolchain_probe_stem(writer: str) -> str:
    """Name the toolchain probe one writer sends to a node's stage root.

    One path per writer for the reason :func:`capacity_probe_stem` gives;
    the toolchain probe is the one serendipity's two runners were measured
    colliding on.

    Args:
        writer: Who sends it: a runner's :func:`runner_name`, or the command
            that probes (``fleet-bootstrap``).

    Returns:
        The stem.
    """
    return f"fleet-toolchain-{writer}"


def stop_stem(run_id: str) -> str:
    """Name the script that stops one dispatch.

    Args:
        run_id: The dispatch.

    Returns:
        The stem.
    """
    return f"stop-{run_id}"


def retire_stem(run_id: str) -> str:
    """Name the script that retires one dispatch's directory.

    It lives under the stage ROOT, like the stop script, because the
    directory it removes is the dispatch's own.

    Args:
        run_id: The dispatch.

    Returns:
        The stem.
    """
    return f"retire-{run_id}"


def root_script_stems(run_id: str) -> tuple[str, ...]:
    """Every script one dispatch leaves under the stage root.

    The makers of the export and of its staging directory, the stop and the
    retire itself: each is named after the run so two runs cannot overwrite
    each other's, and so none of them is inside a directory the retire
    removes. Measured on diphtheria 2026-09-27: 133 ``mkdir-`` scripts had
    outlived every tree they made.

    Args:
        run_id: The dispatch.

    Returns:
        Their stems, in the order the run writes them.
    """
    return (
        make_directory_stem(run_id),
        make_directory_stem(stage_name(run_id)),
        stop_stem(run_id),
        retire_stem(run_id),
    )


__all__ = [
    "ARCHIVE_NAME",
    "BUILD_STEM",
    "CACHE_DIRECTORY",
    "COLLECT_STEM",
    "COMPANION_REPOSITORY_STEM",
    "DIGEST_STEM",
    "EXTRACT_STEM",
    "INIT_REPOSITORY_STEM",
    "INSTALL_STEM",
    "LAUNCH_STEM",
    "LOGS_DIRECTORY",
    "LOG_TAIL_STEM",
    "PID_NAME",
    "RESULT_NAME",
    "STAGE_SUFFIX",
    "cache_root",
    "capacity_probe_stem",
    "companion_directory",
    "dispatch_directory",
    "log_path",
    "make_directory_stem",
    "recipe_directory",
    "reset_directory_stem",
    "retained_log_path",
    "retire_stem",
    "root_script_stems",
    "runner_name",
    "stage_name",
    "staging_directory",
    "stop_stem",
    "task_name",
    "toolchain_probe_stem",
]
