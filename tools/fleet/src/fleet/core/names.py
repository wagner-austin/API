"""What a dispatch's directory on a node is called, file by file.

ONE SPELLING, SHARED BY EVERY DIALECT AND EVERY LIFECYCLE STEP. The build
writes the result file, the collector reads it, the stager writes the encoded
archive and the reassembler decodes it; the dispatch registers a task or unit
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

#: The reassembled archive, under a dispatch's directory.
ARCHIVE_NAME = "tree.tgz"

#: The archive as it crossed the transport: standard base64, one line.
ENCODED_NAME = "tree.b64"

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

#: The script that decodes the archive and prints its digest.
REASSEMBLE_STEM = "reassemble"

#: The script that unpacks a verified archive.
EXTRACT_STEM = "extract"

#: The script that makes the staged tree a git repository.
INIT_REPOSITORY_STEM = "init-repo"

#: The constant capacity probe, under a node's stage root.
CAPACITY_PROBE_STEM = "fleet-capacity"

#: The constant toolchain probe, under a node's stage root.
TOOLCHAIN_PROBE_STEM = "fleet-toolchain"

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
#: are ``<project>-<epoch>`` and the project grammar cannot spell this name
#: alone.
CACHE_DIRECTORY = "cache"


def cache_root(stage_root: str) -> str:
    """Where a node keeps the dependency caches its export runs share.

    Args:
        stage_root: The node's declared stage root.

    Returns:
        ``<stage_root>/cache``.
    """
    return f"{stage_root}/{CACHE_DIRECTORY}"


def log_path(target: str) -> str:
    """Where a build's transcript is written under its dispatch directory.

    Args:
        target: The dispatch's absolute remote directory.

    Returns:
        ``<target>/<RESULT_NAME>.log``, the one spelling the build writes
        and the collector reads.
    """
    return f"{target}/{RESULT_NAME}.log"


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
        name (a run id is a project path with its slashes replaced and an
        epoch second, so it carries only word characters and hyphens).
    """
    return f"fleet-{run_id}"


def make_directory_stem(run_id: str) -> str:
    """Name the script that creates one dispatch's directory.

    It lives under the stage ROOT rather than the dispatch directory, which
    does not exist until it has run; named per run so two dispatches staging
    at once cannot overwrite each other's.

    Args:
        run_id: The dispatch.

    Returns:
        The stem.
    """
    return f"mkdir-{run_id}"


def stop_stem(run_id: str) -> str:
    """Name the script that stops one dispatch.

    Args:
        run_id: The dispatch.

    Returns:
        The stem.
    """
    return f"stop-{run_id}"


__all__ = [
    "ARCHIVE_NAME",
    "BUILD_STEM",
    "CACHE_DIRECTORY",
    "CAPACITY_PROBE_STEM",
    "COLLECT_STEM",
    "ENCODED_NAME",
    "EXTRACT_STEM",
    "INIT_REPOSITORY_STEM",
    "INSTALL_STEM",
    "LAUNCH_STEM",
    "LOG_TAIL_STEM",
    "REASSEMBLE_STEM",
    "RESULT_NAME",
    "TOOLCHAIN_PROBE_STEM",
    "cache_root",
    "log_path",
    "make_directory_stem",
    "recipe_directory",
    "stop_stem",
    "task_name",
]
