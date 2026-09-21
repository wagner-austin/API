"""The scripts a node is handed, in the language that node speaks.

EVERY REMOTE ACT IS A SCRIPT SENT AND RUN BY PATH (:mod:`fleet.core.remote`),
and until 2026-09-20 every one of those scripts was PowerShell: the fleet was
three Windows laptops and a Windows hub. diphtheria (Ubuntu 24.04, the corvis
Docker host) is the first Linux node, and a dispatch to it needs the same
twelve acts in ``sh`` -- how a file is written over ssh, how a script is run,
how a directory is made, how the archive is reassembled and digested, how the
suite is detached from the connection, how its result and the node's capacity
are read, how a run is stopped. Board task 33bb86ce, Phase C step 3. The
session observer's two acts joined the set under board task cd5010c4: its
PowerShell script had been sent to every node regardless, and diphtheria
answered ``powershell: command not found`` on every tick, logged as a node
that did not answer.

A DIALECT IS THE WHOLE SET, CHOSEN ONCE BY THE NODE'S DECLARED PLATFORM. The
alternative -- an ``if platform == "linux"`` inside each of the seven modules
that render a script -- would scatter one decision over seven files and let
a new act be added to one dialect and not the other without anything
noticing. Here the :class:`Dialect` protocol names every act, both
implementations must supply all of them or fail to type-check, and a caller
asks :func:`for_platform` once and never reasons about the platform again.

WHAT IS SHARED AND WHY IT IS HERE. ``tar -xzmf`` and ``git init`` are the
same command on both platforms (Windows ships bsdtar, every node has git), so
they are functions in this module rather than methods repeated in two
classes; the lifecycle modules call them directly. Everything else differs
in at least one token that matters, which is why it is a method.
"""

from __future__ import annotations

from typing import Protocol

from fleet.contracts.node import NodePlatform
from fleet.core import names
from fleet.core.dialect_linux import LinuxDialect
from fleet.core.dialect_windows import WindowsDialect


class Dialect(Protocol):
    """Every script a dispatch sends, rendered for one platform.

    The docstrings on each implementation carry the incidents behind the
    text; this protocol carries only what each act is for, so a reader
    choosing between the two knows what must be the same.
    """

    def script_path(self, directory: str, stem: str) -> str:
        """Name a script file under a remote directory.

        Args:
            directory: Absolute remote directory.
            stem: The script's name without its extension.

        Returns:
            The absolute path, with the platform's extension.
        """
        ...

    def fleet_directory(self, user: str) -> str:
        """The provisioned account's ``.fleet`` directory.

        Where a script that belongs to no dispatch lands: the session
        observer's, and anything else sent to the account rather than into a
        staged tree. Spelled from the account name because the write command
        on the far side expands nothing, so a home directory must be an
        absolute literal path.

        Args:
            user: The provisioned account.

        Returns:
            The absolute directory, with no trailing separator.
        """
        ...

    def observe_sessions_script(self) -> str:
        """The constant script that reports the account's Claude Code sessions.

        Returns:
            The script's text. It prints ONE compact JSON document:
            ``platform`` in the harness's own spelling (``win32``,
            ``linux``), ``hostname`` lowercased, and ``records``, every
            ``~/.claude/sessions/*.json`` verbatim. An absent directory is
            zero records, not a fault.
        """
        ...

    def write_command(self, path: str) -> str:
        """The remote command that writes standard input to a path.

        Args:
            path: Absolute remote path; its parent is created.

        Returns:
            The command, as one argument for ssh's remote shell.
        """
        ...

    def invocation(self) -> tuple[str, ...]:
        """How a script file is run by path.

        Returns:
            The argv prefix; the path is appended.
        """
        ...

    def echo_command(self, text: str) -> str:
        """One line that prints a literal to standard output.

        Args:
            text: The literal, which contains no quote characters.

        Returns:
            The line.
        """
        ...

    def make_directory_script(self, target: str) -> str:
        """The script that creates a dispatch's directory.

        Args:
            target: Absolute remote directory for this dispatch.

        Returns:
            The script's text.
        """
        ...

    def reassemble_script(self, target: str) -> str:
        """The script that decodes the archive and prints its digest.

        It must NOT extract: the sender compares the digest first.

        Args:
            target: Absolute remote directory for this dispatch.

        Returns:
            The script's text, whose only output is the lowercase hex SHA-256.
        """
        ...

    def build_script(self, *, target: str, project: str, workers: int) -> str:
        """The script that runs the suite and records its status last.

        Args:
            target: Absolute remote directory holding the staged tree.
            project: Repo-relative project path.
            workers: Test workers the capacity check granted.

        Returns:
            The script's text. Its last act writes the recipe's exit status
            to the result file, so the file's absence means running.
        """
        ...

    def launch_script(self, *, target: str, run_id: str) -> str:
        """The script that detaches the build from the ssh connection.

        Args:
            target: Absolute remote directory holding the staged tree.
            run_id: The dispatch, which names its own task or unit.

        Returns:
            The script's text. It prints ``launched`` only once the build has
            actually begun, and exits non-zero otherwise.
        """
        ...

    def result_script(self, target: str) -> str:
        """The script that reports how a dispatch ended, if it has.

        Args:
            target: Absolute remote directory holding the staged tree.

        Returns:
            The script's text. It prints the exit status and the epoch
            second the result was written, space-separated, or nothing while
            the run is still going.
        """
        ...

    def stop_script(self, run_id: str) -> str:
        """The script that stops one dispatch's task or unit.

        Args:
            run_id: The dispatch.

        Returns:
            The script's text. It never prompts and it exits 0 whether or not
            the task still existed.
        """
        ...

    def capacity_probe_script(self) -> str:
        """The constant script that reports free memory and disk.

        Returns:
            The script's text: ``key=value`` lines, one per field.
        """
        ...

    def toolchain_probe_script(self) -> str:
        """The constant script that reports each required tool's presence.

        Returns:
            The script's text: ``name=yes|no=version`` lines.
        """
        ...


def for_platform(platform: NodePlatform) -> Dialect:
    """Choose the dialect a node speaks.

    Args:
        platform: The node's declared platform.

    Returns:
        The dialect.
    """
    if platform == "windows":
        return WindowsDialect()
    return LinuxDialect()


def extract_script(target: str) -> str:
    """Render the script that unpacks a verified archive.

    THE SAME ON BOTH PLATFORMS: Windows ships bsdtar and the flags agree.
    ``-m`` makes extracted files take the node's clock rather than the
    sender's. Without it a tree staged from a machine whose clock is ahead
    produces make targets that look newer than their sources, and the build
    does nothing at all -- which reads as a suite that passed instantly.

    Args:
        target: Absolute remote directory for this dispatch.

    Returns:
        The script's text.
    """
    return f"tar -xzmf '{target}/{names.ARCHIVE_NAME}' -C '{target}'\n"


def init_repository_script(target: str) -> str:
    """Render the script that makes a staged tree a git repository.

    THE SAME ON BOTH PLATFORMS, and without it a staged build lints
    different files from a local one. Ruff honours ``.gitignore`` and applies
    it ONLY inside a git repository -- so a tree that carries the file but no
    ``.git`` silently widens what gets linted to include everything the
    repository deliberately excludes.

    Measured on lavender 2026-09-04, dispatching ``tools/hpc3``: 902 ruff
    errors, all in ``tools/hpc3/runs``, which ``.gitignore`` line 170 excludes
    as build artifacts while explicitly tracking the run documents beside
    them. The same tree with ``git init`` run in it reports ``All checks
    passed``. Locally ``ruff check .`` passes and
    ``ruff check . --no-respect-gitignore`` reports exactly 902 -- the same
    number, which identifies the mechanism rather than suggesting it.

    THE ALTERNATIVE WAS TO ADD AN EXCLUDE TO THE PROJECT, AND IT WOULD HAVE
    BEEN WRONG. The repository already states which paths are build output;
    a ruff ``exclude`` restating it is a second copy of one policy, and the
    copy that drifts is the one nobody looks at. Reproducing the environment
    a build is defined against is this package's job, not the project's.

    Args:
        target: Absolute remote directory holding the staged tree.

    Returns:
        The script's text. It initialises an empty repository and commits
        nothing: the ignore rules are read from the working tree, so the
        marker is all ruff needs.
    """
    return f"git -C '{target}' init --quiet\n"


__all__ = [
    "Dialect",
    "extract_script",
    "for_platform",
    "init_repository_script",
]
