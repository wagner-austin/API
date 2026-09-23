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

WHAT IS SHARED AND WHY IT IS HERE. ``tar -xzmf`` and the ``git`` acts -- the
staged tree's ``init``, and the ``init`` plus ``commit`` that makes a
companion readable as a workspace -- are the same COMMANDS on both platforms
(Windows ships bsdtar, every node has git), so they are functions in this
module rather than methods repeated in two classes. Everything else differs
in at least one token that matters, which is why it is a method: replacing a
directory is ``Remove-Item -Recurse -Force`` on one platform and ``rm -rf``
on the other.

THE SAME COMMAND IS NOT THE SAME SCRIPT, AND THAT COST A SILENT FAILURE.
Until 2026-09-22 those shared functions returned script TEXT, on the reading
that a command which is identical needs no dialect. It is not identical in
the one respect a runner depends on -- whether a failure is reported.
Measured on sedona that day, staging a companion: ``New-Item`` was given
``-LiteralPath``, a parameter it does not have (``-Path`` is the one), so the
directory was never created; ``tar`` then wrote ``could not chdir to
'C:/fleet/stage/MCPs'`` and exited non-zero; and ``powershell -File`` exited
**0** anyway, because with ``-File`` a native command's status is not the
script's unless the script says so. The transport saw success, the stage
reported success, and the node held a companion directory that did not exist.
``sh`` has the opposite default: the last command's status IS the script's,
and ``set -e`` stops at the first failure. So the shared functions return
COMMAND LINES and each dialect wraps them in its own status handling
(:meth:`Dialect.checked_script`) -- one place per platform where "a command
that failed ends the script with its status" is written down, rather than one
per script where it can be forgotten.
"""

from __future__ import annotations

from typing import Protocol

from fleet.contracts.node import NodePlatform
from fleet.core.dialect_linux import LinuxDialect
from fleet.core.dialect_windows import WindowsDialect

#: Who a companion's one commit is authored by, passed to ``git`` with
#: ``-c`` on the commit itself.
#:
#: A node has no git identity and is not given one: a dispatch that
#: configured ``user.email`` would leave that setting behind on somebody's
#: workstation, and the next thing they committed there would carry it. The
#: address is in the reserved ``.invalid`` domain (RFC 2606), so it cannot
#: reach anybody even if a tree staged here were ever pushed.
COMPANION_AUTHOR_NAME = "fleet"
COMPANION_AUTHOR_EMAIL = "fleet@corvis.invalid"


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

    def checked_script(self, commands: tuple[str, ...]) -> str:
        """Render commands as a script that ends at the first failure.

        The one place per platform where "a command that failed ends the
        script with its status" is written down. The module docstring
        carries the silent failure that put it here.

        Args:
            commands: Command lines, in order, each already quoted for a
                shell that single-quotes a literal the same way on both
                platforms.

        Returns:
            The script's text: every command in order, and a non-zero
            status from any of them as the script's own.
        """
        ...

    def reset_directory_script(self, target: str) -> str:
        """The script that empties a directory and creates it.

        For the companion trees, which land at a path derived from a
        declared name rather than from a run id, so one node's directory is
        written by every run that carries that companion. Extracting over
        what the last run left would leave a file the workspace has since
        deleted sitting in a tree that is then committed AS the workspace,
        and the check reading it would be comparing against a tree nobody
        has.

        Args:
            target: Absolute remote directory to replace.

        Returns:
            The script's text; a directory that does not exist yet is
            created, not an error.
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

    def build_script(
        self,
        *,
        target: str,
        path: str,
        workers: int,
        install: tuple[tuple[str, ...], ...],
        cache_root: str,
    ) -> str:
        """The script that readies the tree, runs the suite and records its
        status last.

        Args:
            target: Absolute remote directory holding the staged tree, its
                root.
            path: The project's directory inside it, ``""`` for the root.
            workers: Test workers the capacity check granted.
            install: The project's declared install steps, run at the root
                before the recipe, each an argv in the source grammar.
            cache_root: The node's cache directory, which the package
                managers are pointed at.

        Returns:
            The script's text. Its last act writes the recipe's exit status
            to the result file, so the file's absence means running.
        """
        ...

    def log_tail_script(self, target: str, lines: int) -> str:
        """The script that prints the end of the build's transcript.

        Args:
            target: Absolute remote directory holding the staged tree.
            lines: How many lines from the end.

        Returns:
            The script's text; an absent transcript prints nothing.
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

    def stop_script(self, *, target: str, run_id: str) -> str:
        """The script that stops one dispatch's build, its whole process tree.

        Args:
            target: Absolute remote directory holding the staged tree, where
                a build that must be stopped by process id recorded it.
            run_id: The dispatch, which names its own task or unit.

        Returns:
            The script's text. It never prompts, it ends every process the
            build started and not only the one its task or unit launched, and
            it exits 0 whether or not the build was still running.
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


def extract_commands(archive: str, destination: str) -> tuple[str, ...]:
    """The command that unpacks a verified archive.

    THE SAME COMMAND ON BOTH PLATFORMS: Windows ships bsdtar and the flags
    agree. ``-m`` makes extracted files take the node's clock rather than
    the sender's. Without it a tree staged from a machine whose clock is
    ahead produces make targets that look newer than their sources, and the
    build does nothing at all -- which reads as a suite that passed
    instantly.

    THE TWO PATHS ARE SEPARATE because a companion's are: its archive stays
    in a staging directory of its own and only the repository lands in the
    directory the recipe reads, so the tree committed there is the export
    and nothing else. A project's dispatch passes its own directory for
    both, which is where its archive already sits.

    Args:
        archive: Absolute path to the ``.tgz`` on the node.
        destination: Absolute remote directory to unpack it into.

    Returns:
        The one command line, for a dialect's
        :meth:`Dialect.checked_script`.
    """
    return (f"tar -xzmf '{archive}' -C '{destination}'",)


def companion_repository_commands(target: str, sha: str) -> tuple[str, ...]:
    """The commands that make a staged companion a one-commit repository.

    THEY COMMIT, where :func:`init_repository_commands` deliberately does
    not. A companion exists to be read as a workspace, and the thing that
    reads it reads HEAD -- slime's lift check compares its lifted files
    against ``git show HEAD:<path>`` precisely so that an uncommitted edit
    in the workspace is not mistaken for the code. On a node there is no
    HEAD until one is made, so the export is committed here, and the
    identity is passed with ``-c`` rather than configured: nothing this
    package does leaves state on a node.

    ``--force`` on the add, which the project's tree does not use and must
    not: there the ``.gitignore`` is what makes a staged tree index the same
    files a checkout tracks. Here the directory holds the archive of a
    commit and nothing else, so the tracked set is already decided, and an
    ignore rule that skipped one of those files would leave the check
    reporting a workspace file as missing when the workspace has it.

    Args:
        target: Absolute remote directory holding the extracted companion.
        sha: The commit the archive was written from, recorded in the
            message so the tree on the node names what it is.

    Returns:
        The three command lines, in order, for a dialect's
        :meth:`Dialect.checked_script`.
    """
    return (
        f"git -C '{target}' init --quiet",
        f"git -C '{target}' add --all --force",
        f"git -C '{target}' -c user.name='{COMPANION_AUTHOR_NAME}' "
        f"-c user.email='{COMPANION_AUTHOR_EMAIL}' commit --quiet "
        f"--message 'fleet companion export {sha}'",
    )


def init_repository_commands(target: str) -> tuple[str, ...]:
    """The commands that make a staged tree a git repository.

    THE SAME ON BOTH PLATFORMS, and without them a staged build lints
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

    THE INDEX IS FILLED TOO, since the first fleet verdict (MCPs board task
    fd5cabfa, 2026-09-21T10:03Z): ``MCPs/packages/maketools`` at 9ee70275
    on diphtheria read ``1 failed, 416 passed``, the one failure
    ``test_git_lists_the_tracked_files_matching_a_pattern`` asserting that
    ``git ls-files CLAUDE.md`` names the file, and in a repository with an
    empty index it names nothing. A checkout's suite reads its own tracked
    set (that package's ``lint-makefiles`` lints exactly the tracked
    Makefiles), so a staged tree whose index is empty is not the tree the
    suite was written against. ``git add --all`` under the tree's own
    ``.gitignore`` indexes what a checkout tracks: an export is the tracked
    files by construction, and a working tree's ignored build output stays
    out the way it does in the checkout. Nothing is committed, because
    nothing reads a commit in the tree the recipe runs in; the one staged
    tree that IS read as a commit is a companion, and
    :func:`companion_repository_commands` says why it is the exception.

    Args:
        target: Absolute remote directory holding the staged tree.

    Returns:
        The two command lines, in order, for a dialect's
        :meth:`Dialect.checked_script`.
    """
    return (f"git -C '{target}' init --quiet", f"git -C '{target}' add --all")


__all__ = [
    "COMPANION_AUTHOR_EMAIL",
    "COMPANION_AUTHOR_NAME",
    "Dialect",
    "companion_repository_commands",
    "extract_commands",
    "for_platform",
    "init_repository_commands",
]
