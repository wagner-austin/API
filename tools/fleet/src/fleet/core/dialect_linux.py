"""The ``sh`` dialect: every script a Linux node is handed.

WRITTEN FOR diphtheria (Ubuntu 24.04, the corvis Docker host), the first
Linux node the fleet may dispatch to, and held to the same protocol as the
PowerShell dialect so the two cannot drift apart act by act. Every script is
POSIX ``sh`` under ``set -eu``: a command that fails ends the script with its
status, which :mod:`fleet.core.remote` turns into ``DISPATCH_FAILED``
carrying the node's own stderr, and an unset variable is a fault rather than
an empty string.

HOW THE BUILD OUTLIVES THE CONNECTION. sshd on Linux does not put a session
in a job object, but it does send SIGHUP to the session's process group when
the connection closes, and a user's systemd manager stops with the user's
last session unless lingering is enabled. The build is therefore started as
a TRANSIENT USER UNIT (``systemd-run --user``), which is to the user manager
what a scheduled task is to Task Scheduler: owned by the machine, not by the
connection, stoppable by name, and started or refused synchronously. The
launch script refuses first when lingering is off, naming the one command
that turns it on, because a unit started without it would be killed the
moment the ssh session that started it ended -- and would report nothing.

WHERE POETRY LIVES. pipx and poetry's own installer both put the executable
under ``~/.local/bin``, which a login shell adds to PATH and a non-interactive
ssh command does not (Ubuntu's ``~/.profile`` is read by login shells only).
Every script here prepends it, so the same ``poetry`` the operator ran by
hand is the one the build finds.
"""

from __future__ import annotations

from fleet.contracts.project import MAKE_TARGET
from fleet.core import names

#: How a script file is run by path.
SH_INVOCATION = ("/bin/sh",)

#: How a script's body is written on the far side.
#:
#: The same shape as the PowerShell command: one argument for the remote
#: login shell, the parent created in the same round trip, the body streamed
#: from standard input so no shell between here and the disk interprets it.
#: The path is interpolated inside single quotes, which ``sh`` reads
#: literally; a path with a single quote in it cannot be staged, and stage
#: roots and run ids carry none by construction.
WRITE_COMMAND = "mkdir -p \"$(dirname '{path}')\" && cat > '{path}'"

#: The first lines of every script: fail fast, and find the user's tools.
PROLOGUE = 'set -eu\nPATH="$HOME/.local/bin:$PATH"\nexport PATH\n'

#: The capacity probe, verbatim. Nothing is substituted into it.
#:
#: ``MemAvailable`` rather than ``MemFree``: the kernel's own estimate of
#: what a new process can take without swapping, which counts reclaimable
#: cache the way Windows' ``FreePhysicalMemory`` does not have to. ``df`` of
#: the root filesystem mirrors the PowerShell probe's drive C, the disk the
#: stage root lives on for every node declared so far; both are read in
#: kibibytes and divided to gibibytes with three decimals so the parser sees
#: one shape from both dialects.
CAPACITY_PROBE_SCRIPT = (
    PROLOGUE
    + "awk '/^MemAvailable:/ { printf \"free_ram_gb=%.3f\\n\", $2 / 1048576 }' /proc/meminfo\n"
    "df -kP / | awk 'NR == 2 { printf \"free_disk_gb=%.3f\\n\", $4 / 1048576 }'\n"
    "printf 'logical_cores=%s\\n' \"$(nproc)\"\n"
)

#: The toolchain probe, verbatim.
#:
#: THE INTERPRETER IS ``python3`` AND IS REPORTED AS ``python``. The required
#: tool is named ``python`` because that is what a Windows PATH calls it; on
#: Linux the Makefile prologue (``scripts/make/shell.mk``) calls ``python3``,
#: so that is what the probe asks for, and it reports the answer under the
#: name the contract requires so :func:`fleet.contracts.toolchain.python_is_right`
#: reads one report on every platform. The managers asked about are the two
#: that exist here: ``apt-get`` for git and make, ``pipx`` for poetry.
TOOLCHAIN_PROBE_SCRIPT = (
    PROLOGUE + "report() {\n"
    '  if command -v "$2" > /dev/null 2>&1; then\n'
    '    printf \'%s=yes=%s\\n\' "$1" "$("$2" --version 2>&1 | head -n 1 | tr -d \'\\r\')"\n'
    "  else\n"
    "    printf '%s=no=\\n' \"$1\"\n"
    "  fi\n"
    "}\n"
    "report python python3\n"
    "report poetry poetry\n"
    "report git git\n"
    "report make make\n"
    "report tar tar\n"
    "report apt-get apt-get\n"
    "report pipx pipx\n"
)


#: The body of the session-observer script: what python3 runs, verbatim.
#:
#: THE DOCUMENT IS BUILT BY python3, NOT BY sh. The records go out verbatim
#: inside one JSON array, and ``sh`` has no way to join files into one that
#: survives an empty directory, a record without a trailing newline or a
#: field with a quote in it; ``python3`` is on every Linux node by the
#: toolchain contract (``report python python3`` above), on the PATH the
#: prologue prepends. ``sys.platform`` is ``linux``, the same word Node's
#: ``process.platform`` writes into the harness's ``pidDomain``; the
#: hostname is lowercased as the harness spells it (measured on diphtheria
#: 2026-09-21: ``hostname`` prints ``Diphtheria``). Records are read in file
#: name order so two runs over one directory agree, and an absent directory
#: is an empty array rather than an error. Kept apart from the ``sh`` that
#: feeds it so a test can run exactly this text under the interpreter it has.
OBSERVE_SESSIONS_PYTHON = (
    "import json\n"
    "import os\n"
    "import socket\n"
    "import sys\n"
    "directory = os.path.join(os.path.expanduser('~'), '.claude', 'sessions')\n"
    "records = []\n"
    "if os.path.isdir(directory):\n"
    "    for name in sorted(os.listdir(directory)):\n"
    "        if name.endswith('.json'):\n"
    "            with open(os.path.join(directory, name), encoding='utf-8') as handle:\n"
    "                records.append(json.load(handle))\n"
    "document = {\n"
    "    'platform': sys.platform,\n"
    "    'hostname': socket.gethostname().lower(),\n"
    "    'records': records,\n"
    "}\n"
    "print(json.dumps(document, separators=(',', ':')))\n"
)

#: The session-observer script, verbatim: the prologue, then
#: :data:`OBSERVE_SESSIONS_PYTHON` fed to ``python3`` through a quoted
#: heredoc, so nothing in the body is expanded by ``sh`` on the way.
OBSERVE_SESSIONS_SCRIPT = f"{PROLOGUE}python3 - <<'PY'\n{OBSERVE_SESSIONS_PYTHON}PY\n"


class LinuxDialect:
    """The ``sh`` rendering of every remote act."""

    def script_path(self, directory: str, stem: str) -> str:
        """Name a ``.sh`` under a remote directory.

        Args:
            directory: Absolute remote directory.
            stem: The script's name without its extension.

        Returns:
            The absolute path.
        """
        return f"{directory}/{stem}.sh"

    def fleet_directory(self, user: str) -> str:
        """The provisioned account's ``.fleet`` directory.

        Args:
            user: The provisioned account.

        Returns:
            ``/home/<user>/.fleet``, literal and absolute, the path
            :data:`WRITE_COMMAND` single-quotes: a ``~`` inside those quotes
            would be a directory called ``~``.
        """
        return f"/home/{user}/.fleet"

    def observe_sessions_script(self) -> str:
        """The constant session-observer script.

        Returns:
            :data:`OBSERVE_SESSIONS_SCRIPT`.
        """
        return OBSERVE_SESSIONS_SCRIPT

    def write_command(self, path: str) -> str:
        """The remote command that writes standard input to a path.

        Args:
            path: Absolute remote path; its parent is created.

        Returns:
            :data:`WRITE_COMMAND` with the path in place.
        """
        return WRITE_COMMAND.format(path=path)

    def invocation(self) -> tuple[str, ...]:
        """How a script file is run by path.

        Returns:
            :data:`SH_INVOCATION`.
        """
        return SH_INVOCATION

    def echo_command(self, text: str) -> str:
        """One line that prints a literal.

        Args:
            text: The literal, which contains no quote characters.

        Returns:
            A ``printf`` of it.
        """
        return f"printf '%s\\n' '{text}'"

    def make_directory_script(self, target: str) -> str:
        """The script that creates a dispatch's directory.

        Args:
            target: Absolute remote directory for this dispatch.

        Returns:
            ``mkdir -p`` of it.
        """
        return f"{PROLOGUE}mkdir -p '{target}'\n"

    def reassemble_script(self, target: str) -> str:
        """Decode the one-line base64 and print the SHA-256, extracting nothing.

        Args:
            target: Absolute remote directory for this dispatch.

        Returns:
            The script's text. ``sha256sum`` prints the digest and the file
            name; only the digest is kept, so the output is exactly what the
            sender compares.
        """
        encoded = f"{target}/{names.ENCODED_NAME}"
        archive = f"{target}/{names.ARCHIVE_NAME}"
        return (
            f"{PROLOGUE}base64 -d '{encoded}' > '{archive}'\n"
            f"sha256sum '{archive}' | cut -d ' ' -f 1\n"
        )

    def build_script(self, *, target: str, project: str, workers: int) -> str:
        """Run the recipe in the project and write its status last.

        Deliberately NOT under ``set -e`` for the recipe itself: a failing
        suite is a result to record, not a fault in the script, so ``make``'s
        status is captured and written rather than ending the script before
        the write. Everything before it is still fail-fast.

        Args:
            target: Absolute remote directory holding the staged tree.
            project: Repo-relative project path.
            workers: Test workers the capacity check granted.

        Returns:
            The script's text. Its last act writes the recipe's exit status
            to the result file.
        """
        result = f"{target}/{names.RESULT_NAME}"
        return (
            f"{PROLOGUE}cd '{target}/{project}'\n"
            f"PYTEST_XDIST_AUTO_NUM_WORKERS='{workers}'\n"
            f"export PYTEST_XDIST_AUTO_NUM_WORKERS\n"
            f"set +e\n"
            f"make {MAKE_TARGET} > '{result}.log' 2>&1\n"
            f"status=$?\n"
            f"set -e\n"
            f"printf '%s\\n' \"$status\" > '{result}'\n"
        )

    def launch_script(self, *, target: str, run_id: str) -> str:
        """Start the build as a transient user unit, and prove it began.

        ``systemd-run`` returns once the manager has accepted and started the
        unit, and exits non-zero when it refuses, so the ``launched`` line
        after it is the same proof the Windows dialect waits for. ``--collect``
        lets a unit that failed be forgotten rather than sit in the failed
        state where the next dispatch of the same run id could not start;
        ``--quiet`` keeps the "Running as unit" line off the transcript that
        the sender compares to ``launched``. The lingering check comes first
        and names the fix, because a unit started without it dies with the
        ssh session and reports nothing.

        Args:
            target: Absolute remote directory holding the staged tree.
            run_id: The dispatch, which names its own unit.

        Returns:
            The script's text.
        """
        unit = names.task_name(run_id)
        build = f"{target}/{names.BUILD_STEM}.sh"
        return (
            f'{PROLOGUE}user="$(id -un)"\n'
            f'if [ "$(loginctl show-user "$user" -p Linger --value)" != yes ]; then\n'
            f"  printf '%s\\n' \"lingering is off for $user, so a user unit would die with this "
            f'ssh session; enable it once with: sudo loginctl enable-linger $user" >&2\n'
            f"  exit 1\n"
            f"fi\n"
            f"systemd-run --user --unit='{unit}' --collect --quiet "
            f"--property=WorkingDirectory='{target}' /bin/sh '{build}'\n"
            f"printf 'launched\\n'\n"
        )

    def result_script(self, target: str) -> str:
        """Print the status and the epoch second it was written, or nothing.

        Args:
            target: Absolute remote directory holding the staged tree.

        Returns:
            The script's text. ``stat -c %Y`` is the file's modification time
            in epoch seconds, UTC by definition, so no offset arithmetic is
            needed on this platform.
        """
        result = f"{target}/{names.RESULT_NAME}"
        return (
            f"{PROLOGUE}if [ -f '{result}' ]; then\n"
            f"  code=\"$(tr -d '[:space:]' < '{result}')\"\n"
            f"  written=\"$(stat -c %Y '{result}')\"\n"
            f'  printf \'%s %s\\n\' "$code" "$written"\n'
            f"fi\n"
        )

    def stop_script(self, run_id: str) -> str:
        """Stop the unit if it is still active, and say so either way.

        Args:
            run_id: The dispatch.

        Returns:
            The script's text. Guarded by ``is-active`` rather than stopping
            unconditionally, because ``systemctl stop`` of a unit that has
            already been collected exits 5 (not loaded) and the cancel must
            close the row whether the build was still going or not.
        """
        unit = names.task_name(run_id)
        return (
            f"{PROLOGUE}if systemctl --user is-active --quiet '{unit}'; then\n"
            f"  systemctl --user stop '{unit}'\n"
            f"fi\n"
            f"printf 'stopped {unit}\\n'\n"
        )

    def capacity_probe_script(self) -> str:
        """The constant capacity probe.

        Returns:
            :data:`CAPACITY_PROBE_SCRIPT`.
        """
        return CAPACITY_PROBE_SCRIPT

    def toolchain_probe_script(self) -> str:
        """The constant toolchain probe.

        Returns:
            :data:`TOOLCHAIN_PROBE_SCRIPT`.
        """
        return TOOLCHAIN_PROBE_SCRIPT


__all__ = [
    "CAPACITY_PROBE_SCRIPT",
    "OBSERVE_SESSIONS_PYTHON",
    "OBSERVE_SESSIONS_SCRIPT",
    "PROLOGUE",
    "SH_INVOCATION",
    "TOOLCHAIN_PROBE_SCRIPT",
    "WRITE_COMMAND",
    "LinuxDialect",
]
