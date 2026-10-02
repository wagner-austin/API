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

WHY POETRY'S KEYRING IS OFF. Every build exports ``POETRY_KEYRING_ENABLED``
false (:data:`fleet.core.linux_isolated_build.POETRY_KEYRING_OFF` says why):
left on, poetry waits on a D-Bus SecretService that a headless node never
answers, and the build times out inside ``poetry install`` (MCPs board task
c837fd8d).
"""

from __future__ import annotations

import posixpath
import shlex

from fleet.contracts.project import MAKE_TARGET
from fleet.contracts.runner_slice import CI_SLICE_NAME
from fleet.core import names
from fleet.core.agent_label import AGENT_LABEL_VARIABLE, require_agent_label
from fleet.core.linux_capacity_probe import CAPACITY_PROBE_BODY
from fleet.core.linux_isolated_build import POETRY_KEYRING_OFF, isolated_build_lines
from fleet.core.linux_toolchain_probe import TOOLCHAIN_PROBE_BODY

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
#: ``~/.cargo/bin`` is where rustup puts cargo at user scope, and it is added
#: for the same reason as ``~/.local/bin``: the probe that measures a node's
#: ``rust`` declaration and the build that compiles a crate must find the
#: one cargo the operator installed (MCPs board task 1e2da299).
PROLOGUE = 'set -eu\nPATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"\nexport PATH\n'

#: The capacity probe, verbatim. Nothing is substituted into it.
#:
#: ``MemAvailable`` rather than ``MemFree``: the kernel's own estimate of
#: what a new process can take without swapping, which counts reclaimable
#: cache the way Windows' ``FreePhysicalMemory`` does not have to, net of
#: what the node's memory-capped containers may still grow into
#: (:mod:`fleet.core.linux_capacity_probe` says why, MCPs board task
#: a282400d). ``df`` of the root filesystem mirrors the PowerShell probe's
#: drive C, the disk the stage root lives on for every node declared so far;
#: both are read in kibibytes or bytes and divided to gibibytes with three
#: decimals so the parser sees one shape from both dialects.
#:
#: THE ``ci_slice_*`` PAIR is printed only on a node whose CI runners run in
#: ``runners.slice`` and whose ``memory.high`` is a number rather than
#: ``max`` (MCPs board task 5d6e57e7): a refusal on such a node can then say
#: CI holds the lane. Measured on lavender-wsl, 2026-09-29 18:2xZ:
#: memory.current 17179713536 against memory.high 17179869184, with 8.2 GB
#: available; diphtheria has no such cgroup and prints neither.
CAPACITY_PROBE_SCRIPT = (
    PROLOGUE + CAPACITY_PROBE_BODY + f"s=/sys/fs/cgroup/{CI_SLICE_NAME}\n"
    'if [ -r "$s/memory.current" ] && [ -r "$s/memory.high" ]; then\n'
    '  h="$(cat "$s/memory.high")"\n'
    '  case "$h" in\n'
    "    ''|*[!0-9]*) ;;\n"
    '    *) awk -v c="$(cat "$s/memory.current")" -v h="$h" \'BEGIN { printf '
    '"ci_slice_current_gb=%.3f\\nci_slice_high_gb=%.3f\\n", c / 1073741824, h / 1073741824 }\' ;;\n'
    "  esac\n"
    "fi\n"
)

#: The toolchain probe, verbatim: the prologue, then
#: :data:`fleet.core.linux_toolchain_probe.TOOLCHAIN_PROBE_BODY`, which says
#: what each line asks.
TOOLCHAIN_PROBE_SCRIPT = PROLOGUE + TOOLCHAIN_PROBE_BODY


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
        return self.checked_script((("mkdir", "-p", target),))

    def checked_script(self, commands: tuple[tuple[str, ...], ...]) -> str:
        """Render commands as a script that ends at the first failure.

        Nothing is added per command: :data:`PROLOGUE`'s ``set -e`` already
        ends the script at the first non-zero status and ``sh`` exits with
        it, which is the default the other dialect has to be given. Each
        word is quoted by :func:`shlex.quote`, which leaves a plain path bare
        and single-quotes anything else.

        Args:
            commands: Commands in order, each an argument vector.

        Returns:
            The script's text.
        """
        lines = (" ".join(shlex.quote(word) for word in command) for command in commands)
        return PROLOGUE + "".join(f"{line}\n" for line in lines)

    def reset_directory_script(self, target: str) -> str:
        """The script that empties a companion's directory and creates it.

        Args:
            target: Absolute remote directory to replace.

        Returns:
            The script's text. ``rm -rf`` needs no guard for an absent path,
            which is the first run on a node, and ``-f`` removes the
            read-only loose objects of the git repository a previous run
            made there.
        """
        return self.checked_script((("rm", "-rf", target), ("mkdir", "-p", target)))

    def retire_script(
        self, *, target: str, retained: str, scripts: tuple[str, ...], task: str
    ) -> str:
        """Keep a settled run's transcript, then remove its directories and scripts.

        Args:
            target: The dispatch's absolute remote directory; its staging
                directory beside it (:func:`fleet.core.names.staging_directory`)
                goes with it.
            retained: Where its transcript is kept.
            scripts: The scripts it left under the stage root.
            task: The run's transient unit. Nothing removes it here: the
                launch starts it with ``--collect``, so the user manager
                unloads it when the build exits, failed or not.

        Returns:
            The script's text. ``mv`` is guarded because a run cancelled
            before its build started wrote no transcript; ``rm -rf`` and
            ``rm -f`` need no guard for what is already gone, and ``-f``
            removes the read-only objects of the repository staged there.
        """
        log = shlex.quote(names.log_path(target))
        lines = (
            f"mkdir -p {shlex.quote(posixpath.dirname(retained))}",
            f"if [ -e {log} ]; then mv -f {log} {shlex.quote(retained)}; fi",
            f"rm -rf {shlex.quote(target)} {shlex.quote(names.staging_directory(target))}",
            "rm -f " + " ".join(shlex.quote(script) for script in scripts),
        )
        return PROLOGUE + "".join(f"{line}\n" for line in lines)

    def digest_script(self, target: str) -> str:
        """Print the landed archive's SHA-256, extracting nothing.

        Args:
            target: Absolute remote directory for this dispatch.

        Returns:
            The script's text. ``sha256sum`` prints the digest and the file
            name; only the digest is kept, so the output is exactly what the
            sender compares.
        """
        return f"{PROLOGUE}sha256sum '{target}/{names.ARCHIVE_NAME}' | cut -d ' ' -f 1\n"

    def build_script(
        self,
        *,
        target: str,
        path: str,
        workers: int,
        install: tuple[tuple[str, ...], ...],
        cache_root: str,
        isolated_docker: bool,
        elevated: bool,
        agent: str,
    ) -> str:
        """Ready the tree, run the recipe in the project, write its status last.

        Deliberately NOT under ``set -e`` for the install steps or the recipe
        itself: a failing suite is a result to record, not a fault in the
        script, so each status is captured and written rather than ending the
        script before the write. Everything else is still fail-fast. The
        caches, the install steps and their order are the Windows dialect's,
        whose docstring carries the why.

        A docker project's build is :mod:`fleet.core.linux_isolated_build`'s
        instead: the same steps in the same order, run as execdocker against
        its rootless daemon, with that user's caches rather than the node's.

        Args:
            target: Absolute remote directory holding the export, its root.
            path: The project's directory inside the export, ``""`` for the
                root.
            workers: Test workers the capacity check granted.
            install: The project's declared install steps, argv each.
            cache_root: The node's cache directory, unread by a docker
                project's build, whose caches are execdocker's.
            isolated_docker: True for a project that declares the ``docker``
                tag.
            elevated: True for a project that declares the ``elevated`` tag,
                which no linux node carries.
            agent: The job's submitting agent label, exported as
                ``BOARD_AGENT_LABEL``.

        Returns:
            The script's text. Its last act writes the recipe's exit status
            to the result file.

        Raises:
            ValueError: When ``elevated`` is True, refused for the reason
                :meth:`launch_script` gives, or ``agent`` is outside the
                board's label grammar.
        """
        if elevated:
            raise ValueError(
                f"the project at {path!r} requires the elevated tag, and a linux node has no "
                "elevated runner; it runs only on a Windows node that declares one"
            )
        label = require_agent_label(agent)
        if isolated_docker:
            lines = isolated_build_lines(
                target=target, path=path, workers=workers, install=install, agent=label
            )
            return PROLOGUE + "\n".join(lines) + "\n"
        log = names.log_path(target)
        result = f"{target}/{names.RESULT_NAME}"
        lines = [
            f"{PROLOGUE}npm_config_cache='{cache_root}/npm'",
            f"POETRY_CACHE_DIR='{cache_root}/pypoetry'",
            f"{POETRY_KEYRING_OFF[0]}='{POETRY_KEYRING_OFF[1]}'",
            f"PLAYWRIGHT_BROWSERS_PATH='{cache_root}/ms-playwright'",
            f"PYTEST_XDIST_AUTO_NUM_WORKERS='{workers}'",
            f"{AGENT_LABEL_VARIABLE}='{label}'",
            f"export npm_config_cache POETRY_CACHE_DIR {POETRY_KEYRING_OFF[0]} "
            f"PLAYWRIGHT_BROWSERS_PATH PYTEST_XDIST_AUTO_NUM_WORKERS {AGENT_LABEL_VARIABLE}",
            f"cd '{target}'",
        ]
        for step in install:
            command = " ".join(step)
            lines.append(f"printf '$ %s\\n' '{command}' >> '{log}'")
            lines.append("set +e")
            lines.append(f"{command} >> '{log}' 2>&1")
            lines.append("status=$?")
            lines.append("set -e")
            lines.append(
                f"if [ \"$status\" -ne 0 ]; then printf '%s\\n' \"$status\" > '{result}'; "
                f"exit 0; fi"
            )
        lines.append(f"cd '{names.recipe_directory(target, path)}'")
        lines.append("set +e")
        lines.append(f"make {MAKE_TARGET} >> '{log}' 2>&1")
        lines.append("status=$?")
        lines.append("set -e")
        lines.append(f"printf '%s\\n' \"$status\" > '{result}'")
        return "\n".join(lines) + "\n"

    def log_tail_script(self, target: str, lines: int) -> str:
        """Print the last lines of the build's transcript, or nothing.

        Args:
            target: Absolute remote directory holding the export.
            lines: How many lines from the end.

        Returns:
            The script's text; an absent transcript prints nothing.
        """
        log = names.log_path(target)
        return f"{PROLOGUE}if [ -f '{log}' ]; then tail -n {lines} '{log}'; fi\n"

    def launch_script(self, *, target: str, run_id: str, elevated: bool) -> str:
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
            elevated: Whether the project requires the ``elevated`` tag.

        Returns:
            The script's text.

        Raises:
            ValueError: When ``elevated`` is True. A linux node never
                declares an elevated runner (the node decoder refuses one),
                so the capacity check places no elevated project here; a
                build rendered for one anyway would run without the rights
                its suite was written for, so it is refused rather than
                rendered.
        """
        if elevated:
            raise ValueError(
                f"the dispatch {run_id!r} requires the elevated tag, and a linux node has no "
                "elevated runner; it runs only on a Windows node that declares one"
            )
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

    def stop_script(self, *, target: str, run_id: str) -> str:
        """Stop the unit if it is still active, and say so either way.

        Stopping a unit stops its control group, every process the build
        started, so the unit's name is all this needs and ``target`` is not
        read: the process id the Windows build records has no work to do here.

        Args:
            target: Absolute remote directory holding the staged tree; unused
                on this platform, for the reason above.
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
