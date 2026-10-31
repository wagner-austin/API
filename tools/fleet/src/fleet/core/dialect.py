"""Every script a node is handed, in ``sh``.

EVERY REMOTE ACT IS A SCRIPT SENT AND RUN BY PATH (:mod:`fleet.core.remote`),
and every node is a Linux machine. Until 2026-09-20 every one of those
scripts was PowerShell: the fleet was three Windows laptops and a Windows hub.
diphtheria (Ubuntu 24.04, the corvis Docker host) was the first Linux node,
and for five weeks this module was one of two dialects behind a protocol,
chosen by each node's declared platform. On 2026-10-25 the fleet took its
last Windows machine, loki, out of the roster (board task e705f08c; the
operator, that day: "dude. we cannot have tge bottlnecks. so no windows shit
then ok ?"), and the PowerShell dialect went with it, so each act here is a
function rather than a method one of two classes had to supply.

Every script is POSIX ``sh`` under ``set -eu``: a command that fails ends the
script with its status, which :mod:`fleet.core.remote` turns into
``DISPATCH_FAILED`` carrying the node's own stderr, and an unset variable is
a fault rather than an empty string.

HOW THE BUILD OUTLIVES THE CONNECTION, INSIDE DISPATCH'S SHARE. sshd on Linux
sends SIGHUP to the session's process group when the connection closes, so
the build is started as a TRANSIENT SYSTEM UNIT: owned by the machine, not by
the connection, stoppable by name, and started or refused synchronously. It
starts through MCPs' fleet-check-unit (:mod:`fleet.contracts.check_unit`,
MCPs board task bc8c445c), as the login account, in the checks' slice inside
``system-dispatch.slice``, so the roster's dispatch share holds its CPU and
its disk against the owner's production. Until then it was a unit of the
account's own manager (``systemd-run --user``) under ``user.slice``, which
neither share reached.

WHERE POETRY LIVES. pipx and poetry's own installer both put the executable
under ``~/.local/bin``, which a login shell adds to PATH and a non-interactive
ssh command does not (Ubuntu's ``~/.profile`` is read by login shells only).
Every script here prepends it, so the same ``poetry`` the operator ran by
hand is the one the build finds.

WHY POETRY'S KEYRING IS OFF. Every build exports ``POETRY_KEYRING_ENABLED``
false (:data:`fleet.core.names.POETRY_KEYRING_OFF` says why): left on, poetry
waits on a D-Bus SecretService a headless node never answers.

A COMMAND IS AN ARGUMENT VECTOR, NOT A LINE, since MCPs board task d69786fa
(2026-09-27): :func:`checked_script` quotes each word with
:func:`shlex.quote`, so a caller never assembles a shell line itself.
"""

from __future__ import annotations

import posixpath
import shlex

from fleet.contracts.action_cache import STATUS_FRESH_MINUTES, STATUS_PATH
from fleet.contracts.check_unit import CHECK_UNIT
from fleet.contracts.project import MAKE_TARGET
from fleet.contracts.runner_slice import CI_SLICE_NAME
from fleet.contracts.source import InstallStep
from fleet.core import names
from fleet.core.agent_label import AGENT_LABEL_VARIABLE, require_agent_label
from fleet.core.ci_load import CI_CPU_STAT, sh_ci_load_end_lines, sh_ci_load_start_lines
from fleet.core.linux_capacity_probe import CAPACITY_PROBE_BODY
from fleet.core.linux_isolated_build import EXEC_HOME_LINE, exec_root, isolated_build_lines
from fleet.core.linux_loopback import BUILD_EXPORTS, recipe_command
from fleet.core.linux_observe_sessions import OBSERVE_SESSIONS_PYTHON
from fleet.core.linux_toolchain_probe import TOOLCHAIN_PROBE_BODY
from fleet.core.linux_unit_end import unit_end_path, unit_end_script
from fleet.core.names import (
    CACHE_VARIABLE,
    POETRY_IN_PROJECT,
    POETRY_KEYRING_OFF,
    REGISTRY_VARIABLE,
    WORKSPACE_VARIABLE,
)
from fleet.core.phase_markers import sh_phase_lines
from fleet.core.tool_caches import sh_build_lines, tool_caches_directory

#: How a script file is run by path; the path is appended.
SH_INVOCATION = ("/bin/sh",)

#: The quoted heredoc's delimiter that carries the unit-end script inside the
#: launch script; the unit-end script never holds a line equal to it.
UNIT_END_DELIMITER = "FLEET_UNIT_END_SCRIPT"

#: How a script's body is written on the far side.
#:
#: One argument for the remote login shell, the parent created in the same
#: round trip, the body streamed from standard input so no shell between here
#: and the disk interprets it. The path is interpolated inside single quotes,
#: which ``sh`` reads literally; a path with a single quote in it cannot be
#: staged, and stage roots and run ids carry none by construction.
WRITE_COMMAND = "mkdir -p \"$(dirname '{path}')\" && cat > '{path}'"

#: The first lines of every script: fail fast, and find the user's tools.
#: ``~/.cargo/bin`` is where rustup puts cargo at user scope, and it is added
#: for the same reason as ``~/.local/bin``: the probe that measures a node's
#: ``rust`` declaration and the build that compiles a crate must find the
#: one cargo the operator installed (MCPs board task 1e2da299).
PROLOGUE = 'set -eu\nPATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"\nexport PATH\n'

#: The capacity probe, verbatim. Nothing is substituted into it. It prints
#: ``key=value`` lines, one per field.
#:
#: ``MemAvailable`` rather than ``MemFree``: the kernel's own estimate of
#: what a new process can take without swapping, which counts reclaimable
#: cache, net of what the node's memory-capped containers may still grow into
#: (:mod:`fleet.core.linux_capacity_probe` says why, MCPs board task
#: a282400d). ``df`` reads the root filesystem, the disk the stage root lives
#: on for every node declared so far, in kibibytes divided to gibibytes with
#: three decimals.
#:
#: THE ``ci_slice_*`` PAIR is printed only on a node whose CI runners run in
#: ``runners.slice`` and whose ``memory.high`` is a number rather than
#: ``max`` (MCPs board task 5d6e57e7): a refusal on such a node can then say
#: CI holds the lane. Measured on lavender-wsl, 2026-09-29 18:2xZ:
#: memory.current 17179713536 against memory.high 17179869184, with 8.2 GB
#: available; diphtheria has no such cgroup and prints neither.
#:
#: THE ``action_cache_*`` FOUR are printed only on a runner host whose root
#: timer has written its counts within :data:`STATUS_FRESH_MINUTES` (MCPs
#: board task ce2b92e0, :mod:`fleet.contracts.action_cache`), and only lines
#: of that exact shape are passed on, so nothing else in the file can stand
#: in for a reading this probe takes itself.
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
    f"c={STATUS_PATH}\n"
    f'if [ -n "$(find "$c" -mmin -{STATUS_FRESH_MINUTES} 2>/dev/null)" ]; then\n'
    # A file holding no such line prints nothing, which the parser reads as
    # no counts; grep's exit 1 for no match must not end the probe as failed.
    "  grep -E '^action_cache_(runners|reading|served|missed)=[0-9]+$' \"$c\" || true\n"
    "fi\n"
)

#: The toolchain probe, verbatim: the prologue, then
#: :data:`fleet.core.linux_toolchain_probe.TOOLCHAIN_PROBE_BODY`, which says
#: what each ``name=yes|no=version`` line asks.
TOOLCHAIN_PROBE_SCRIPT = PROLOGUE + TOOLCHAIN_PROBE_BODY


#: The session-observer script, verbatim: the prologue, then
#: :data:`fleet.core.linux_observe_sessions.OBSERVE_SESSIONS_PYTHON` fed to
#: ``python3`` through a quoted heredoc, so nothing in the body is expanded
#: by ``sh`` on the way. It prints ONE compact JSON document: ``platform`` in
#: the harness's own spelling, ``hostname`` lowercased, and ``records``, every
#: ``~/.claude/sessions/*.json`` verbatim.
OBSERVE_SESSIONS_SCRIPT = f"{PROLOGUE}python3 - <<'PY'\n{OBSERVE_SESSIONS_PYTHON}PY\n"


def script_path(directory: str, stem: str) -> str:
    """Name a ``.sh`` under a remote directory.

    Args:
        directory: Absolute remote directory.
        stem: The script's name without its extension.

    Returns:
        The absolute path.
    """
    return f"{directory}/{stem}.sh"


def fleet_directory(user: str) -> str:
    """The provisioned account's ``.fleet`` directory.

    Where a script that belongs to no dispatch lands: the session observer's,
    and anything else sent to the account rather than into a staged tree.

    Args:
        user: The provisioned account.

    Returns:
        ``/home/<user>/.fleet``, literal and absolute, the path
        :data:`WRITE_COMMAND` single-quotes: a ``~`` inside those quotes
        would be a directory called ``~``.
    """
    return f"/home/{user}/.fleet"


def write_command(path: str) -> str:
    """The remote command that writes standard input to a path.

    Args:
        path: Absolute remote path; its parent is created.

    Returns:
        :data:`WRITE_COMMAND` with the path in place.
    """
    return WRITE_COMMAND.format(path=path)


def echo_command(text: str) -> str:
    """One line that prints a literal.

    Args:
        text: The literal, which contains no quote characters.

    Returns:
        A ``printf`` of it.
    """
    return f"printf '%s\\n' '{text}'"


def checked_script(commands: tuple[tuple[str, ...], ...]) -> str:
    """Render commands as a script that ends at the first failure.

    Nothing is added per command: :data:`PROLOGUE`'s ``set -e`` already ends
    the script at the first non-zero status and ``sh`` exits with it. Each
    word is quoted by :func:`shlex.quote`, which leaves a plain path bare and
    single-quotes anything else.

    Args:
        commands: Commands in order, each an argument vector whose first word
            is the tool.

    Returns:
        The script's text.
    """
    lines = (" ".join(shlex.quote(word) for word in command) for command in commands)
    return PROLOGUE + "".join(f"{line}\n" for line in lines)


def make_directory_script(target: str) -> str:
    """The script that creates a dispatch's directory.

    Args:
        target: Absolute remote directory for this dispatch.

    Returns:
        ``mkdir -p`` of it.
    """
    return checked_script((("mkdir", "-p", target),))


def reset_directory_script(target: str) -> str:
    """The script that empties a companion's directory and creates it.

    For the companion trees, which land at a path derived from a declared
    name rather than from a run id, so one node's directory is written by
    every run that carries that companion. Extracting over what the last run
    left would leave a file the workspace has since deleted sitting in a tree
    that is then committed AS the workspace.

    Args:
        target: Absolute remote directory to replace.

    Returns:
        The script's text. ``rm -rf`` needs no guard for an absent path,
        which is the first run on a node, and ``-f`` removes the read-only
        loose objects of the git repository a previous run made there.
    """
    return checked_script((("rm", "-rf", target), ("mkdir", "-p", target)))


def retire_script(*, target: str, retained: str, scripts: tuple[str, ...], isolated: bool) -> str:
    """Keep a settled run's transcript, then remove its directories and scripts.

    Run once the runner has read the transcript's tail, when a run has
    finished, been stopped past its lease or been cancelled (MCPs board task
    bfca20e6): nothing reads the export again, and a stage root that kept
    every one filled diphtheria's disk. The run's transient unit needs no
    removal: the launch starts it with ``--collect``, so the manager unloads
    it when the build exits, failed or not.

    Args:
        target: The dispatch's absolute remote directory; its staging
            directory beside it (:func:`fleet.core.names.staging_directory`)
            goes with it.
        retained: Where its transcript is kept, under the stage root's logs
            directory, which is created if absent.
        scripts: The scripts it left under the stage root, this one included.
        isolated: Whether the node runs docker projects' builds as
            execdocker; the run's copy in that user's home then goes too, as
            root, which such a build removes itself only when it runs to its
            end (:mod:`fleet.core.linux_isolated_build`).

    Returns:
        The script's text. ``mv`` is guarded because a run cancelled before
        its build started wrote no transcript; ``rm -rf`` and ``rm -f`` need
        no guard for what is already gone, and ``-f`` removes the read-only
        objects of the repository staged there. ``-f`` does not remove an
        entry of a directory without its write bit, so each of the run's own
        directories lacking it gets it back first (MCPs board task 05dd6588):
        since MCPs f27b4822e seals release payloads read-only, every MCPs
        packages/maketools run leaves fixture releases sealed under its
        ``.pytest-tmp``, and from 2026-10-23T23:26Z to 2026-10-24T03:51Z the
        retires of 15 such runs on six nodes stopped at ``rm -rf``, stranding
        them. ``find`` follows no link, so nothing outside the two trees is
        unlocked. Every step tolerates what an earlier attempt already did,
        so a retire that failed part way succeeds when it is run again.
    """
    log = shlex.quote(names.log_path(target))
    trees = f"{shlex.quote(target)} {shlex.quote(names.staging_directory(target))}"
    lines = [
        f"mkdir -p {shlex.quote(posixpath.dirname(retained))}",
        f"if [ -e {log} ]; then mv -f {log} {shlex.quote(retained)}; fi",
        f"for tree in {trees}; do",
        '  if [ -d "$tree" ]; then find "$tree" -type d ! -perm -u=w -exec chmod u+w {} +; fi',
        "done",
        f"rm -rf {trees}",
        "rm -f " + " ".join(shlex.quote(script) for script in scripts),
    ]
    if isolated:
        run = posixpath.basename(target.rstrip("/"))
        copy = exec_root(run)
        lines += [EXEC_HOME_LINE, f'sudo -n rm -rf "{copy}" "{names.staging_directory(copy)}"']
    return PROLOGUE + "".join(f"{line}\n" for line in lines)


def digest_script(target: str) -> str:
    """Print the landed archive's SHA-256, extracting nothing.

    It must NOT extract: the sender compares the digest first.

    Args:
        target: Absolute remote directory for this dispatch.

    Returns:
        The script's text. ``sha256sum`` prints the digest and the file name;
        only the digest is kept, so the output is exactly what the sender
        compares.
    """
    return f"{PROLOGUE}sha256sum '{target}/{names.ARCHIVE_NAME}' | cut -d ' ' -f 1\n"


def build_script(
    *,
    target: str,
    path: str,
    workers: int,
    install: tuple[InstallStep, ...],
    cache_root: str,
    isolated_docker: bool,
    agent: str,
    loopback_unit: str | None,
) -> str:
    """Ready the tree, run the recipe in the project, write its status last.

    Deliberately NOT under ``set -e`` for the install steps or the recipe
    itself: a failing suite is a result to record, not a fault in the script,
    so each status is captured and written rather than ending the script
    before the write. Everything else is still fail-fast. Each step, and the
    recipe as the phase ``check``, is timed (:mod:`fleet.core.phase_markers`).

    A docker project's build is :mod:`fleet.core.linux_isolated_build`'s
    instead: the same steps in the same order, run as execdocker against its
    rootless daemon, with that user's caches rather than the node's.

    A loopback-only project's install steps run as every project's do, in the
    host's network, and its recipe through :func:`recipe_command`, in a
    network namespace holding only lo.

    Args:
        target: Absolute remote directory holding the export, its root.
        path: The project's directory inside the export, ``""`` for the root.
        workers: Test workers the capacity check granted.
        install: The project's declared install steps, each an argv in the
            source grammar timed as the phase it names, run at the root
            before the recipe.
        cache_root: The node's cache directory, whose current generation npm,
            poetry and playwright are pointed at (:mod:`fleet.core.tool_caches`);
            unread by a docker project's build, whose caches are execdocker's.
        isolated_docker: True for a project that declares the ``docker`` tag.
        agent: The job's submitting agent label, exported as
            ``BOARD_AGENT_LABEL`` so a hold the suite takes is attributed to
            whoever asked for the run (:mod:`fleet.core.agent_label`).
        loopback_unit: The check's own unit, ``fleet-<run id>``, for a project
            that declares ``loopback_only``, whose ``make check`` then runs
            through ``fleet-check-unit loopback`` in a network namespace
            holding only lo (:mod:`fleet.contracts.check_unit`); None for
            every other project.

    Returns:
        The script's text. Its last act writes the recipe's exit status to
        the result file, so the file's absence means running.

    Raises:
        ValueError: When ``isolated_docker`` and ``loopback_unit`` are both
            set, because a docker project's build runs as execdocker, which
            holds no sudo to call ``fleet-check-unit``; or when ``agent`` is
            outside the board's label grammar.
    """
    if isolated_docker and loopback_unit is not None:
        raise ValueError(
            f"the project at {path!r} declares both the docker tag and loopback_only, and "
            "its build runs as execdocker, which holds no sudo to start a check in a "
            "network namespace of its own; declare one or the other"
        )
    label = require_agent_label(agent)
    if isolated_docker:
        lines = isolated_build_lines(
            target=target, path=path, workers=workers, install=install, agent=label
        )
        return PROLOGUE + "\n".join(lines) + "\n"
    log = names.log_path(target)
    result = f"{target}/{names.RESULT_NAME}"
    written = f"printf '%s\\n' \"$status\" > '{result}'"
    lines = [
        *sh_build_lines(
            tools=shlex.quote(tool_caches_directory(cache_root)),
            marker=shlex.quote(f"{target}/{names.GENERATION_MARKER_NAME}"),
            run_as="",
        ),
        'npm_config_cache="$tool_cache/npm"',
        'POETRY_CACHE_DIR="$tool_cache/pypoetry"',
        f"{POETRY_IN_PROJECT[0]}='{POETRY_IN_PROJECT[1]}'",
        f"{POETRY_KEYRING_OFF[0]}='{POETRY_KEYRING_OFF[1]}'",
        'PLAYWRIGHT_BROWSERS_PATH="$tool_cache/ms-playwright"',
        f"PYTEST_XDIST_AUTO_NUM_WORKERS='{workers}'",
        f"CORVIS_TEST_MAX_WORKERS='{workers}'",
        f"{AGENT_LABEL_VARIABLE}='{label}'",
        f"{CACHE_VARIABLE}='{cache_root}'",
        f"{WORKSPACE_VARIABLE}={shlex.quote(path or '.')}",
        f"{REGISTRY_VARIABLE}='{names.registry_directory(target)}'",
        f"export {' '.join(BUILD_EXPORTS)}",
        f"cd '{target}'",
    ]
    for step in install:
        command = " ".join(step["argv"])
        lines.append(f"printf '$ %s\\n' '{command}' >> '{log}'")
        lines.extend(sh_phase_lines(phase=step["phase"], command=command, log=log))
        lines.append(f'if [ "$status" -ne 0 ]; then {written}; exit 0; fi')
    lines.append(f"cd '{names.recipe_directory(target, path)}'")
    lines.extend(sh_ci_load_start_lines(CI_CPU_STAT))
    lines.extend(sh_phase_lines(phase=MAKE_TARGET, command=recipe_command(loopback_unit), log=log))
    lines.extend(sh_ci_load_end_lines(CI_CPU_STAT, log))
    lines.append(written)
    return PROLOGUE + "\n".join(lines) + "\n"


def log_tail_script(target: str, lines: int) -> str:
    """Print the last lines of the build's transcript, or nothing.

    Args:
        target: Absolute remote directory holding the export.
        lines: How many lines from the end.

    Returns:
        The script's text; an absent transcript prints nothing.
    """
    log = names.log_path(target)
    return f"{PROLOGUE}if [ -f '{log}' ]; then tail -n {lines} '{log}'; fi\n"


def launch_script(*, target: str, run_id: str) -> str:
    """Start the build as a transient system unit in dispatch's share, and prove it began.

    The helper's ``start`` (:mod:`fleet.contracts.check_unit`) runs
    ``systemd-run --collect --quiet`` and returns once systemd has accepted
    and started the unit, exiting non-zero when either refuses, so the
    ``launched`` line after it proves the build began. The lingering check
    comes first and names the fix: the unit is given the account's runtime
    directory and session bus, which the account's own manager keeps only
    while it lingers, and a check that asks ``systemctl --user`` would
    otherwise lose them with this ssh session.

    The unit's ``ExecStopPost=`` runs the unit-end script, which records how
    the unit ended whenever the build did not write its own status
    (:mod:`fleet.core.linux_unit_end`, MCPs board task c8585623). The script
    is written by a quoted heredoc, so nothing in it is expanded on the way
    to the disk.

    Args:
        target: Absolute remote directory holding the staged tree.
        run_id: The dispatch, which names its own unit.

    Returns:
        The script's text. It prints ``launched`` only once the build has
        actually begun, and exits non-zero otherwise.
    """
    unit = names.task_name(run_id)
    build = f"{target}/{names.BUILD_STEM}.sh"
    ended = unit_end_path(target)
    return (
        f'{PROLOGUE}user="$(id -un)"\n'
        f'if [ "$(loginctl show-user "$user" -p Linger --value)" != yes ]; then\n'
        f"  printf '%s\\n' \"lingering is off for $user, so the runtime directory and bus "
        f"a check is given would end with this ssh session; enable it once with: "
        f'sudo loginctl enable-linger $user" >&2\n'
        f"  exit 1\n"
        f"fi\n"
        f"cat > '{ended}' <<'{UNIT_END_DELIMITER}'\n"
        f"{unit_end_script(target=target, unit=unit, prologue=PROLOGUE)}"
        f"{UNIT_END_DELIMITER}\n"
        f"{CHECK_UNIT} start '{unit}' '{target}' '{build}' '{ended}'\n"
        f"printf 'launched\\n'\n"
    )


#: What the result script prints for a directory no launch wrote into.
UNLAUNCHED_ANSWER = "unlaunched"


def result_script(target: str) -> str:
    """Print the status and when it was written, nothing, or :data:`UNLAUNCHED_ANSWER`.

    A build that ended without writing its status has had it written by its
    unit's ``ExecStopPost=`` (:mod:`fleet.core.linux_unit_end`) by the time
    systemd considers the unit stopped, so this reads the result file alone,
    and nothing while the run is still going. The launch writes the unit-end
    script before it starts the unit and the ledger row is written after, so
    a live run's directory without it was launched from another stage root
    (MCPs board task f02b144e); sending this script made the directory.

    Args:
        target: Absolute remote directory holding the staged tree.

    Returns:
        The script's text. ``stat -c %Y`` is the file's modification time in
        epoch seconds, UTC by definition, so no offset arithmetic is needed.
    """
    result = f"{target}/{names.RESULT_NAME}"
    return (
        f"{PROLOGUE}if [ ! -f '{unit_end_path(target)}' ]; then\n"
        f"  printf '%s\\n' '{UNLAUNCHED_ANSWER}'\n"
        f"elif [ -f '{result}' ]; then\n"
        f"  code=\"$(tr -d '[:space:]' < '{result}')\"\n"
        f"  written=\"$(stat -c %Y '{result}')\"\n"
        f'  printf \'%s %s\\n\' "$code" "$written"\n'
        f"fi\n"
    )


def stop_script(run_id: str) -> str:
    """Stop the unit if it is still active, and say so either way.

    Stopping a unit stops its control group, every process the build
    started, so the unit's name is all this needs.

    Args:
        run_id: The dispatch, which names its own unit.

    Returns:
        The script's text. The helper's ``stop``
        (:mod:`fleet.contracts.check_unit`) stops the unit only while it is
        active and exits 0 otherwise, because ``systemctl stop`` of a unit
        that has already been collected exits 5 (not loaded) and the cancel
        must close the row whether the build was still going or not.
    """
    unit = names.task_name(run_id)
    return f"{PROLOGUE}{CHECK_UNIT} stop '{unit}'\nprintf 'stopped {unit}\\n'\n"


def extract_commands(archive: str, destination: str) -> tuple[tuple[str, ...], ...]:
    """The command that unpacks a verified archive.

    ``-m`` makes extracted files take the node's clock rather than the
    sender's. Without it a tree staged from a machine whose clock is ahead
    produces make targets that look newer than their sources, and the build
    does nothing at all -- which reads as a suite that passed instantly.

    THE TWO PATHS ARE SEPARATE so a payload can stay out of the tree it
    builds; a project's dispatch passes its own directory for both, which is
    where its archive already sits. (A companion, which used this too until
    MCPs board task 2026dfbc, is now cloned from a bundle instead:
    :func:`fleet.core.stage_repository.companion_repository_commands`.)

    Args:
        archive: Absolute path to the ``.tgz`` on the node.
        destination: Absolute remote directory to unpack it into.

    Returns:
        The one command, for :func:`checked_script`.
    """
    return (("tar", "-xzmf", archive, "-C", destination),)


__all__ = [
    "CAPACITY_PROBE_SCRIPT",
    "OBSERVE_SESSIONS_SCRIPT",
    "PROLOGUE",
    "SH_INVOCATION",
    "TOOLCHAIN_PROBE_SCRIPT",
    "UNIT_END_DELIMITER",
    "UNLAUNCHED_ANSWER",
    "WRITE_COMMAND",
    "build_script",
    "checked_script",
    "digest_script",
    "echo_command",
    "extract_commands",
    "fleet_directory",
    "launch_script",
    "log_tail_script",
    "make_directory_script",
    "reset_directory_script",
    "result_script",
    "retire_script",
    "script_path",
    "stop_script",
    "write_command",
]
