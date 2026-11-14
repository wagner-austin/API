"""The registry commit a run's install steps came from, kept beside the run.

WHY (MCPs board task 1fd01f09). A node runner serves for up to
``node_serve_seconds`` from one rolled extraction, and until that task it
went on claiming after a roll with the extraction's registry: job 0fa14fab
ran ``CI_BUILD_PLAN: roots=execution``, the install steps of the registry
before roll 84bc5ff0b6a9, and its verdict line said nothing of which
registry that was, so a session read a failure of the rolled fix where there
was a run of the code before it. Now every verdict line names the commit.

WHERE IT IS KEPT. Beside the run on its node, in ``<target>/`` with the
transcript, the result and the unit-end script: the launch writes it before
it starts the unit (:func:`launch_script`), so every run that began carries
it however it ends, and the run's own directory is the one record that
outlives the serve that launched it. A run is settled by whichever serve of
its node reads its result, which after a roll is a serve on another commit,
so the commit is never read off the settling serve.

HOW IT IS READ. The tail read (:func:`log_tail_script`) prints the
transcript's last lines and then one line, ``FLEET_REGISTRY: <commit>``, so
the verdict reads it off the same answer as everything else it judges
(:func:`read_registry`); a run launched before the record, or by
``fleet-run`` from a checkout, says so rather than guessing.

The launch and the tail read live here rather than in
:mod:`fleet.core.dialect`, which they left at its line ceiling to carry this.
"""

from __future__ import annotations

import re
from typing import Final

from fleet.contracts.check_unit import CHECK_UNIT
from fleet.core import names
from fleet.core.check_results import sh_failures_line, sh_results_digest_line, sh_totals_line
from fleet.core.commit_tree import COMMIT_PATTERN
from fleet.core.dialect import PROLOGUE
from fleet.core.linux_unit_end import unit_end_path, unit_end_script
from fleet.core.run_summary import UNREAD

#: The quoted heredoc's delimiter that carries the unit-end script inside the
#: launch script; the unit-end script never holds a line equal to it.
UNIT_END_DELIMITER: Final = "FLEET_UNIT_END_SCRIPT"

#: The file beside the run that holds the commit.
REGISTRY_COMMIT_NAME: Final = "registry-commit.txt"

#: What a run launched by ``fleet-run`` records: its registry was read from
#: a checkout's working tree, which no commit names.
WORKING_TREE: Final = "working-tree"

#: The tail answer's line's opening token.
REGISTRY_MARKER: Final = "FLEET_REGISTRY"

#: The line, as :func:`log_tail_script` prints it.
REGISTRY_LINE: Final[re.Pattern[str]] = re.compile(
    rf"^{REGISTRY_MARKER}: ([0-9a-f]{{40}}|{WORKING_TREE})$", re.MULTILINE
)


def registry_commit_path(target: str) -> str:
    """Where a run's registry commit is written.

    Args:
        target: The run's absolute remote directory.

    Returns:
        ``<target>/registry-commit.txt``.
    """
    return f"{target}/{REGISTRY_COMMIT_NAME}"


def require_registry_commit(registry_commit: str) -> str:
    """Refuse a registry commit the tail read could not print back.

    Args:
        registry_commit: A full commit id, or :data:`WORKING_TREE`.

    Returns:
        It, unchanged.

    Raises:
        ValueError: When it is neither, since it is written into a shell
            line and read back by :data:`REGISTRY_LINE`.
    """
    if registry_commit != WORKING_TREE and COMMIT_PATTERN.fullmatch(registry_commit) is None:
        raise ValueError(
            f"a run's registry commit is a full lowercase commit id or {WORKING_TREE!r}, "
            f"got {registry_commit!r}"
        )
    return registry_commit


def launch_script(*, target: str, run_id: str, registry_commit: str) -> str:
    """Record the registry commit, start the build as a transient unit, prove it began.

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
    to the disk. The registry commit is written before either, so a unit
    that started has it.

    Args:
        target: Absolute remote directory holding the staged tree.
        run_id: The dispatch, which names its own unit.
        registry_commit: The rolled commit whose registry gave the build
            its install steps, or :data:`WORKING_TREE`.

    Returns:
        The script's text. It prints ``launched`` only once the build has
        actually begun, and exits non-zero otherwise.

    Raises:
        ValueError: From :func:`require_registry_commit`.
    """
    recorded = require_registry_commit(registry_commit)
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
        f"printf '%s\\n' '{recorded}' > '{registry_commit_path(target)}'\n"
        f"cat > '{ended}' <<'{UNIT_END_DELIMITER}'\n"
        f"{unit_end_script(target=target, unit=unit, prologue=PROLOGUE)}"
        f"{UNIT_END_DELIMITER}\n"
        f"{CHECK_UNIT} start '{unit}' '{target}' '{build}' '{ended}'\n"
        f"printf 'launched\\n'\n"
    )


def log_tail_script(target: str, lines: int) -> str:
    """Print the transcript's tail, its registry commit, its kept reports' digest and failures.

    Args:
        target: Absolute remote directory holding the export.
        lines: How many lines from the end.

    Returns:
        The script's text; an absent transcript prints no lines, a run
        whose launch recorded no commit prints no registry line, a run
        that kept no report prints no digest line
        (:func:`fleet.core.check_results.sh_results_digest_line`), and a
        run whose check named no failures file prints no failures line
        (:func:`fleet.core.check_results.sh_failures_line`), and one whose
        check counted no JUnit report prints no totals line
        (:func:`fleet.core.check_results.sh_totals_line`).
    """
    log = names.log_path(target)
    recorded = registry_commit_path(target)
    # An assignment, so a ``cat`` that fails ends the read under ``set -e``
    # rather than printing a line with no commit.
    return (
        f"{PROLOGUE}if [ -f '{log}' ]; then tail -n {lines} '{log}'; fi\n"
        f"if [ -f '{recorded}' ]; then commit=\"$(cat '{recorded}')\"; "
        f"printf '{REGISTRY_MARKER}: %s\\n' \"$commit\"; fi\n"
        f"{sh_results_digest_line(target)}\n"
        f"{sh_failures_line(target)}\n"
        f"{sh_totals_line(target)}\n"
    )


def read_registry(tail: str) -> str:
    """Read the run's registry commit off the tail read's answer.

    Args:
        tail: What :func:`log_tail_script` printed.

    Returns:
        The last registry line's commit or :data:`WORKING_TREE`, or
        :data:`~fleet.core.run_summary.UNREAD` for a run launched before
        the record.
    """
    last: re.Match[str] | None = None
    for match in REGISTRY_LINE.finditer(tail):
        last = match
    return UNREAD if last is None else f"{last.group(1)}"


__all__ = [
    "REGISTRY_COMMIT_NAME",
    "REGISTRY_LINE",
    "REGISTRY_MARKER",
    "UNIT_END_DELIMITER",
    "WORKING_TREE",
    "launch_script",
    "log_tail_script",
    "read_registry",
    "registry_commit_path",
    "require_registry_commit",
]
