"""Scoring a CI host against the runner roster.

One rendered bash script per host, sent to the host's ``payload_dir`` and run
there as root (:func:`fleet.core.remote.attempt_root_script`), under the same
rule as every other remote act in this package: the bytes that run are the
bytes that were sent. The script emits exactly one ``CHECK <id> OK`` or
``CHECK <id> DRIFT <detail>`` line per declared item, in declaration order,
and exits 0 -- drift is data, not a command failure, so a host with three
problems reports three lines rather than dying on the first. It runs without
``set -e`` for that reason and with ``set -u``, so a row that names a
variable nobody set ends the script, and the transcript, early.

THE TRANSCRIPT IS VALIDATED AGAINST THE ROSTER, NOT JUST PARSED. The script
promises one line per expected check in a known order; a transcript with a
missing, duplicated, reordered or malformed line means the script died midway
or the transport mangled it, and scoring what remains would let a crashed
audit read as a clean host. That is ``RUNNER_AUDIT_UNPARSABLE``, raised, not
reported -- an audit that could not complete has established nothing.

AS ROOT, BECAUSE THE ROWS READ WHAT ONLY ROOT MAY: the runner account's
cache, every install's ``_work`` tree and the helpers under
``/usr/local/sbin``. A row about what the runner account itself can do (an
asset it must write) asks as that account.
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import remote
from fleet.core.runner_audit_cache import cache_rows, render_cache_check_lines
from fleet.core.runner_audit_disk import disk_check_id, render_disk_check_lines
from fleet.core.runner_audit_types import AuditOutcome, ExpectedCheck
from fleet.core.runner_dropped_check import attempt_dropped_check
from fleet.core.runner_install import RUNNER_ACCOUNT, install_root_for, payload_path
from fleet.core.runner_job_scope_render import (
    PLACEMENT_REASON,
    placement_check_id,
    render_placement_check_lines,
)
from fleet.core.runner_orphan_check import (
    ORPHAN_REASON,
    orphan_check_id,
    render_orphan_check_lines,
)
from fleet.core.runner_transcript import score_transcript
from fleet.core.script_values import scriptable

#: The file-name stem the rendered audit script lands under in ``payload_dir``.
AUDIT_SCRIPT_STEM = "fleet-runner-audit"

#: How long the audit may run. Its slowest rows are the ``du`` of each
#: install's ``_work`` tree, measured at a few seconds each on lavender.
AUDIT_TIMEOUT_SECONDS = 300

#: The functions every row calls, defined once at the top of the script.
#:
#: ``probe`` runs one command with its stderr folded in and keeps its words,
#: its first line and its exit. ``check`` prints the row: OK, or DRIFT with
#: the detail flattened onto one line, so the transcript stays one row per
#: line. ``verdict_of`` turns a command's success into ``ok`` or ``drift``.
#: ``within`` holds a whole-GB figure to a ceiling, false when the figure is
#: empty or the probe failed. ``gb_at`` reads line N of the probe's words as
#: whole GB and prints nothing unless the line IS a size, so a probe that
#: answered with an error can never read as zero GB.
AUDIT_FUNCTIONS: tuple[str, ...] = (
    "probe() {",
    '    PROBE_TEXT="$("$@" 2>&1)"',
    "    PROBE_EXIT=$?",
    "    PROBE_FIRST=\"${PROBE_TEXT%%$'\\n'*}\"",
    '    PROBE_TEXT="$PROBE_TEXT (exit $PROBE_EXIT)"',
    "}",
    "check() {",
    '    if [ "$2" = ok ]; then',
    "        printf 'CHECK %s OK\\n' \"$1\"",
    "    else",
    "        printf 'CHECK %s DRIFT %s\\n' \"$1\" \"$(printf '%s' \"$3\" | tr '\\r\\n' '  ')\"",
    "    fi",
    "}",
    "verdict_of() {",
    '    if "$@"; then',
    "        echo ok",
    "    else",
    "        echo drift",
    "    fi",
    "}",
    "within() {",
    '    [ "$PROBE_EXIT" -eq 0 ] && [ -n "$1" ] && [ "$1" -le "$2" ]',
    "}",
    "answered() {",
    '    [ "$PROBE_EXIT" -eq 0 ] && [ "$PROBE_FIRST" = "$1" ]',
    "}",
    "said_something() {",
    '    [ "$PROBE_EXIT" -eq 0 ] && [ -n "$PROBE_FIRST" ]',
    "}",
    "starts_with() {",
    '    [ "$PROBE_EXIT" -eq 0 ] && case "$PROBE_FIRST" in "$1"*) true ;; *) false ;; esac',
    "}",
    "succeeded() {",
    '    [ "$PROBE_EXIT" -eq 0 ]',
    "}",
    "gb_at() {",
    "    printf '%s\\n' \"$PROBE_TEXT\" | sed -n \"$1\"'s/^[[:space:]]*\\([0-9][0-9]*\\)G.*/\\1/p'",
    "}",
)

#: Why the memory row exists, printed beside its drift line.
MEMORY_REASON = (
    "the CI slice's ceiling is held below the memory the roster declares, so a host with "
    "less than that gives the runners what sshd and the fleet's own checks need"
)


def memory_check_id(spec: HostRunnerSpec) -> str:
    """The memory row's id, which carries the declared size.

    Args:
        spec: The host's roster entry.

    Returns:
        E.g. ``memory:31gb``.
    """
    return f"memory:{spec['memory_gb']}gb"


def _install_ids(install: RunnerInstall) -> str:
    """The part of a per-install row's id that names the install.

    Args:
        install: The install.

    Returns:
        ``<repo>:<runner_name>``, unique across repositories, since two
        installs on one host may share a runner name across repos.
    """
    return f"{install['repo']}:{install['runner_name']}"


def expected_checks(spec: HostRunnerSpec) -> list[ExpectedCheck]:
    """Every check the audit script for this host will report, in order.

    The single source of both the script's ``check`` lines and the transcript
    validator, so the two cannot disagree about what a complete audit is.

    Args:
        spec: The host's roster entry.

    Returns:
        One entry per check, in the order the script emits them.
    """
    checks = [
        ExpectedCheck(check_id=memory_check_id(spec), reason=MEMORY_REASON),
        ExpectedCheck(
            check_id=disk_check_id(spec),
            reason="a runner host holds nothing that needs keeping, so its root stays near "
            "the idle baseline; growth past the ceiling is a leak, as on 2026-09-25",
        ),
    ]
    checks.extend(
        ExpectedCheck(check_id=row["check_id"], reason=row["reason"]) for row in cache_rows(spec)
    )
    if spec["gpu_required"]:
        # One per runner, on that runner's own PATH: the jobs that digest the
        # card are the runner's, and a host-level check from an interactive
        # shell passed on the rebuilt lavender while its runners could not
        # find nvidia-smi (board task 1aa6a021).
        checks.extend(
            ExpectedCheck(
                check_id=f"gpu:{_install_ids(install)}",
                reason="runner jobs on this host digest a real GPU",
            )
            for install in spec["installs"]
        )
    checks.extend(
        ExpectedCheck(check_id=f"timer:{timer}", reason="host hygiene is part of the provision")
        for timer in spec["systemd_timers"]
    )
    for install in spec["installs"]:
        checks += [
            ExpectedCheck(
                check_id=f"service:{install['service']}",
                reason=f"runs the {install['repo']} runner {install['runner_name']}",
            ),
            ExpectedCheck(
                check_id=f"workdir:{_install_ids(install)}",
                reason="the install's _work tree, which its venvs are path-bound to",
            ),
            ExpectedCheck(check_id=placement_check_id(install), reason=PLACEMENT_REASON),
            ExpectedCheck(check_id=orphan_check_id(install), reason=ORPHAN_REASON),
        ]
    for asset in spec["assets"]:
        checks.append(ExpectedCheck(check_id=f"asset:{asset['path']}", reason=asset["reason"]))
        if asset["sha256"] is not None:
            checks.append(ExpectedCheck(check_id=f"sha256:{asset['path']}", reason=asset["reason"]))
        if asset["writable"]:
            checks.append(
                ExpectedCheck(check_id=f"writable:{asset['path']}", reason=asset["reason"])
            )
    return checks


def _memory_lines(spec: HostRunnerSpec) -> list[str]:
    """Audit lines that hold the host's memory to the declared size.

    The kernel reports a little less than the installed memory (the firmware
    and the kernel keep some), so the row asks for the declared size less one
    GB, read from ``free -g``'s ``Mem:`` line.

    Args:
        spec: The host.

    Returns:
        The bash lines.
    """
    floor = spec["memory_gb"] - 1
    return [
        "probe free -g",
        "total=$(printf '%s\\n' \"$PROBE_TEXT\" | "
        "sed -n 's/^Mem:[[:space:]]*\\([0-9][0-9]*\\).*/\\1/p')",
        f'check \'{memory_check_id(spec)}\' "$(verdict_of [ "${{total:-0}}" -ge {floor} ])" '
        f'"the host reports ${{total:-an unreadable figure}} GB, below the {floor} GB a '
        f'{spec["memory_gb"]} GB host reports; free said: $PROBE_TEXT"',
    ]


def _gpu_lines(spec: HostRunnerSpec) -> list[str]:
    """Audit lines that ask each runner's own PATH for the GPU.

    The row asks for exit 0 as well as a name, so a probe that answered with
    an error is not read as a card (MCPs board task d69786fa).

    Args:
        spec: The host, with ``gpu_required`` set.

    Returns:
        Two lines per install: nvidia-smi run with PATH read from the
        runner's ``.path`` (the PATH runsvc.sh gives its jobs), and the
        ``check``.

    Raises:
        ValueError: When a runner directory cannot be embedded verbatim.
    """
    lines: list[str] = []
    for install in spec["installs"]:
        runner_dir = scriptable(install_root_for(install), label="workdir")
        lines += [
            f"probe env PATH=\"$(cat '{runner_dir}/.path')\" nvidia-smi --query-gpu=name "
            "--format=csv,noheader",
            f"check 'gpu:{_install_ids(install)}' \"$(verdict_of said_something)\" "
            '"nvidia-smi on the runner PATH said: $PROBE_TEXT"',
        ]
    return lines


def _install_lines(spec: HostRunnerSpec, install: RunnerInstall) -> list[str]:
    """Audit lines for one install: its service, workdir, placement and leftovers.

    Args:
        spec: The host, for its job timeout.
        install: The install.

    Returns:
        The bash lines, in :func:`expected_checks`' order.

    Raises:
        ValueError: When a roster value cannot be embedded verbatim.
    """
    service = scriptable(install["service"], label="service")
    workdir = scriptable(install["workdir"], label="workdir")
    return [
        f"probe systemctl is-active '{service}'",
        f"check 'service:{service}' \"$(verdict_of answered active)\" "
        '"systemctl is-active said: $PROBE_TEXT"',
        f"probe test -d '{workdir}'",
        f"check 'workdir:{_install_ids(install)}' \"$(verdict_of succeeded)\" "
        f'"test -d {workdir}: $PROBE_TEXT"',
        *render_placement_check_lines(install),
        *render_orphan_check_lines(spec, install),
    ]


def _asset_lines(spec: HostRunnerSpec) -> list[str]:
    """Audit lines for every asset: present, its pin, and writable by the runners.

    Args:
        spec: The host.

    Returns:
        The bash lines, in :func:`expected_checks`' order.

    Raises:
        ValueError: When a path cannot be embedded verbatim.
    """
    lines: list[str] = []
    for asset in spec["assets"]:
        path = scriptable(asset["path"], label="asset path")
        lines += [
            f"probe test -e '{path}'",
            f'check \'asset:{path}\' "$(verdict_of succeeded)" "test -e {path}: $PROBE_TEXT"',
        ]
        pin = asset["sha256"]
        if pin is not None:
            lines += [
                f"probe sha256sum '{path}'",
                f"check 'sha256:{path}' \"$(verdict_of starts_with {pin})\" "
                '"sha256sum said: $PROBE_TEXT"',
            ]
        if asset["writable"]:
            lines += [
                f"probe runuser -u {RUNNER_ACCOUNT} -- test -w '{path}'",
                f"check 'writable:{path}' \"$(verdict_of succeeded)\" "
                f'"test -w {path} as {RUNNER_ACCOUNT}: $PROBE_TEXT"',
            ]
    return lines


def render_audit_script(spec: HostRunnerSpec) -> str:
    """The bash audit for one host.

    Args:
        spec: The host's roster entry.

    Returns:
        The complete script text. Emits one ``CHECK`` line per entry of
        :func:`expected_checks`, in that order, and exits 0 -- drift is data.

    Raises:
        ValueError: When a roster value cannot be embedded verbatim; see
            :func:`fleet.core.script_values.scriptable`.
    """
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        *AUDIT_FUNCTIONS,
        *_memory_lines(spec),
        *render_disk_check_lines(spec),
        *render_cache_check_lines(spec),
    ]
    if spec["gpu_required"]:
        lines += _gpu_lines(spec)
    for timer in spec["systemd_timers"]:
        name = scriptable(timer, label="systemd timer")
        lines += [
            f"probe systemctl is-enabled '{name}'",
            f"check 'timer:{name}' \"$(verdict_of answered enabled)\" "
            '"systemctl is-enabled said: $PROBE_TEXT"',
        ]
    for install in spec["installs"]:
        lines += _install_lines(spec, install)
    lines += _asset_lines(spec)
    lines.append("exit 0")
    return "\n".join(lines) + "\n"


def attempt_audit_host(spec: HostRunnerSpec) -> AuditOutcome:
    """Audit one host, reporting unreachability as a value.

    Args:
        spec: The host's roster entry.

    Returns:
        Every check's verdict, or the reason the host could not be audited.

    Raises:
        AppError: ``RUNNER_AUDIT_UNPARSABLE`` as
            :func:`fleet.core.runner_transcript.score_transcript` describes.
            Deliberately NOT folded
            into the outcome: an unreachable host is a fleet condition to
            report beside the others, while an unscorable transcript from a
            host that answered is a fault in this tooling or its transport,
            and must stop the audit rather than print as one more line.
    """
    outcome = remote.attempt_root_script(
        spec["host"],
        payload_path(spec, AUDIT_SCRIPT_STEM),
        render_audit_script(spec),
        timeout_seconds=AUDIT_TIMEOUT_SECONDS,
    )
    failure = outcome["failure"]
    if failure is not None:
        return AuditOutcome(findings=None, reason=failure["message"])
    findings = score_transcript(spec["name"], expected_checks(spec), outcome["output"])
    dropped = attempt_dropped_check(spec)
    if dropped["findings"] is None:
        return dropped
    return AuditOutcome(findings=[*findings, *dropped["findings"]], reason="")


__all__ = [
    "AUDIT_FUNCTIONS",
    "AUDIT_SCRIPT_STEM",
    "AUDIT_TIMEOUT_SECONDS",
    "MEMORY_REASON",
    "attempt_audit_host",
    "expected_checks",
    "memory_check_id",
    "render_audit_script",
]
