"""Registering a NixOS runner host's runners with GitHub, in one command (MCPs board task 9aba868d).

The operator, 2026-09-26, on lavender: "it justcis a github runner". A
machine whose only job is running CI holds nothing that needs saving, so the
answer to any fault on it is to wipe it and install it again (board task
1aa6a021). On the fleet's NixOS that install is the netboot's: the host's
generation (MCPs ``fleet-mcp/nixos/modules/runner-host.nix``) carries every
runner unit with its drop-ins and every hygiene unit, script and asset,
rendered from the roster (:mod:`fleet.core.runner_host_document`). What no
generation can carry is GitHub's half: each runner's registration, which
``config.sh`` writes with a token GitHub gives for an hour.

``fleet-runners --register --host <name>`` is that half:

1. it asks the host, as root, which installs hold no ``.runner``;
2. it mints one token per repository of those installs with ``gh`` on this
   machine (:func:`fleet.core.runner_github.mint_registration_token`);
3. it writes each to :data:`~fleet.core.runner_install.TOKEN_DIRECTORY` on
   the host, root's alone, streamed over stdin
   (:func:`fleet.core.remote.send_root_file`), never into an argument or
   the store;
4. it restarts each of those units, whose start registers it with its
   repository's token, and waits until every one holds a ``.runner``;
5. it audits the host, and the findings are the verdict.

A host whose installs all hold a registration is audited alone, with no
token minted. Each step is safe to repeat: a registered install is skipped,
and ``config.sh --replace`` takes back an older registration of the same
name. Nothing here catches: a failure raises with the step's own words, and
the steps already done stand.
"""

from __future__ import annotations

from typing import Final

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import _test_hooks, remote, runner_audit
from fleet.core.runner_audit_types import AuditFinding
from fleet.core.runner_github import mint_registration_token
from fleet.core.runner_install import (
    TOKEN_DIRECTORY,
    install_root_for,
    payload_path,
    token_variable,
)
from fleet.core.script_values import scriptable

#: How long the restarted units may take to register, in seconds: each
#: start unpacks nothing new and runs config.sh once against GitHub.
REGISTER_DEADLINE_SECONDS: Final = 300

#: How long to wait between reads of which installs are still unregistered.
REGISTER_POLL_SECONDS: Final = 5

#: The probe's deadline: a stat per install.
PROBE_TIMEOUT_SECONDS: Final = 60

#: The file-name stem the probe lands under in the host's ``payload_dir``.
PROBE_STEM: Final = "fleet-register-probe"


class RegisterReport(TypedDict):
    """What one registration did and what the audit said afterwards.

    Attributes:
        steps: One line per step, in order, saying what it did.
        findings: The audit's verdicts.
    """

    steps: list[str]
    findings: list[AuditFinding]


def render_probe(spec: HostRunnerSpec) -> str:
    """The probe that names each install whose root holds no registration.

    Args:
        spec: The host.

    Returns:
        The script, run as root since the runner account's home is its own:
        it prints one install root per line.

    Raises:
        ValueError: When an install root cannot be embedded verbatim.
    """
    roots = " ".join(
        f"'{scriptable(install_root_for(install), label='install root')}'"
        for install in spec["installs"]
    )
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"for root in {roots}; do\n"
        '    [ -f "$root/.runner" ] || echo "$root"\n'
        "done\n"
    )


def unregistered(spec: HostRunnerSpec) -> list[RunnerInstall]:
    """The installs whose roots hold no registration on the host.

    Args:
        spec: The host.

    Returns:
        Those installs, in roster order.

    Raises:
        AppError: The remote layer's codes.
        ValueError: As :func:`render_probe` describes.
    """
    named = set(
        remote.run_root_script(
            spec["host"],
            payload_path(spec, PROBE_STEM),
            render_probe(spec),
            timeout_seconds=PROBE_TIMEOUT_SECONDS,
        ).splitlines()
    )
    return [install for install in spec["installs"] if install_root_for(install) in named]


def _wait_registered(spec: HostRunnerSpec, units: list[str]) -> None:
    """Wait until every install holds a registration.

    Args:
        spec: The host.
        units: The units restarted, for the refusal.

    Raises:
        AppError: ``DISPATCH_FAILED`` naming each install still unregistered
            at :data:`REGISTER_DEADLINE_SECONDS`, whose unit's journal says
            why, and the remote layer's codes.
    """
    deadline = _test_hooks.now() + REGISTER_DEADLINE_SECONDS
    while True:
        waiting = unregistered(spec)
        if not waiting:
            return
        if _test_hooks.now() >= deadline:
            names = ", ".join(install["service"] for install in waiting)
            raise AppError(
                FleetErrorCode.DISPATCH_FAILED,
                f"{spec['name']}: {names} held no registration {REGISTER_DEADLINE_SECONDS}s "
                f"after {len(units)} restart(s); journalctl -u <unit> on the host says why",
            )
        _test_hooks.sleep(REGISTER_POLL_SECONDS)


def register(spec: HostRunnerSpec) -> RegisterReport:
    """Register every unregistered runner of one host, then audit it.

    Args:
        spec: The host's roster entry.

    Returns:
        What each step did and the audit's verdicts.

    Raises:
        AppError: ``RUNNER_TOKEN_UNAVAILABLE`` from the mint,
            ``DISPATCH_FAILED`` when a runner does not register in time,
            ``NODE_UNREACHABLE`` when the host cannot be audited at the end
            -- a registration whose result nobody could measure has not been
            shown to work -- and the remote layer's codes.
        ValueError: As :func:`render_probe` describes.
    """
    steps: list[str] = []
    waiting = unregistered(spec)
    if waiting:
        repos = sorted({install["repo"] for install in waiting})
        for repo in repos:
            remote.send_root_file(
                spec["host"],
                f"{TOKEN_DIRECTORY}/{token_variable(repo)}",
                mint_registration_token(repo) + "\n",
            )
        steps.append(
            f"tokens: minted for {len(repos)} repositories and written to {TOKEN_DIRECTORY}"
        )
        units = [install["service"] for install in waiting]
        for unit in units:
            restart = ("systemctl", "restart", "--no-block", scriptable(unit, label="service"))
            remote.run_ssh(spec["host"], (*remote.ROOT_PREFIX, *restart))
        _wait_registered(spec, units)
        steps.append(f"registered: {len(units)} runner(s), {', '.join(units)}")
    else:
        count = len(spec["installs"])
        steps.append(f"registered: all {count} runner(s) already were; nothing minted")
    outcome = runner_audit.attempt_audit_host(spec)
    findings = outcome["findings"]
    if findings is None:
        raise AppError(
            FleetErrorCode.NODE_UNREACHABLE,
            f"the audit after registering could not reach {spec['name']}: {outcome['reason']}. "
            "Every step ran; re-run the audit before trusting the host.",
        )
    return RegisterReport(steps=steps, findings=findings)


__all__ = [
    "PROBE_STEM",
    "PROBE_TIMEOUT_SECONDS",
    "REGISTER_DEADLINE_SECONDS",
    "REGISTER_POLL_SECONDS",
    "RegisterReport",
    "register",
    "render_probe",
    "unregistered",
]
