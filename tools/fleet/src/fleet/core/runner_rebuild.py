"""Rebuilding a runner host from a stock Windows install, in one command.

The operator, 2026-09-26, on lavender: "it justcis a github runner". A
machine whose only job is running CI holds nothing that needs saving, so the
answer to any fault on it is to wipe it and run the recipe again (board task
1aa6a021). ``fleet-runners --rebuild --host <name>`` is that recipe. It
starts from a machine that node setup has made reachable (the corvis-stick
``fleet-node-setup`` skill: OpenSSH, Tailscale, git, the fleet user's key),
because nothing can reach a stock install before that, and it ends at an
audited runner host:

  1. the Windows base (:func:`~fleet.core.runner_base_render.render_windows_base_script`),
     and the reboot it asks for, waited out;
  2. the distro, imported from the pinned image when it is absent;
  3. ``/etc/wsl.conf``, and a restart of the distro when that changed it;
  4. the Linux base: packages, docker, the runner account;
  5. registration tokens minted with ``gh``, one per repository;
  6. the roster's ``provision.ps1`` and ``provision.sh``
     (:mod:`fleet.core.runner_render`), carrying those tokens;
  7. the audit, whose findings are the verdict.

EVERY STAGE IS IDEMPOTENT, so a rebuild cut short by anything -- a dropped
session, a download that failed its digest, a host that did not come back
from its reboot in time -- is finished by running the same command again.
Nothing here catches: each failure raises with the stage's own words, and
the stages already done stand.

WHAT A RUNNER HOST HOLDS, AND WHY NONE OF IT NEEDS SAVING (A4): the
registration tokens are minted fresh on every run and expire within the
hour; each runner's ``.runner`` and ``.credentials`` are written by
config.sh, and ``--replace`` takes back the old registration of the same
name; the ``_work`` trees, the docker image cache, the poetry venvs and the
pip and model caches are caches, recreated by the next job or by the asset
provisioning; the distro itself is imported from the pinned image. The one
exception is named in the roster and printed by this command: a ``manual``
asset such as the licensed game tree, which no script may fetch.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import (
    _test_hooks,
    remote,
    runner_audit,
    runner_base_render,
    runner_distro,
    runner_onboard,
    runner_render,
)
from fleet.core.script_values import scriptable

#: The deadline for one long stage, in seconds: the distro image is 357 MB
#: and the Linux base installs Chrome and a build toolchain, which on
#: lavender's line took about ten minutes by hand on 2026-09-26. Half an hour
#: bounds a hung download without cutting a slow honest one short.
STAGE_TIMEOUT_SECONDS = 1800

#: How long a rebooting host may take to answer ssh again, in seconds.
REBOOT_DEADLINE_SECONDS = 1200

#: How long to wait between probes of a rebooting host, in seconds.
REBOOT_POLL_SECONDS = 20

#: The script that prints the host's last boot instant, the one fact that
#: tells a host that has rebooted from one that has not gone down yet.
BOOT_INSTANT_SCRIPT = (
    "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')\n"
)


class RebuildReport(TypedDict):
    """What one rebuild did and what the audit said afterwards.

    Attributes:
        steps: One line per stage, in order, saying what it did.
        manual_steps: The assets no script may fetch, each a loud line.
        findings: The post-rebuild audit's verdicts.
    """

    steps: list[str]
    manual_steps: list[str]
    findings: list[runner_audit.AuditFinding]


def _script_path(spec: HostRunnerSpec, name: str) -> str:
    """Where a stage's script lands on the host.

    Args:
        spec: The host.
        name: The script's file name.

    Returns:
        The path under the host's ``scratch_dir``.
    """
    return f"{spec['scratch_dir']}/{name}"


def _boot_instant(spec: HostRunnerSpec) -> str | None:
    """The host's last boot instant, or None while it does not answer.

    Args:
        spec: The host.

    Returns:
        The instant as the host printed it, trimmed, or None when the host
        could not be reached.
    """
    outcome = remote.attempt_script(
        spec["host"],
        _script_path(spec, "fleet-rebuild-boot.ps1"),
        BOOT_INSTANT_SCRIPT,
        platform="windows",
    )
    if outcome["failure"] is not None:
        return None
    return outcome["output"].strip()


def reboot_and_wait(spec: HostRunnerSpec) -> str:
    """Restart the host and wait until it answers from a new boot.

    Args:
        spec: The host.

    Returns:
        The new boot instant.

    Raises:
        AppError: ``NODE_UNREACHABLE`` when the host does not answer from a
            new boot within :data:`REBOOT_DEADLINE_SECONDS`, and the remote
            layer's codes for the restart itself. A host that answers with
            its OLD boot instant has not gone down yet, and is waited on.
    """
    before = remote.run_script(
        spec["host"],
        _script_path(spec, "fleet-rebuild-boot.ps1"),
        BOOT_INSTANT_SCRIPT,
        platform="windows",
    ).strip()
    remote.run_script(
        spec["host"],
        _script_path(spec, "fleet-rebuild-restart.ps1"),
        "shutdown.exe /r /t 10 /c 'fleet-runners --rebuild'\nexit $LASTEXITCODE\n",
        platform="windows",
    )
    deadline = _test_hooks.now() + REBOOT_DEADLINE_SECONDS
    while _test_hooks.now() < deadline:
        _test_hooks.sleep(REBOOT_POLL_SECONDS)
        after = _boot_instant(spec)
        if after is not None and after != before:
            return after
    raise AppError(
        FleetErrorCode.NODE_UNREACHABLE,
        f"{spec['name']} did not answer from a new boot within {REBOOT_DEADLINE_SECONDS}s of "
        f"the rebuild's restart (last boot {before}); re-run --rebuild once it is back, and "
        "every stage already done is skipped",
    )


def _windows_base(spec: HostRunnerSpec, steps: list[str]) -> None:
    """Lay the Windows base, rebooting once when Windows asks.

    Args:
        spec: The host.
        steps: The report's step lines, appended to.

    Raises:
        AppError: ``DISPATCH_FAILED`` when the base still asks for a restart
            after the one it was given -- a second request means the first
            did not take, and looping on restarts would hide that -- and the
            remote layer's codes.
    """
    script = runner_base_render.render_windows_base_script(spec)
    path = _script_path(spec, "fleet-rebuild-windows-base.ps1")
    output = remote.run_script_within(
        spec["host"], path, script, platform="windows", timeout_seconds=STAGE_TIMEOUT_SECONDS
    )
    if runner_base_render.REBOOT_MARKER not in output:
        steps.append("windows base: in place")
        return
    boot = reboot_and_wait(spec)
    steps.append(f"windows base: laid, and the host restarted for it (booted {boot})")
    again = remote.run_script_within(
        spec["host"], path, script, platform="windows", timeout_seconds=STAGE_TIMEOUT_SECONDS
    )
    if runner_base_render.REBOOT_MARKER in again:
        raise AppError(
            FleetErrorCode.DISPATCH_FAILED,
            f"{spec['name']}'s Windows base asked for a second restart after the one it was "
            f"given; the feature or WSL install did not take. Its output: {again.strip()}",
        )


def _distro(spec: HostRunnerSpec, steps: list[str]) -> None:
    """Import the distro, apply ``wsl.conf``, and lay the Linux base.

    Args:
        spec: The host.
        steps: The report's step lines, appended to.

    Raises:
        AppError: The remote layer's codes; a stage exiting non-zero is
            ``DISPATCH_FAILED`` with its own stderr.
    """
    imported = remote.run_script_within(
        spec["host"],
        _script_path(spec, "fleet-rebuild-import.ps1"),
        runner_base_render.render_import_script(spec),
        platform="windows",
        timeout_seconds=STAGE_TIMEOUT_SECONDS,
    ).strip()
    steps.append(f"distro: {imported or 'already registered'}")
    conf = runner_distro.run_distro_script(
        spec,
        "fleet-rebuild-wslconf",
        runner_base_render.render_wslconf_script(),
        timeout_seconds=remote.SSH_TIMEOUT_SECONDS,
    )
    if runner_base_render.WSLCONF_CHANGED_MARKER in conf:
        distro = scriptable(spec["wsl_distro"], label="wsl_distro")
        remote.run_script(
            spec["host"],
            _script_path(spec, "fleet-rebuild-terminate.ps1"),
            f"wsl --terminate '{distro}'\nexit $LASTEXITCODE\n",
            platform="windows",
        )
        steps.append("wsl.conf: written, and the distro restarted into systemd")
    else:
        steps.append("wsl.conf: in place")
    base = runner_distro.run_distro_script(
        spec,
        "fleet-rebuild-linux-base",
        runner_base_render.render_linux_base_script(spec),
        timeout_seconds=STAGE_TIMEOUT_SECONDS,
    )
    steps.append(f"linux base: {base.strip().splitlines()[-1]}")


def _distinct_repos(installs: list[RunnerInstall]) -> list[str]:
    """The repositories some installs register to, each once, in roster order.

    Args:
        installs: The installs.

    Returns:
        Their repositories, first appearance kept.
    """
    repos: list[str] = []
    for install in installs:
        if install["repo"] not in repos:
            repos.append(install["repo"])
    return repos


def _provision(spec: HostRunnerSpec, steps: list[str]) -> list[str]:
    """Mint tokens and run the roster's own provision scripts with them.

    Args:
        spec: The host.
        steps: The report's step lines, appended to.

    Returns:
        The manual steps no script may perform.

    Raises:
        AppError: ``RUNNER_TOKEN_UNAVAILABLE`` from the mint, and the remote
            layer's codes.
    """
    repos = _distinct_repos(spec["installs"])
    tokens = {repo: runner_onboard.mint_registration_token(repo) for repo in repos}
    steps.append(f"tokens: minted for {len(tokens)} repositories")
    rendered = runner_render.render_provision(spec)
    windows_repos = _distinct_repos([i for i in spec["installs"] if i["side"] == "windows"])
    windows_script = "\n".join(
        [
            *(f"$env:{runner_render.token_variable(r)} = '{tokens[r]}'" for r in windows_repos),
            rendered["windows_script"],
        ]
    )
    remote.run_script_within(
        spec["host"],
        _script_path(spec, "fleet-rebuild-provision.ps1"),
        windows_script,
        platform="windows",
        timeout_seconds=STAGE_TIMEOUT_SECONDS,
    )
    steps.append(f"provision.ps1: ran for {len(windows_repos)} Windows-side repositories")
    wsl_repos = _distinct_repos([i for i in spec["installs"] if i["side"] == "wsl"])
    linux_script = "\n".join(
        [
            "#!/usr/bin/env bash",
            *(f"export {runner_render.token_variable(r)}='{tokens[r]}'" for r in wsl_repos),
            rendered["linux_script"],
        ]
    )
    runner_distro.run_distro_script(
        spec, "fleet-rebuild-provision", linux_script, timeout_seconds=STAGE_TIMEOUT_SECONDS
    )
    steps.append(f"provision.sh: ran for {len(wsl_repos)} WSL-side repositories")
    return rendered["manual_steps"]


def rebuild(spec: HostRunnerSpec) -> RebuildReport:
    """Rebuild one runner host end to end and audit it.

    Args:
        spec: The host's roster entry.

    Returns:
        What each stage did, the manual steps left, and the audit's verdicts.

    Raises:
        AppError: As each stage describes, and ``NODE_UNREACHABLE`` when the
            host cannot be audited at the end -- a rebuild whose result
            nobody could measure has not been shown to work.
    """
    steps: list[str] = []
    _windows_base(spec, steps)
    _distro(spec, steps)
    manual = _provision(spec, steps)
    outcome = runner_audit.attempt_audit_host(spec)
    findings = outcome["findings"]
    if findings is None:
        raise AppError(
            FleetErrorCode.NODE_UNREACHABLE,
            f"the post-rebuild audit could not reach {spec['name']}: {outcome['reason']}. "
            "Every stage ran; re-run the audit before trusting the host.",
        )
    return RebuildReport(steps=steps, manual_steps=manual, findings=findings)


__all__ = [
    "BOOT_INSTANT_SCRIPT",
    "REBOOT_DEADLINE_SECONDS",
    "REBOOT_POLL_SECONDS",
    "STAGE_TIMEOUT_SECONDS",
    "RebuildReport",
    "reboot_and_wait",
    "rebuild",
]
