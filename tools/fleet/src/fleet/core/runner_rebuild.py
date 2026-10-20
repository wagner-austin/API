"""Rebuilding a runner host from a fresh fleet node, in one command.

The operator, 2026-09-26, on lavender: "it justcis a github runner". A
machine whose only job is running CI holds nothing that needs saving, so the
answer to any fault on it is to wipe it and run the recipe again (board task
1aa6a021). ``fleet-runners --rebuild --host <name>`` is that recipe. It
starts from a fleet node a CORVIS stick or a netboot installed and enrolled
(Ubuntu, Tailscale, the fleet user's key and passwordless sudo), because
nothing can reach a machine before that, and it ends at an audited runner
host:

  1. the base (:func:`~fleet.core.runner_render.render_base_script`):
     packages, docker, the runner account; and the reboot a package asks
     for, the GPU driver's kernel module above all, waited out;
  2. registration tokens minted with ``gh``, one per repository;
  3. the roster's provision (:func:`~fleet.core.runner_render.render_provision_script`),
     carrying those tokens;
  4. the audit, whose findings are the verdict.

EVERY STAGE IS IDEMPOTENT, so a rebuild cut short by anything -- a dropped
session, a download that failed, a host that did not come back from its
reboot in time -- is finished by running the same command again. Nothing
here catches: each failure raises with the stage's own words, and the
stages already done stand.

WHAT A RUNNER HOST HOLDS, AND WHY NONE OF IT NEEDS SAVING (A4): the
registration tokens are minted fresh on every run and expire within the
hour; each runner's ``.runner`` and ``.credentials`` are written by
config.sh, and ``--replace`` takes back the old registration of the same
name; the ``_work`` trees, the docker image cache, the poetry venvs and the
pip and model caches are caches, recreated by the next job or by the asset
provisioning. The one exception is named in the roster and printed by this
command: a ``manual`` asset such as the licensed game tree, which no script
may fetch.
"""

from __future__ import annotations

from typing import Final

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.node import NodePlatform
from fleet.contracts.runners import HostRunnerSpec
from fleet.core import _test_hooks, remote, runner_audit, runner_render
from fleet.core.runner_audit_types import AuditFinding
from fleet.core.runner_github import mint_registration_token
from fleet.core.runner_install import payload_path

#: How long a rebooting host may take to answer ssh again, in seconds.
REBOOT_DEADLINE_SECONDS: Final = 1200

#: How long to wait between probes of a rebooting host, in seconds.
REBOOT_POLL_SECONDS: Final = 20

#: Seconds between the reboot being asked for and taken, so the ssh session
#: that asked returns before the host goes down.
REBOOT_DELAY_SECONDS: Final = 10

#: The kernel's id for the running boot, new on every boot: the one fact that
#: tells a host that has rebooted from one that has not gone down yet.
BOOT_ID_PATH: Final = "/proc/sys/kernel/random/boot_id"


class RebuildReport(TypedDict):
    """What one rebuild did and what the audit said afterwards.

    Attributes:
        steps: One line per stage, in order, saying what it did.
        manual_steps: The assets no script may fetch, each a loud line.
        findings: The post-rebuild audit's verdicts.
    """

    steps: list[str]
    manual_steps: list[str]
    findings: list[AuditFinding]


def _boot_id(spec: HostRunnerSpec) -> str | None:
    """The host's boot id, or None while it does not answer.

    Args:
        spec: The host.

    Returns:
        The id as the kernel printed it, trimmed, or None when the host
        could not be reached.
    """
    outcome = remote.attempt_ssh(spec["host"], ("cat", BOOT_ID_PATH))
    if outcome["failure"] is not None:
        return None
    return outcome["output"].strip()


def reboot_and_wait(spec: HostRunnerSpec) -> str:
    """Restart the host and wait until it answers from a new boot.

    The reboot is a transient systemd timer :data:`REBOOT_DELAY_SECONDS`
    out rather than an immediate ``systemctl reboot``, whose dropped ssh
    session would read as a failed command.

    Args:
        spec: The host.

    Returns:
        The new boot id.

    Raises:
        AppError: ``NODE_UNREACHABLE`` when the host does not answer from a
            new boot within :data:`REBOOT_DEADLINE_SECONDS`, and the remote
            layer's codes for the restart itself. A host that answers with
            its OLD boot id has not gone down yet, and is waited on.
    """
    before = remote.run_ssh(spec["host"], ("cat", BOOT_ID_PATH)).strip()
    remote.run_ssh(
        spec["host"],
        (
            *remote.ROOT_PREFIX,
            "systemd-run",
            f"--on-active={REBOOT_DELAY_SECONDS}",
            "/usr/bin/systemctl",
            "reboot",
        ),
    )
    deadline = _test_hooks.now() + REBOOT_DEADLINE_SECONDS
    while _test_hooks.now() < deadline:
        _test_hooks.sleep(REBOOT_POLL_SECONDS)
        after = _boot_id(spec)
        if after is not None and after != before:
            return after
    raise AppError(
        FleetErrorCode.NODE_UNREACHABLE,
        f"{spec['name']} did not answer from a new boot within {REBOOT_DEADLINE_SECONDS}s of "
        f"the rebuild's restart (boot {before}); re-run --rebuild once it is back, and every "
        "stage already done is skipped",
    )


def _base(spec: HostRunnerSpec, steps: list[str]) -> None:
    """Lay the base, rebooting once when a package asks.

    Args:
        spec: The host.
        steps: The report's step lines, appended to.

    Raises:
        AppError: ``DISPATCH_FAILED`` when the base still asks for a reboot
            after the one it was given -- a second request means the first
            did not take, and looping on reboots would hide that -- and the
            remote layer's codes.
    """
    path = payload_path(spec, "fleet-rebuild-base")
    script = runner_render.render_base_script(spec)
    output = remote.run_root_script(
        spec["host"],
        path,
        script,
        platform=NodePlatform.LINUX,
        timeout_seconds=runner_render.STAGE_TIMEOUT_SECONDS,
    )
    if runner_render.REBOOT_MARKER not in output:
        steps.append("base: laid, and nothing asked for a reboot")
        return
    boot = reboot_and_wait(spec)
    steps.append(f"base: laid, and the host rebooted for it (boot {boot})")
    again = remote.run_root_script(
        spec["host"],
        path,
        script,
        platform=NodePlatform.LINUX,
        timeout_seconds=runner_render.STAGE_TIMEOUT_SECONDS,
    )
    if runner_render.REBOOT_MARKER in again:
        raise AppError(
            FleetErrorCode.DISPATCH_FAILED,
            f"{spec['name']}'s base asked for a second reboot after the one it was given; "
            f"what asked for it did not take. Its output: {again.strip()}",
        )


def _provision(spec: HostRunnerSpec, steps: list[str]) -> None:
    """Mint tokens and run the roster's own provision with them.

    Args:
        spec: The host.
        steps: The report's step lines, appended to.

    Raises:
        AppError: ``RUNNER_TOKEN_UNAVAILABLE`` from the mint, and the remote
            layer's codes.
    """
    repos = sorted({install["repo"] for install in spec["installs"]})
    tokens = {repo: mint_registration_token(repo) for repo in repos}
    steps.append(f"tokens: minted for {len(tokens)} repositories")
    remote.run_root_script(
        spec["host"],
        payload_path(spec, "fleet-rebuild-provision"),
        runner_render.render_provision_script(spec, tokens),
        platform=NodePlatform.LINUX,
        timeout_seconds=runner_render.STAGE_TIMEOUT_SECONDS,
    )
    steps.append(f"provision: ran for {len(spec['installs'])} runners")


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
    _base(spec, steps)
    _provision(spec, steps)
    outcome = runner_audit.attempt_audit_host(spec)
    findings = outcome["findings"]
    if findings is None:
        raise AppError(
            FleetErrorCode.NODE_UNREACHABLE,
            f"the post-rebuild audit could not reach {spec['name']}: {outcome['reason']}. "
            "Every stage ran; re-run the audit before trusting the host.",
        )
    return RebuildReport(
        steps=steps, manual_steps=runner_render.manual_steps(spec), findings=findings
    )


__all__ = [
    "BOOT_ID_PATH",
    "REBOOT_DEADLINE_SECONDS",
    "REBOOT_DELAY_SECONDS",
    "REBOOT_POLL_SECONDS",
    "RebuildReport",
    "reboot_and_wait",
    "rebuild",
]
