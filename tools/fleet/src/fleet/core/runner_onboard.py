"""Onboarding a repository onto the self-hosted fleet, push-button.

Operator mandate, 2026-09-09, issued minutes after the last hand-rolled
runner install on this fleet (tree-bot's, which took two quoting traps and
three ssh round trips to land by hand): registering a repo's CI onto the
self-hosted runners must be ONE command that a future session cannot get
wrong. This module is that command's engine. ``fleet-runners --onboard
owner/repo --host lavender`` does, in order:

  1. REFUSES if the roster already carries installs for that repo on that
     host -- re-registering is how duplicate runners are born, and the
     roster is the memory that prevents it.
  2. MINTS the registration token itself (``gh api``, through the command
     hook) -- tokens expire hourly, so any flow that asks a human to paste
     one has already lost the idiot-proofing race.
  3. DERIVES the install entries from the fleet's own measured convention
     (one WSL install named/labelled ``lavender-wsl`` in
     ``/home/gharunner/actions-runner-<repo>-1``, one Windows install
     named/labelled ``lavender`` in ``C:/actions-runner-<repo>`` -- the
     layout every existing repo on the box already follows).
  4. PROVISIONS both sides by render-send-run: the same install renderers
     ``--render`` uses, sent as files and executed by path, because every
     inline-quoting shortcut tried on this fleet has been defeated by the
     ssh->cmd->powershell->wsl->bash gauntlet -- twice on the night this
     module was written.
  5. REWRITES the roster in the same act, so the audit and the load
     sampler cover the new installs from birth rather than after somebody
     remembers.
  6. AUDITS the host as verification and returns the findings -- onboarding
     is not done when the scripts exit; it is done when the audit says the
     services are running.

Nothing here is optional and nothing falls back: a failure at any step
raises with a typed code and the steps already taken stand (a registered
runner with no roster entry is exactly what step 5 exists to prevent, so
step 5 failing after step 4 is a loud error naming the repair, not a
shrug).
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import _test_hooks, remote, runner_audit, runner_render


class OnboardPlan(TypedDict):
    """What onboarding one repo onto one host will create.

    Attributes:
        repo: The repository, ``owner/name``.
        installs: The install entries that will be provisioned and added to
            the roster, derived from the fleet's naming convention.
    """

    repo: str
    installs: list[RunnerInstall]


def _repo_slug(repo: str) -> str:
    """The repo's name as it appears in install directory names.

    Args:
        repo: The repository, ``owner/name``.

    Returns:
        The name half, lowercased -- matching the measured convention
        (``actions-runner-treebot-1`` came from a hand-rolled variation;
        the convention this module enforces keeps the name verbatim, so
        the directory always answers "whose runner is this" by itself.
    """
    return repo.split("/")[1].lower()


def plan_onboard(spec: HostRunnerSpec, repo: str, *, sides: tuple[str, ...]) -> OnboardPlan:
    """The installs onboarding will create, per the fleet convention.

    Args:
        spec: The target host's roster entry.
        repo: The repository, ``owner/name``.
        sides: Which environments to provision, from {"wsl", "windows"}.

    Returns:
        The plan.

    Raises:
        AppError: ``RUNNER_ALREADY_ONBOARDED`` when the roster already
            carries any install for this repo on this host. The roster is
            the memory; a second registration would be a duplicate runner,
            not a refresh.
        ValueError: On an empty or unknown ``sides`` entry, or a repo that
            is not ``owner/name`` -- refused here rather than downstream so
            the message names the flag to fix.
    """
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"--onboard takes owner/repo, got {repo!r}")
    if not sides:
        raise ValueError("sides must name at least one of wsl, windows")
    for side in sides:
        if side not in ("wsl", "windows"):
            raise ValueError(f"unknown side {side!r}; sides are wsl and windows")
    existing = [i for i in spec["installs"] if i["repo"] == repo]
    if existing:
        names = ", ".join(i["runner_name"] for i in existing)
        raise AppError(
            FleetErrorCode.RUNNER_ALREADY_ONBOARDED,
            f"{repo} already has {len(existing)} install(s) on {spec['name']} ({names}); "
            "re-onboarding would register duplicate runners. If the goal is repair, "
            "run the audit; if the goal is removal, that is a deliberate act this "
            "command does not perform.",
        )
    slug = _repo_slug(repo)
    owner_dashed = repo.replace("/", "-")
    installs: list[RunnerInstall] = []
    if "wsl" in sides:
        installs.append(
            RunnerInstall(
                repo=repo,
                runner_name="lavender-wsl",
                side="wsl",
                service=f"actions.runner.{owner_dashed}.lavender-wsl.service",
                workdir=f"/home/gharunner/actions-runner-{slug}-1/_work",
                labels=["lavender-wsl"],
            )
        )
    if "windows" in sides:
        installs.append(
            RunnerInstall(
                repo=repo,
                runner_name="lavender",
                side="windows",
                service=f"actions.runner.{owner_dashed}.lavender",
                workdir=f"C:/actions-runner-{slug}/_work",
                labels=["lavender"],
            )
        )
    return OnboardPlan(repo=repo, installs=installs)


def mint_registration_token(repo: str) -> str:
    """A fresh runner registration token for one repository.

    Args:
        repo: The repository, ``owner/name``.

    Returns:
        The token.

    Raises:
        AppError: ``RUNNER_TOKEN_UNAVAILABLE`` when ``gh`` exits non-zero
            or returns nothing -- an auth or permission condition on THIS
            machine, named as such rather than surfacing later as a remote
            config.sh failure that looks like a fleet fault.
    """
    result = _test_hooks.run(
        [
            "gh",
            "api",
            "-X",
            "POST",
            f"repos/{repo}/actions/runners/registration-token",
            "-q",
            ".token",
        ]
    )
    token = result["stdout"].strip()
    if result["returncode"] != 0 or not token:
        detail = result["stderr"].strip() or "<no stderr>"
        raise AppError(
            FleetErrorCode.RUNNER_TOKEN_UNAVAILABLE,
            f"gh could not mint a registration token for {repo}: {detail}. This is "
            "local auth or repo permissions, not the fleet.",
        )
    if not token.isalnum():
        # The token is embedded in rendered scripts inside single quotes;
        # a token that is not plain alphanumeric (every real one is) would
        # either be shell-mangled or be something other than a token, and
        # both deserve a refusal that names the field rather than a remote
        # config.sh error that names nothing.
        raise AppError(
            FleetErrorCode.RUNNER_TOKEN_UNAVAILABLE,
            f"gh returned a non-alphanumeric registration token for {repo} "
            f"({len(token)} chars); refusing to embed it in a provision script.",
        )
    return token


def _windows_to_wsl_path(path: str) -> str:
    """A Windows drive path as the distro sees it.

    Args:
        path: A forward-slashed drive-letter path, e.g. ``C:/fleet/stage``.

    Returns:
        The ``/mnt/<drive>/...`` form.

    Raises:
        ValueError: On a path with no drive letter -- translating a
            relative or POSIX path would silently point the distro at the
            wrong file.
    """
    if len(path) < 3 or not path[0].isalpha() or path[1] != ":" or path[2] != "/":
        raise ValueError(f"not a forward-slashed drive path: {path!r}")
    return f"/mnt/{path[0].lower()}{path[2:]}"


def _provision_wsl(spec: HostRunnerSpec, install: RunnerInstall, token: str) -> None:
    """Provision one WSL-side install on the host.

    The bash payload is sent as a file and executed by path through a
    one-line PowerShell driver, per the remote layer's rule; the token is
    injected as the payload's own environment variable line rather than a
    command-line argument, so it never crosses a shell boundary unquoted.

    Args:
        spec: The host.
        install: The install, ``side`` ``"wsl"``.
        token: The registration token.

    Raises:
        AppError: ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            remote layer -- config.sh refusing (already configured, expired
            token) surfaces as the latter with the script's own stderr.
    """
    token_var = runner_render.token_variable(install["repo"])
    payload = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"export {token_var}='{token}'",
            *runner_render.render_wsl_install_lines(install),
        ]
    )
    windows_payload_path = f"{spec['scratch_dir']}/fleet-onboard-{_repo_slug(install['repo'])}.sh"
    remote.send_script(spec["host"], windows_payload_path, payload)
    wsl_payload_path = _windows_to_wsl_path(windows_payload_path)
    driver = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "$env:WSL_UTF8 = '1'",
            f"wsl -d '{spec['wsl_distro']}' -u root -- bash '{wsl_payload_path}'",
            "exit $LASTEXITCODE",
        ]
    )
    remote.run_script(
        spec["host"],
        f"{spec['scratch_dir']}/fleet-onboard-{_repo_slug(install['repo'])}-driver.ps1",
        driver,
    )


def _provision_windows(
    spec: HostRunnerSpec,
    install: RunnerInstall,
    token: str,
    python_versions: tuple[str, ...],
) -> None:
    """Provision one Windows-side install on the host.

    Args:
        spec: The host.
        install: The install, ``side`` ``"windows"``.
        token: The registration token.
        python_versions: Exact Python versions to seed into the install's
            tool cache, so setup-python finds instead of installing --
            empty for repos that bring no Python.

    Raises:
        AppError: ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            remote layer.
    """
    token_var = runner_render.token_variable(install["repo"])
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12",
        f"$env:{token_var} = '{token}'",
        *runner_render.render_windows_install_lines(install),
    ]
    if python_versions:
        lines += runner_render.render_windows_python_toolcache_lines(install, python_versions)
    remote.run_script(
        spec["host"],
        f"{spec['scratch_dir']}/fleet-onboard-{_repo_slug(install['repo'])}-win.ps1",
        "\n".join(lines),
    )


def onboard(
    spec: HostRunnerSpec,
    repo: str,
    *,
    sides: tuple[str, ...],
    python_versions: tuple[str, ...] = (),
) -> tuple[OnboardPlan, list[runner_audit.AuditFinding]]:
    """Onboard one repository onto one host, end to end.

    Args:
        spec: The target host's roster entry. MUTATED: the plan's installs
            are appended, so the caller's subsequent roster write and audit
            see the new state -- the whole point of onboarding being one
            act.
        repo: The repository, ``owner/name``.
        sides: Which environments to provision.
        python_versions: Exact Python versions to seed into the Windows
            install's tool cache. Required knowledge for any repo whose
            Windows jobs run setup-python -- the install path it falls back
            to needs registry rights the runner service does not have.

    Returns:
        The executed plan and the post-onboarding audit findings for the
        host, new installs included.

    Raises:
        AppError: ``RUNNER_ALREADY_ONBOARDED``,
            ``RUNNER_TOKEN_UNAVAILABLE``, ``NODE_UNREACHABLE``,
            ``DISPATCH_FAILED`` or ``RUNNER_AUDIT_UNPARSABLE`` as the
            steps describe.
        ValueError: From the planner's flag validation.
    """
    plan = plan_onboard(spec, repo, sides=sides)
    token = mint_registration_token(repo)
    for install in plan["installs"]:
        if install["side"] == "wsl":
            _provision_wsl(spec, install, token)
        else:
            _provision_windows(spec, install, token, python_versions)
    spec["installs"].extend(plan["installs"])
    outcome = runner_audit.attempt_audit_host(spec)
    findings = outcome["findings"]
    if findings is None:
        raise AppError(
            FleetErrorCode.NODE_UNREACHABLE,
            f"post-onboard audit could not reach {spec['name']}: {outcome['reason']}. "
            "The runners were provisioned and the roster entries are staged; re-run "
            "the audit before trusting either.",
        )
    return plan, findings


__all__ = [
    "OnboardPlan",
    "mint_registration_token",
    "onboard",
    "plan_onboard",
]
