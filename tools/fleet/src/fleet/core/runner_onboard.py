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
  3. DERIVES the install entry from the fleet's convention: one runner
     named and labelled after the host, in
     ``/home/<runner account>/actions-runner-<repo>-1``, so a workflow
     targets the host's label and the directory says whose runner it is.
  4. PROVISIONS by re-running the host's whole provision
     (:func:`fleet.core.runner_render.render_provision_script`) over the
     roster with the new install in it, sent as a file and run as root.
     Every step is idempotent, and the host-wide units that list every
     install pick the new one up in the same run.
  5. REWRITES the roster in the same act (the caller, from the mutated
     host entry), so the audit and the load sampler cover the new install
     from birth rather than after somebody remembers.
  6. AUDITS the host as verification and returns the findings -- onboarding
     is not done when the script exits; it is done when the audit says the
     service is running.

Nothing here is optional and nothing falls back: a failure at any step
raises with a typed code and the steps already taken stand.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.runner_registration import runner_service_name
from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import remote, runner_audit, runner_render
from fleet.core.runner_audit_types import AuditFinding
from fleet.core.runner_github import mint_registration_token
from fleet.core.runner_install import RUNNER_ACCOUNT, payload_path


class OnboardPlan(TypedDict):
    """What onboarding one repo onto one host will create.

    Attributes:
        repo: The repository, ``owner/name``.
        install: The install that will be provisioned and added to the
            roster, derived from the fleet's naming convention.
    """

    repo: str
    install: RunnerInstall


def _repo_slug(repo: str) -> str:
    """The repo's name as it appears in install directory names.

    Args:
        repo: The repository, ``owner/name``.

    Returns:
        The name half, lowercased, so the directory answers "whose runner is
        this" by itself.
    """
    return repo.split("/")[1].lower()


def plan_onboard(spec: HostRunnerSpec, repo: str) -> OnboardPlan:
    """The install onboarding will create, per the fleet convention.

    Args:
        spec: The target host's roster entry.
        repo: The repository, ``owner/name``.

    Returns:
        The plan.

    Raises:
        AppError: ``RUNNER_ALREADY_ONBOARDED`` when the roster already
            carries any install for this repo on this host. The roster is
            the memory; a second registration would be a duplicate runner,
            not a refresh.
        ValueError: On a repo that is not ``owner/name`` -- refused here
            rather than downstream so the message names the flag to fix.
    """
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"--onboard takes owner/repo, got {repo!r}")
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
    name = spec["name"]
    install = RunnerInstall(
        repo=repo,
        runner_name=name,
        service=runner_service_name(repo, name),
        workdir=f"/home/{RUNNER_ACCOUNT}/actions-runner-{_repo_slug(repo)}-1/_work",
        labels=[name],
    )
    return OnboardPlan(repo=repo, install=install)


def onboard(spec: HostRunnerSpec, repo: str) -> tuple[OnboardPlan, list[AuditFinding]]:
    """Onboard one repository onto one host, end to end.

    Args:
        spec: The target host's roster entry. MUTATED: the plan's install is
            appended before the provision runs, because the provision is
            rendered from the roster, and the caller's roster write and the
            audit then see the new state -- the whole point of onboarding
            being one act.
        repo: The repository, ``owner/name``.

    Returns:
        The executed plan and the post-onboarding audit findings for the
        host, the new install included.

    Raises:
        AppError: ``RUNNER_ALREADY_ONBOARDED``,
            ``RUNNER_TOKEN_UNAVAILABLE``, ``NODE_UNREACHABLE``,
            ``DISPATCH_FAILED`` or ``RUNNER_AUDIT_UNPARSABLE`` as the
            steps describe; config.sh refusing (an expired token) surfaces
            as ``DISPATCH_FAILED`` with the script's own stderr.
        ValueError: From the planner's flag validation.
    """
    plan = plan_onboard(spec, repo)
    token = mint_registration_token(repo)
    spec["installs"].append(plan["install"])
    remote.run_root_script(
        spec["host"],
        payload_path(spec, f"fleet-onboard-{_repo_slug(repo)}"),
        runner_render.render_provision_script(spec, {repo: token}),
        timeout_seconds=runner_render.STAGE_TIMEOUT_SECONDS,
    )
    outcome = runner_audit.attempt_audit_host(spec)
    findings = outcome["findings"]
    if findings is None:
        raise AppError(
            FleetErrorCode.NODE_UNREACHABLE,
            f"post-onboard audit could not reach {spec['name']}: {outcome['reason']}. "
            "The runner was provisioned and its roster entry is staged; re-run the "
            "audit before trusting either.",
        )
    return plan, findings


__all__ = [
    "OnboardPlan",
    "onboard",
    "plan_onboard",
]
