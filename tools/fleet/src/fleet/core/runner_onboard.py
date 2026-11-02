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
  2. DERIVES the install entry from the fleet's convention: one runner
     named and labelled after the host, in
     ``/home/<runner account>/actions-runner-<repo>-1``, so a workflow
     targets the host's label and the directory says whose runner it is.
  3. REWRITES the roster in the same act (the caller, from the mutated
     host entry), so the audit and the load sampler cover the new install
     from birth rather than after somebody remembers.

The registration token is minted by ``--register`` itself (``gh api``):
tokens expire hourly, so any flow that asks a human to paste one has
already lost the idiot-proofing race.

THE HOST GETS THE RUNNER FROM ITS GENERATION (MCPs board task 9aba868d). A
runner host is a NixOS machine whose every runner unit its generation
declares from the roster, so the new install's unit exists once the roster
is rendered (``make render-documents``), rolled, carried into MCPs'
``fleet-mcp/nixos`` by ``fleet-reach render`` and switched to on the host;
``fleet-runners --register`` then mints its token and registers it, and
audits the host, which is when onboarding is done. The command says so.

Nothing here is optional and nothing falls back: a refusal raises with a
typed code and changes nothing.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.runner_registration import runner_service_name
from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core.runner_install import RUNNER_ACCOUNT


class OnboardPlan(TypedDict):
    """What onboarding one repo onto one host will create.

    Attributes:
        repo: The repository, ``owner/name``.
        install: The install added to the roster, derived from the fleet's
            naming convention.
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


def onboard(spec: HostRunnerSpec, repo: str) -> OnboardPlan:
    """Add one repository's install to one host's roster entry.

    Args:
        spec: The target host's roster entry. MUTATED: the plan's install is
            appended, and the caller writes the roster from it.
        repo: The repository, ``owner/name``.

    Returns:
        The plan, its install now in ``spec``.

    Raises:
        AppError: ``RUNNER_ALREADY_ONBOARDED`` as :func:`plan_onboard`
            describes.
        ValueError: From the planner's flag validation.
    """
    plan = plan_onboard(spec, repo)
    spec["installs"].append(plan["install"])
    return plan


__all__ = [
    "OnboardPlan",
    "onboard",
    "plan_onboard",
]
