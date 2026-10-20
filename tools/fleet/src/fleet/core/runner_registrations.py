"""GitHub runner registrations no roster host declares: the audit rows naming them, and the prune.

MCPs board task 1e9ab738. lavender was reimaged on 2026-10-19 from Windows,
whose WSL guest lavender-wsl carried the Linux runners, to Ubuntu, and
rebuilt from this roster on 2026-10-20. GitHub still listed every runner the
two old installs had registered: lavender-wsl and lavender-wsl-2 for API,
lavender-wsl through lavender-wsl-4 for MCPs, lavender-wsl for corvis-stick
and tree-bot, all offline. Nothing in the audit could say so, because the
audit asked GitHub only about the runners the roster declares, and a
workflow still asking for a label only those registrations carried queued
forever: on 2026-10-20 an MCPs run of 2026-10-19T20:31Z held one job asking
``[self-hosted, lavender-wsl]``.

So the audit reads every registration of each repository the roster's
installs name, and holds it to the roster: one row per repository, ok when
every registration GitHub lists is a ``runner_name`` some roster host
declares for it. ``fleet-runners --prune yes`` removes each undeclared one
that GitHub reads offline and not busy; an online one is a live machine the
roster does not know, which is named and never removed, since deleting it
would leave a running listener with no registration and a job mid-run.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from fleet.contracts.runners import RunnerSpec
from fleet.core.runner_audit_types import AuditFinding
from fleet.core.runner_github import GitHubRunner, delete_registration, read_github_runners

#: Why the row matters, printed beside a drift line.
REGISTRATIONS_REASON = (
    "a registration no roster host declares outlives its machine: on 2026-10-20 eight "
    "lavender-wsl registrations outlived the WSL guest, and a job asking their label queued "
    "until it was cancelled"
)


class Registration(TypedDict):
    """One registration GitHub lists that no roster host declares.

    Attributes:
        repo: The repository, ``owner/name``.
        name: The runner's registered name.
        github: GitHub's reading of it.
    """

    repo: str
    name: str
    github: GitHubRunner


class RegistrationReading(TypedDict):
    """One repository's registrations held to the roster.

    Attributes:
        repo: The repository.
        examined: How many registrations GitHub listed.
        declared: How many runner names the roster declares for it.
        undeclared: Each listed registration the roster does not declare,
            in GitHub's order.
    """

    repo: str
    examined: int
    declared: int
    undeclared: list[Registration]


class PruneReport(TypedDict):
    """What a prune did.

    Attributes:
        removed: Each undeclared registration it deleted.
        kept: Each undeclared registration it left, GitHub reading it online
            or busy.
        examined: How many registrations it read, across every repository.
    """

    removed: list[Registration]
    kept: list[Registration]
    examined: int


def registrations_check_id(repo: str) -> str:
    """The row's id for one repository.

    Args:
        repo: The repository.

    Returns:
        E.g. ``registrations:wagner-austin/API``.
    """
    return f"registrations:{repo}"


def declared_names(spec: RunnerSpec) -> dict[str, set[str]]:
    """Every runner name the roster declares, by repository.

    Args:
        spec: The roster.

    Returns:
        Each repository any host's installs name, Linux and Windows alike,
        with those installs' runner names.
    """
    declared: dict[str, set[str]] = {}
    for host in spec["hosts"]:
        for install in host["installs"]:
            declared.setdefault(install["repo"], set()).add(install["runner_name"])
    for windows in spec["windows_hosts"]:
        for runner in windows["installs"]:
            declared.setdefault(runner["repo"], set()).add(runner["runner_name"])
    return declared


def read_registrations(spec: RunnerSpec, repos: list[str]) -> list[RegistrationReading]:
    """Hold each repository's registrations to the roster.

    Args:
        spec: The whole roster, whose every host's declarations count.
        repos: The repositories to read, each one the roster names.

    Returns:
        One reading per repository, in the order given.

    Raises:
        AppError: ``RUNNER_GITHUB_UNREADABLE`` as
            :func:`fleet.core.runner_github.read_github_runners` says.
    """
    declared = declared_names(spec)
    readings: list[RegistrationReading] = []
    for repo in repos:
        listed = read_github_runners(repo)
        names = declared[repo]
        undeclared = [
            Registration(repo=repo, name=name, github=github)
            for name, github in listed.items()
            if name not in names
        ]
        readings.append(
            RegistrationReading(
                repo=repo, examined=len(listed), declared=len(names), undeclared=undeclared
            )
        )
    return readings


def describe_registration(registration: Registration) -> str:
    """One registration in words.

    Args:
        registration: The registration.

    Returns:
        E.g. ``lavender-wsl (id 21, offline, idle)``.
    """
    github = registration["github"]
    state = "busy" if github["busy"] else "idle"
    return f"{registration['name']} (id {github['id']}, {github['status']}, {state})"


def registration_finding(reading: RegistrationReading) -> AuditFinding:
    """The audit row for one repository.

    Args:
        reading: The repository's reading.

    Returns:
        Ok when GitHub lists no registration the roster does not declare;
        either way the detail says how many it examined.
    """
    examined = (
        f"examined {reading['examined']} registration(s) of {reading['repo']} against the "
        f"{reading['declared']} runner name(s) the roster declares for it"
    )
    undeclared = reading["undeclared"]
    if undeclared:
        detail = (
            f"{examined}; {len(undeclared)} no roster host declares: "
            + ", ".join(describe_registration(each) for each in undeclared)
            + "; fleet-runners --prune yes removes the offline idle ones"
        )
    else:
        detail = f"{examined}; every one is declared"
    return AuditFinding(
        check_id=registrations_check_id(reading["repo"]),
        ok=not undeclared,
        detail=detail,
        reason=REGISTRATIONS_REASON,
    )


def prune_registrations(spec: RunnerSpec) -> PruneReport:
    """Delete every registration no roster host declares that GitHub reads offline and idle.

    Args:
        spec: The roster; every repository its installs name is read.

    Returns:
        What was removed and what was kept.

    Raises:
        AppError: ``RUNNER_GITHUB_UNREADABLE`` from the read, or
            ``RUNNER_DEREGISTRATION_REFUSED`` from a deletion, which stops
            the prune so no registration it did not remove is reported as
            removed.
    """
    readings = read_registrations(spec, sorted(declared_names(spec)))
    removed: list[Registration] = []
    kept: list[Registration] = []
    for reading in readings:
        for registration in reading["undeclared"]:
            github = registration["github"]
            if github["status"] == "offline" and not github["busy"]:
                delete_registration(registration["repo"], github["id"])
                removed.append(registration)
            else:
                kept.append(registration)
    examined = sum(reading["examined"] for reading in readings)
    return PruneReport(removed=removed, kept=kept, examined=examined)


__all__ = [
    "REGISTRATIONS_REASON",
    "PruneReport",
    "Registration",
    "RegistrationReading",
    "declared_names",
    "describe_registration",
    "prune_registrations",
    "read_registrations",
    "registration_finding",
    "registrations_check_id",
]
