"""What the fleet asks GitHub about its self-hosted runners, through ``gh`` on this machine.

Two reads and one removal, one auth: a registration token for onboarding and
rebuilding a host (:mod:`fleet.core.runner_onboard`,
:mod:`fleet.core.runner_rebuild`); each repository's runner list, for the
audit's dropped-job row (:mod:`fleet.core.runner_dropped_check`, MCPs board
task 715b9ec5) and its registrations row (:mod:`fleet.core.runner_registrations`);
and the deletion of a registration no roster host declares. Each runs
``gh api`` on the machine running ``fleet-runners``, so a failure of any is a
local auth or permissions condition and is named as one.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.core import _test_hooks

#: Each ``gh api`` call's deadline, in seconds: one HTTPS round trip to
#: GitHub. A minute outlasts any answer ``gh`` gives and ends a hung login
#: prompt or a proxy that never connects with a named refusal.
GH_API_TIMEOUT_SECONDS = 60


class GitHubRunner(TypedDict):
    """GitHub's reading of one self-hosted runner.

    Attributes:
        id: GitHub's id for the registration, which a deletion names.
        status: ``online`` or ``offline``.
        busy: Whether GitHub holds a job on it.
    """

    id: int
    status: str
    busy: bool


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
        ],
        timeout_seconds=GH_API_TIMEOUT_SECONDS,
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


def read_github_runners(repo: str) -> dict[str, GitHubRunner]:
    """GitHub's reading of every self-hosted runner of one repository.

    Args:
        repo: The repository, ``owner/name``.

    Returns:
        Each runner by name.

    Raises:
        AppError: ``RUNNER_GITHUB_UNREADABLE`` when ``gh`` exits non-zero or
            prints a line that is not ``<id> <name> <status> <busy>``: a
            local auth or permission condition, or a change in GitHub's
            answer, named as such rather than read as runners GitHub does
            not list.
    """
    result = _test_hooks.run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repo}/actions/runners",
            "--jq",
            '.runners[] | "\\(.id) \\(.name) \\(.status) \\(.busy)"',
        ],
        timeout_seconds=GH_API_TIMEOUT_SECONDS,
    )
    if result["returncode"] != 0:
        detail = result["stderr"].strip() or "<no stderr>"
        raise AppError(
            FleetErrorCode.RUNNER_GITHUB_UNREADABLE,
            f"gh could not list {repo}'s runners: {detail}. This is local auth or repo "
            "permissions, not the fleet.",
        )
    runners: dict[str, GitHubRunner] = {}
    for line in result["stdout"].splitlines():
        words = line.split()
        if len(words) != 4 or not words[0].isdigit() or words[3] not in ("true", "false"):
            raise AppError(
                FleetErrorCode.RUNNER_GITHUB_UNREADABLE,
                f"gh listed a runner of {repo} as {line!r}, not '<id> <name> <status> <busy>'",
            )
        runners[words[1]] = GitHubRunner(id=int(words[0]), status=words[2], busy=words[3] == "true")
    return runners


def delete_registration(repo: str, runner_id: int) -> None:
    """Remove one runner registration from a repository.

    Args:
        repo: The repository, ``owner/name``.
        runner_id: GitHub's id for the registration.

    Raises:
        AppError: ``RUNNER_DEREGISTRATION_REFUSED`` when ``gh`` exits
            non-zero, naming its stderr: a local auth or permission
            condition, never a registration to report as removed.
    """
    result = _test_hooks.run(
        ["gh", "api", "-X", "DELETE", f"repos/{repo}/actions/runners/{runner_id}"],
        timeout_seconds=GH_API_TIMEOUT_SECONDS,
    )
    if result["returncode"] != 0:
        detail = result["stderr"].strip() or "<no stderr>"
        raise AppError(
            FleetErrorCode.RUNNER_DEREGISTRATION_REFUSED,
            f"gh could not delete runner {runner_id} of {repo}: {detail}. This is local auth "
            "or repo permissions, not the fleet.",
        )


__all__ = [
    "GH_API_TIMEOUT_SECONDS",
    "GitHubRunner",
    "delete_registration",
    "mint_registration_token",
    "read_github_runners",
]
