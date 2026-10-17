"""What the fleet asks GitHub about its self-hosted runners, through ``gh`` on this machine.

Two reads, one auth: a registration token for onboarding and rebuilding a
host (:mod:`fleet.core.runner_onboard`, :mod:`fleet.core.runner_rebuild`), and
each repository's runner list for the audit's dropped-job row
(:mod:`fleet.core.runner_dropped_check`, MCPs board task 715b9ec5). Both run
``gh api`` on the machine running ``fleet-runners``, so a failure of either is
a local auth or permissions condition and is named as one.
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
        status: ``online`` or ``offline``.
        busy: Whether GitHub holds a job on it.
    """

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
            prints a line that is not ``<name> <status> <busy>``: a local
            auth or permission condition, or a change in GitHub's answer,
            named as such rather than read as runners GitHub does not list.
    """
    result = _test_hooks.run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repo}/actions/runners",
            "--jq",
            '.runners[] | "\\(.name) \\(.status) \\(.busy)"',
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
        if len(words) != 3 or words[2] not in ("true", "false"):
            raise AppError(
                FleetErrorCode.RUNNER_GITHUB_UNREADABLE,
                f"gh listed a runner of {repo} as {line!r}, not '<name> <status> <busy>'",
            )
        runners[words[0]] = GitHubRunner(status=words[1], busy=words[2] == "true")
    return runners


__all__ = [
    "GH_API_TIMEOUT_SECONDS",
    "GitHubRunner",
    "mint_registration_token",
    "read_github_runners",
]
