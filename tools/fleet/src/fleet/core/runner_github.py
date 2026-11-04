"""What the fleet asks GitHub about its self-hosted runners, through ``gh`` on this machine.

One auth for all of it: a registration token for registering a host's
runners (:mod:`fleet.core.runner_register`); each repository's runner list,
for the audit's dropped-job row (:mod:`fleet.core.runner_dropped_check`, MCPs
board task 715b9ec5), its registrations row
(:mod:`fleet.core.runner_registrations`) and the drain
(:mod:`fleet.core.runner_drain`); the deletion of a registration no roster
host declares; and a runner's custom labels, read, removed and set, which
take a runner out of GitHub's pool and put it back (MCPs board task
f19d66a5). Each runs ``gh api`` on the machine running ``fleet-runners``, so
a failure of any is a local auth or permissions condition and is named as
one.
"""

from __future__ import annotations

from collections.abc import Sequence

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


#: What every label call prints: the runner's custom labels, one per line.
#: GitHub's answer also lists its read-only labels (``self-hosted``, the OS
#: and the architecture), which no call here can change and no workflow on
#: the fleet targets alone.
CUSTOM_LABELS_JQ = '.labels[] | select(.type == "custom") | .name'

#: What a refused label change says beside gh's own words, so a drain or
#: resume cut short names its repair.
LABELS_REPAIR = (
    "This is local auth or repo permissions, not the fleet; fleet-runners --resume yes for "
    "the runner's host puts the roster's labels back."
)


def _labels_path(repo: str, runner_id: int) -> str:
    """The API path of one runner's labels.

    Args:
        repo: The repository, ``owner/name``.
        runner_id: GitHub's id for the registration.

    Returns:
        ``repos/<repo>/actions/runners/<id>/labels``.
    """
    return f"repos/{repo}/actions/runners/{runner_id}/labels"


def _custom_labels(stdout: str) -> list[str]:
    """The custom labels a label call printed, in GitHub's order.

    Args:
        stdout: What :data:`CUSTOM_LABELS_JQ` printed.

    Returns:
        One label per non-empty line.
    """
    return [line for line in stdout.splitlines() if line]


def read_custom_labels(repo: str, runner_id: int) -> list[str]:
    """The custom labels one runner answers to.

    Args:
        repo: The repository, ``owner/name``.
        runner_id: GitHub's id for the registration.

    Returns:
        Its custom labels, in GitHub's order.

    Raises:
        AppError: ``RUNNER_GITHUB_UNREADABLE`` when ``gh`` exits non-zero,
            naming its stderr.
    """
    result = _test_hooks.run(
        ["gh", "api", _labels_path(repo, runner_id), "--jq", CUSTOM_LABELS_JQ],
        timeout_seconds=GH_API_TIMEOUT_SECONDS,
    )
    if result["returncode"] != 0:
        detail = result["stderr"].strip() or "<no stderr>"
        raise AppError(
            FleetErrorCode.RUNNER_GITHUB_UNREADABLE,
            f"gh could not read the labels of runner {runner_id} of {repo}: {detail}. This is "
            "local auth or repo permissions, not the fleet.",
        )
    return _custom_labels(result["stdout"])


def clear_custom_labels(repo: str, runner_id: int) -> None:
    """Remove every custom label from one runner, and read that none is left.

    A runner with no custom label matches no ``runs-on`` of the fleet's
    workflows, so GitHub assigns it nothing new; a job it already holds runs
    on.

    Args:
        repo: The repository, ``owner/name``.
        runner_id: GitHub's id for the registration.

    Raises:
        AppError: ``RUNNER_LABELS_REFUSED`` when ``gh`` exits non-zero, or
            when GitHub's answer still lists a custom label: a runner that
            still answers to one is still in the pool, and must not be
            reported as out of it.
    """
    result = _test_hooks.run(
        [
            "gh",
            "api",
            "-X",
            "DELETE",
            _labels_path(repo, runner_id),
            "--jq",
            CUSTOM_LABELS_JQ,
        ],
        timeout_seconds=GH_API_TIMEOUT_SECONDS,
    )
    if result["returncode"] != 0:
        detail = result["stderr"].strip() or "<no stderr>"
        raise AppError(
            FleetErrorCode.RUNNER_LABELS_REFUSED,
            f"gh could not remove the labels of runner {runner_id} of {repo}: {detail}. "
            f"{LABELS_REPAIR}",
        )
    left = _custom_labels(result["stdout"])
    if left:
        raise AppError(
            FleetErrorCode.RUNNER_LABELS_REFUSED,
            f"GitHub still lists custom labels {left} on runner {runner_id} of {repo} after "
            f"removing them. {LABELS_REPAIR}",
        )


def set_custom_labels(repo: str, runner_id: int, labels: Sequence[str]) -> None:
    """Make one runner's custom labels exactly these, and read that they are.

    Args:
        repo: The repository, ``owner/name``.
        runner_id: GitHub's id for the registration.
        labels: The labels, at least one; :func:`clear_custom_labels` is how
            a runner comes to answer to none.

    Raises:
        ValueError: When ``labels`` is empty.
        AppError: ``RUNNER_LABELS_REFUSED`` when ``gh`` exits non-zero, or
            when GitHub's answer lists other custom labels than these.
    """
    if not labels:
        raise ValueError(
            f"set_custom_labels was given no label for runner {runner_id} of {repo}; "
            "clear_custom_labels is how a runner comes to answer to none"
        )
    fields = [word for label in labels for word in ("-f", f"labels[]={label}")]
    result = _test_hooks.run(
        [
            "gh",
            "api",
            "-X",
            "PUT",
            _labels_path(repo, runner_id),
            *fields,
            "--jq",
            CUSTOM_LABELS_JQ,
        ],
        timeout_seconds=GH_API_TIMEOUT_SECONDS,
    )
    if result["returncode"] != 0:
        detail = result["stderr"].strip() or "<no stderr>"
        raise AppError(
            FleetErrorCode.RUNNER_LABELS_REFUSED,
            f"gh could not set the labels of runner {runner_id} of {repo} to {list(labels)}: "
            f"{detail}. {LABELS_REPAIR}",
        )
    answered = _custom_labels(result["stdout"])
    if sorted(answered) != sorted(labels):
        raise AppError(
            FleetErrorCode.RUNNER_LABELS_REFUSED,
            f"GitHub lists custom labels {answered} on runner {runner_id} of {repo} after "
            f"setting them to {list(labels)}. {LABELS_REPAIR}",
        )


__all__ = [
    "CUSTOM_LABELS_JQ",
    "GH_API_TIMEOUT_SECONDS",
    "LABELS_REPAIR",
    "GitHubRunner",
    "clear_custom_labels",
    "delete_registration",
    "mint_registration_token",
    "read_custom_labels",
    "read_github_runners",
    "set_custom_labels",
]
