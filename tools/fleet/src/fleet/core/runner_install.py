"""What a runner install, its account and its payloads are called on the host.

The runner release, the account every runner runs as, the environment
variable a repository's registration token arrives in, the directory an
install lives in, and where a payload lands are one fact each, read by the
provision (:mod:`fleet.core.runner_render`), the audit
(:mod:`fleet.core.runner_audit`) and the rebuild alike, so none of them can
name them differently from the others.
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall

#: The GitHub Actions runner release the rendered provisions install.
#:
#: Pinned so two hosts provisioned a week apart run the same agent; measured
#: as the version the live lavender installs run (2026-09-08). Bumping it is
#: a one-line change reviewed like any other.
RUNNER_VERSION = "2.337.0"

#: The account every runner on a host runs as, and owns its install as.
RUNNER_ACCOUNT = "gharunner"


def token_variable(repo: str) -> str:
    """The environment variable a repo's registration token arrives in.

    Args:
        repo: The repository, ``owner/name`` form.

    Returns:
        The variable name, e.g. ``RUNNER_TOKEN_API`` for ``wagner-austin/API``.
    """
    name = repo.split("/")[1]
    sanitized = "".join(c if c.isalnum() else "_" for c in name.upper())
    return f"RUNNER_TOKEN_{sanitized}"


def install_root_for(install: RunnerInstall) -> str:
    """The install directory an install's workdir sits under.

    Args:
        install: The install.

    Returns:
        The parent of the ``_work`` tree.
    """
    return install["workdir"].rsplit("/", 1)[0]


def payload_path(spec: HostRunnerSpec, stem: str) -> str:
    """Where a root payload lands on the runner host.

    Args:
        spec: The host.
        stem: The payload's file-name stem.

    Returns:
        ``<payload_dir>/<stem>.sh``.
    """
    return f"{spec['payload_dir']}/{stem}.sh"


__all__ = [
    "RUNNER_ACCOUNT",
    "RUNNER_VERSION",
    "install_root_for",
    "payload_path",
    "token_variable",
]
