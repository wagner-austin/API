"""What a runner install is called on its host, for both sides' provisions.

The runner release, the environment variable a repository's registration
token arrives in, and the directory an install lives in are one fact each,
read by the Windows provision (:mod:`fleet.core.runner_windows_provision`)
and the Linux one (:mod:`fleet.core.runner_render`) alike, so neither side
can name them differently from the other.
"""

from __future__ import annotations

from fleet.contracts.runners import RunnerInstall

#: The GitHub Actions runner release the rendered provisions install.
#:
#: Pinned so two hosts provisioned a week apart run the same agent; measured
#: as the version the live lavender installs run (2026-09-08). Bumping it is
#: a one-line change reviewed like any other.
RUNNER_VERSION = "2.337.0"


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
        The parent of the ``_work`` tree, backslashed for a windows-side
        install (command lines and cmdlets both take that form) and POSIX
        for a wsl one.
    """
    root = install["workdir"].rsplit("/", 1)[0]
    if install["side"] == "windows":
        return root.replace("/", "\\")
    return root


__all__ = ["RUNNER_VERSION", "install_root_for", "token_variable"]
