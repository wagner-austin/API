"""Registering a project: the last step of onboarding, as one command.

Until this existed, registration was four edits found one red test at a time:
a workspace document, a section in ``docs/RESEARCH.md``, a pinned project list
in ``test_committed_runs.py`` and the generated table. The pinned lists became
derived properties on 2026-09-03; what was left was a procedure written as
intent rather than as a file set, so a first-time reader still met the table
and the filename rule as failures.

WHAT THIS WRITES, AND WHAT IT REFUSES TO. It writes the one workspace
document a project needs and regenerates the table FROM THE DOCUMENTS -- by
re-reading the directory through :func:`~hpc3.core.registry.declared_projects`,
the same reader ``hpc3-research-index`` uses -- never from its own arguments.
A table written from the arguments could disagree with the documents while
both looked current. It does not write the section: what a project measures
and what its provenance leaves out is the one step nothing can derive, so the
section is a PRECONDITION, checked here and named in the refusal.

WHY IT CANNOT RUN EARLY. Registration requires an image digest
(``PROJECT_UNIMAGED``), and this command does not accept one: it READS the
digest of the image file on the cluster, and the command proves the declared
environment and pins inside that image -- with ``env_probe.verify_environment``,
the probe preflight runs -- before anything is written. A
project whose image has not been built cannot be registered, which keeps the
order bootstrap, capture, image, build, register -- the order whose violation
was the paradox ``811c64cb`` removed.

EVERY LOCAL PRECONDITION IS REPORTED AT ONCE. :func:`require_registrable`
collects them all before refusing, because a refusal that names one problem
is the surprise-failure procedure again, one step at a time.
"""

from __future__ import annotations

import os
import pathlib

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue, dump_json_str

from hpc3.contracts.image import ImageReference
from hpc3.contracts.project import ProjectConfig, encode_project_config
from hpc3.contracts.workspace import (
    DEFAULT_QUIET_SECONDS,
    WorkspaceConnection,
    decode_workspace,
)
from hpc3.core import _test_hooks, remote
from hpc3.core.digest import parse_remote_digest
from hpc3.core.index_sections import REGISTERED_HEADING, projects_without_section, section_heading
from hpc3.core.registry import declared_projects, read_document, workspace_filename
from hpc3.core.research_index import render_projects_block, replace_projects_block


def registration_problems(
    *, runs: pathlib.Path, index: pathlib.Path, project: str, repo: pathlib.Path
) -> list[str]:
    """List every local reason a project cannot be registered yet.

    Args:
        runs: Directory holding the workspace documents.
        index: The research index.
        project: The project being registered, already validated as a name.
        repo: Where the project's code lives on this machine.

    Returns:
        One sentence per unmet precondition, empty when there are none.
    """
    problems: list[str] = []
    registered = declared_projects(runs)
    if project in registered:
        problems.append(f"project {project!r} is already declared by a workspace in {runs}")
    document = runs / workspace_filename(project)
    if _test_hooks.file_exists(document):
        problems.append(f"{document} already exists")
    if projects_without_section(read_document(index), [project]) != []:
        problems.append(
            f"{index} has no section headed {section_heading(project)!r} under "
            f"{REGISTERED_HEADING!r}; write what {project!r} measures and what its "
            "provenance does not cover first -- it is the one step nothing derives"
        )
    if not repo.is_dir():
        problems.append(f"--repo {repo} is not a directory")
    return problems


def require_registrable(
    *, runs: pathlib.Path, index: pathlib.Path, project: str, repo: pathlib.Path
) -> None:
    """Refuse a registration with every unmet local precondition named.

    Args:
        runs: Directory holding the workspace documents.
        index: The research index.
        project: The project being registered.
        repo: Where the project's code lives on this machine.

    Raises:
        AppError: With ``REGISTRATION_INCOMPLETE`` naming every problem
            :func:`registration_problems` found, when there is any.
    """
    problems = registration_problems(runs=runs, index=index, project=project, repo=repo)
    if problems:
        raise AppError(
            Hpc3ErrorCode.REGISTRATION_INCOMPLETE,
            f"cannot register {project!r}, {len(problems)} precondition(s) unmet: "
            + "; ".join(problems),
        )


def read_image(host: str, path: str, *, binds: list[str]) -> ImageReference:
    """Name an image by the digest its bytes have on the cluster now.

    The digest is read, never accepted from the caller: a typed digest is a
    claim about a file, and the one this field exists to hold is the file's
    own answer.

    Args:
        host: SSH destination.
        path: Absolute path to the ``.sif`` on the cluster.
        binds: Host directories the payloads must be able to read.

    Returns:
        The reference, pinned to the digest just computed.

    Raises:
        AppError: With ``REMOTE_COMMAND_FAILED`` if the file is absent or
            ``sha256sum`` prints no digest. An image that has not been built
            cannot be registered, which is the point.
    """
    digest = parse_remote_digest(remote.remote_digest(host, path), path)
    return ImageReference(path=path, sha256=digest, binds=binds)


def relative_repo(runs: pathlib.Path, repo: pathlib.Path) -> str:
    """Spell a repository path the way a committed document carries it.

    Relative to the workspace directory, with forward slashes, because
    ``repo`` resolves against the document's own directory -- an absolute path
    would work on exactly one machine.

    Args:
        runs: Directory holding the workspace documents.
        repo: The repository, as given.

    Returns:
        The relative POSIX path, e.g. ``../../../clients/TankpitBot``.
    """
    return pathlib.Path(os.path.relpath(repo.resolve(), runs.resolve())).as_posix()


def workspace_payload(
    connection: WorkspaceConnection, project: str, config: ProjectConfig
) -> dict[str, JSONValue]:
    """Build the workspace document a newly registered project is declared in.

    Args:
        connection: The cluster, host, root and ledger every sibling agrees
            on, with the ledger as written rather than resolved.
        project: The project's name.
        config: Its declaration, with ``repo`` relative to the directory.

    Returns:
        The document, carrying the contract's default triage threshold.
    """
    return {
        "cluster": connection["cluster"],
        "host": connection["host"],
        "root": connection["root"],
        "ledger": connection["ledger"],
        "quiet_seconds": DEFAULT_QUIET_SECONDS,
        "projects": {project: encode_project_config(config)},
    }


def write_registration(
    *,
    runs: pathlib.Path,
    index: pathlib.Path,
    project: str,
    payload: dict[str, JSONValue],
) -> dict[str, ProjectConfig]:
    """Write the workspace document, then regenerate the table from the registry.

    The document is decoded as a whole workspace BEFORE it is written, so a
    declaration the contract refuses never reaches the directory. The table
    is then rendered from a fresh read of every document, the new one
    included -- not from ``payload`` -- so it says what the registry says.

    Args:
        runs: Directory holding the workspace documents.
        index: The research index.
        project: The project being registered.
        payload: The document :func:`workspace_payload` built.

    Returns:
        The registry as re-read after the write, keyed by project.

    Raises:
        JSONTypeError: If the declaration is malformed.
        AppError: If it names hardware the cluster does not have.
        ValueError: If the re-read finds the project declared twice, or the
            index carries no generated block to replace.
    """
    _ = decode_workspace(payload, config_dir=runs)
    _test_hooks.write_text(
        runs / workspace_filename(project), dump_json_str(payload, indent=2) + "\n"
    )
    registry = declared_projects(runs)
    _test_hooks.write_text(
        index, replace_projects_block(read_document(index), render_projects_block(registry))
    )
    return registry


__all__ = [
    "read_image",
    "registration_problems",
    "relative_repo",
    "require_registrable",
    "workspace_payload",
    "write_registration",
]
