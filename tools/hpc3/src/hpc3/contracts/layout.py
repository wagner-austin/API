"""Where a project's jobs live on the cluster, and what they are called.

HPC3 is shared: 102 distinct users had jobs running when this was measured,
and every one of them can see every job name in ``squeue``. A job called
``arm-b-42`` tells them nothing and tells us little more once a second project
is running beside it.

So a project name is required, and it is not decoration:

* It **prefixes every job name**, making ``squeue`` self-describing --
  ``abl.armB-s42`` rather than ``arm-b-42``.
* It **determines the directories**. Scripts and logs are derived from a single
  root, never passed in, so two projects cannot scatter into each other and a
  caller cannot put a job's output somewhere nobody will look for it.

Deriving rather than accepting paths is the point. A caller who can pass a log
directory is a caller who will eventually pass the wrong one, and the job that
results is findable only by whoever remembers what was typed that day.
"""

from __future__ import annotations

from platform_core.json_utils import JSONTypeError, JSONValue, require_str

_ALLOWED = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")

MAX_PROJECT_LENGTH = 24
"""Long enough to name a project, short enough to leave a job name legible.

``squeue``'s default name column truncates, and a prefix that eats it defeats
the purpose of having one.
"""


def require_project(obj: dict[str, JSONValue], key: str) -> str:
    """Read and validate a project name.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The project name.

    Raises:
        JSONTypeError: If the field is missing, not a string, empty, too long,
            or holds anything but lowercase letters, digits and hyphens. The
            charset is narrow because this string becomes both a job-name
            prefix and a directory component; a dot would break the first and
            a slash the second.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    if len(value) > MAX_PROJECT_LENGTH:
        raise JSONTypeError(
            f"Field '{key}' must be at most {MAX_PROJECT_LENGTH} characters, got {len(value)}"
        )
    bad = sorted(set(value) - _ALLOWED)
    if bad != []:
        raise JSONTypeError(
            f"Field '{key}' must be lowercase letters, digits and hyphens only; "
            f"got {value!r} containing {bad}"
        )
    return value


def qualified_name(project: str, name: str) -> str:
    """Build the job name Slurm and every other cluster user will see.

    Args:
        project: Validated project name.
        name: The job's own name within that project.

    Returns:
        ``<project>.<name>``. The dot separates unambiguously because neither
        half may contain one.
    """
    return f"{project}.{name}"


def project_of(job_name: str) -> str | None:
    """Recover the project from a job name Slurm reported.

    The inverse of :func:`qualified_name`, and it is exact rather than a
    guess: a project name may not contain a dot, so the first dot is the
    separator and nothing else can be.

    Exists because ``hpc3-watch`` is handed job IDs, not run documents, and
    a cap now belongs to a project rather than to the workspace. Accounting
    already carries the answer -- the prefix is there precisely so a shared
    ``squeue`` is self-describing -- so watch reads it rather than
    re-deriving it from the ledger.

    Args:
        job_name: The name as ``sacct`` reported it.

    Returns:
        The project prefix, or None when the name carries none. None means
        the job was not submitted by this package, which is a real case on a
        shared cluster and reported rather than raised: a job with no
        declared project has no declared cap, and saying so is honest where
        checking it against someone else's cap would not be.
    """
    project, separator, _ = job_name.partition(".")
    if separator == "" or project == "":
        return None
    return project


def script_dir(root: str, project: str) -> str:
    """Locate a project's batch scripts.

    Args:
        root: Absolute directory holding every project's work.
        project: Validated project name.

    Returns:
        ``<root>/<project>/scripts``.
    """
    return f"{root.rstrip('/')}/{project}/scripts"


def log_dir(root: str, project: str) -> str:
    """Locate a project's job output.

    Args:
        root: Absolute directory holding every project's work.
        project: Validated project name.

    Returns:
        ``<root>/<project>/logs``.
    """
    return f"{root.rstrip('/')}/{project}/logs"


def require_root(parsed_root: str) -> str:
    """Validate the cluster-side root directory.

    Args:
        parsed_root: The root as given on the command line.

    Returns:
        The root, without a trailing slash.

    Raises:
        ValueError: If the root is not an absolute POSIX path, or contains a
            ``..`` segment. Every project directory is joined onto this, so a
            relative or escaping root scatters jobs somewhere unintended.
    """
    if not parsed_root.startswith("/"):
        raise ValueError(f"--root must be an absolute POSIX path, got {parsed_root!r}")
    if "\\" in parsed_root:
        raise ValueError(f"--root must be forward-slashed, got {parsed_root!r}")
    if ".." in parsed_root.split("/"):
        raise ValueError(f"--root must not contain '..', got {parsed_root!r}")
    return parsed_root.rstrip("/") or "/"


__all__ = [
    "MAX_PROJECT_LENGTH",
    "log_dir",
    "project_of",
    "qualified_name",
    "require_project",
    "require_root",
    "script_dir",
]
