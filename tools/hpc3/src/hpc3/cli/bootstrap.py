"""CLI: create the first environment, the step the image flow starts after.

Usage:
    hpc3-bootstrap --config hpc3.json --project newcomer \\
        --env-path /pub/wagnera3/envs/newcomer --python 3.11

Then capture it, which is where the documented flow previously began:

    hpc3-image-capture --config hpc3.json --project newcomer \\
        --env-path /pub/wagnera3/envs/newcomer ... --out specs/newcomer-image.json

WHY THIS EXISTS. Capture probes a live environment over SSH. A project being
onboarded has none, so the four-command flow began one step after the
beginning and the first step was improvised every time. This package had 34
typed refusals and not one command that created anything -- an asymmetry whose
cost is invisible until somebody starts something new, because every refusal
is correct for every project that is already finished.

THE CONNECTION, NOT THE WORKSPACE. This reads ``--config`` through
:func:`~hpc3.cli._config.load_workspace_connection`, which leaves the project
registry unread. That is not an optimisation: a project being bootstrapped
CANNOT be registered yet, because registration requires an image digest and
producing one starts here. Loading the full workspace would refuse this
command with ``PROJECT_UNIMAGED`` -- the same bootstrap paradox that made
capture unusable for onboarding until ``811c64cb`` split the two loaders
([[invariant-placement]]).

``--project`` is therefore a NAME, not a lookup. It is recorded in the audit
event so a directory on shared storage can be traced back to what it was made
for, which is the thing ``/pub/wagnera3/envs`` currently cannot answer for any
of its three environments.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core import cli_args

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.core import audit
from hpc3.core.bootstrap import CONDA_MODULE, bootstrap_environment

_PROJECT_FLAG = "--project"
_ENV_PATH_FLAG = "--env-path"
_PYTHON_FLAG = "--python"

_FLAGS = (_config.CONFIG_FLAG, _PROJECT_FLAG, _ENV_PATH_FLAG, _PYTHON_FLAG)


def main(argv: Sequence[str] | None = None) -> int:
    """Create and verify a project's first environment.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when the environment exists and its own interpreter
        confirms both the requested version and that the environment owns it.

    Raises:
        ValueError: If a required flag is missing or an argument is unknown.
        JSONTypeError: If the workspace document's connection fields are
            invalid.
        AppError: With ``BOOTSTRAP_ENV_EXISTS`` if the path is occupied,
            ``REMOTE_COMMAND_FAILED`` if conda fails,
            ``ENV_PROBE_UNREADABLE`` if the new interpreter cannot be read,
            or ``BOOTSTRAP_PYTHON_MISMATCH`` /
            ``BOOTSTRAP_ENV_NOT_SELF_CONTAINED`` if what was built is not what
            was asked for. Nothing is caught: an environment that exists and
            is wrong is worse than none, because the next command probes it
            and believes it.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    host = _config.load_workspace_connection(parsed)["host"]
    project = cli_args.require_flag(parsed, _PROJECT_FLAG)
    env_path = cli_args.require_flag(parsed, _ENV_PATH_FLAG)
    python_version = cli_args.require_flag(parsed, _PYTHON_FLAG)

    _test_hooks.emit(f"creating {env_path} on {host} with python={python_version}")
    identity = bootstrap_environment(host, env_path, python_version)

    # Emitted only after the identity check passed. An event announcing an
    # environment that turned out to be the wrong one would be a false trail
    # in the only durable record of how it came to exist.
    audit.environment_bootstrapped(
        host=host,
        project=project,
        env_path=env_path,
        python_version=python_version,
        conda_module=CONDA_MODULE,
    )
    _test_hooks.emit(f"python {identity['version']}, self-contained at {identity['base_prefix']}")
    _test_hooks.emit(f"capture it next: hpc3-image-capture --env-path {env_path}")
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
