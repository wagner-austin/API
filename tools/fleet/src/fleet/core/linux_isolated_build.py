"""The Linux build for a project that runs Docker: as execdocker, never as the runner.

MCPs board tasks a8ee9b21 and 6c4516af, card 722ed843. A project that
declares the ``docker`` tag builds images and runs containers in its own
``make check`` (slime's staging image and browser smoke; the deploy suite's
small compose project). On diphtheria the runner's own user, ``corvis``, is in
the ``docker`` group, so ANY docker command a job ran as that user would
reach ``/var/run/docker.sock``: the production stack's daemon, its images,
its volumes and its running services. A suite that tears down what it
started would tear down production.

So a docker project's build never runs as the runner. The tree the runner
staged is copied into the home of ``execdocker``, a user MCPs'
``scripts/host/lib/fleet-exec-docker.sh`` creates OUTSIDE the docker group
with a rootless daemon of its own, and every install step and the recipe
run as that user through ``sudo -n -u execdocker`` with a cleared
environment whose ``DOCKER_HOST`` is that user's rootless socket. Three
facts, each measured by provisioning or by the host case in
``tests/test_dialect_linux_isolated.py``, make that an isolation rather than
a convention:

- ``execdocker`` cannot open ``/var/run/docker.sock`` (provision refuses to
  finish when it can), so a ``-H`` pointing back at the stack is refused by
  the kernel, not by this script.
- ``env -i`` starts the build from nothing, so no ``DOCKER_HOST``,
  ``DOCKER_CONTEXT`` or credential the runner carries reaches it.
- ``execdocker`` cannot read the runner's home (mode 750), which is why the
  tree is copied rather than read in place.

The runner keeps everything the protocol reads: the transcript and the
result file stay in the dispatch directory, appended to by the runner's own
shell around each ``sudo``. The copy is removed after the recipe, as root,
because a rootless container writes files under subordinate ids its user
cannot delete; a failure to remove it ends the script before the result is
written, which the collector reads as a crashed build rather than a pass.

What this does not stop: containers the suite leaves running belong to
execdocker's daemon, not to the build's unit, so stopping the unit does not
stop them. A suite that starts containers removes them itself, as the deploy
suite's ``compose down -v`` does.
"""

from __future__ import annotations

from fleet.contracts.project import MAKE_TARGET
from fleet.core import names

#: The user whose rootless daemon a docker project's build runs against.
EXEC_USER = "execdocker"

#: The PATH the isolated build starts from: the system's tools only, since
#: ``env -i`` leaves none, and the runner's own ``~/.local/bin`` is unreadable.
EXEC_PATH = "/usr/local/bin:/usr/bin:/bin"


def isolated_build_lines(
    *,
    target: str,
    path: str,
    workers: int,
    install: tuple[tuple[str, ...], ...],
) -> list[str]:
    """The build's lines after the prologue, running every step as execdocker.

    Args:
        target: Absolute remote directory holding the staged tree, its root,
            owned by the runner.
        path: The project's directory inside the export, ``""`` for the root.
        workers: Test workers the capacity check granted.
        install: The project's declared install steps, argv each.

    Returns:
        The lines, in order. The last writes the recipe's exit status to the
        result file, as the runner's own build does, so the collector reads
        both the same way.
    """
    log = names.log_path(target)
    result = f"{target}/{names.RESULT_NAME}"
    run = target.rstrip("/").rsplit("/", 1)[-1]
    lines = [
        f'exec_uid="$(id -u {EXEC_USER})"',
        f'exec_home="$(getent passwd {EXEC_USER} | cut -d : -f 6)"',
        f'exec_root="$exec_home/fleet/stage/{run}"',
        'exec_cache="$exec_home/fleet/cache"',
        'sudo -n rm -rf "$exec_root"',
        'sudo -n mkdir -p "$exec_root" "$exec_cache"',
        f"sudo -n cp -a '{target}/.' \"$exec_root/\"",
        f'sudo -n chown -R {EXEC_USER}:{EXEC_USER} "$exec_root" "$exec_cache"',
        "as_exec() {",
        f"  sudo -n -u {EXEC_USER} env -i "
        f'HOME="$exec_home" USER={EXEC_USER} LOGNAME={EXEC_USER} PATH={EXEC_PATH} '
        'XDG_RUNTIME_DIR="/run/user/$exec_uid" '
        'DOCKER_HOST="unix:///run/user/$exec_uid/docker.sock" '
        'npm_config_cache="$exec_cache/npm" '
        'POETRY_CACHE_DIR="$exec_cache/pypoetry" '
        'PLAYWRIGHT_BROWSERS_PATH="$exec_cache/ms-playwright" '
        f"PYTEST_XDIST_AUTO_NUM_WORKERS='{workers}' "
        'sh -eu -c "$1"',
        "}",
        "status=0",
    ]
    for step in install:
        command = " ".join(step)
        lines.append(f"printf '$ %s\\n' '{command}' >> '{log}'")
        lines.append("set +e")
        lines.append(f"as_exec \"cd '$exec_root' && {command}\" >> '{log}' 2>&1")
        lines.append("status=$?")
        lines.append("set -e")
        lines.append('if [ "$status" -ne 0 ]; then')
        lines.append('  sudo -n rm -rf "$exec_root"')
        lines.append(f"  printf '%s\\n' \"$status\" > '{result}'")
        lines.append("  exit 0")
        lines.append("fi")
    recipe = names.recipe_directory("$exec_root", path)
    lines.append("set +e")
    lines.append(f"as_exec \"cd '{recipe}' && make {MAKE_TARGET}\" >> '{log}' 2>&1")
    lines.append("status=$?")
    lines.append("set -e")
    lines.append('sudo -n rm -rf "$exec_root"')
    lines.append(f"printf '%s\\n' \"$status\" > '{result}'")
    return lines


__all__ = ["EXEC_PATH", "EXEC_USER", "isolated_build_lines"]
