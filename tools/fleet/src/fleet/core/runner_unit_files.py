"""Writing one systemd unit file on a runner host, the way every provision step does.

A runner host carries two slice units (:mod:`fleet.core.runner_slice_render`)
and each runner unit four drop-ins, each its own fact owned by its own
module: the restart policy (:mod:`fleet.core.runner_recovery`), the kill mode
(:mod:`fleet.core.runner_reaper_render`), the job-started hook
(:mod:`fleet.core.runner_job_scope_render`) and the listener slice. How one
of them reaches the host is one fact: written through a quoted heredoc, so
the body lands byte for byte, and systemd reloaded, only when the file does
not already hold exactly that body, so a re-run provision changes nothing on
a host that holds it. It is written here once.
"""

from __future__ import annotations

from fleet.contracts.runners import RunnerInstall
from fleet.core.script_values import scriptable

#: The heredoc delimiter. No unit body carries a line equal to it.
UNIT_FILE_END = "FLEET_UNIT_FILE_EOF"


def drop_in_path(install: RunnerInstall, name: str) -> str:
    """Where one of a runner unit's drop-ins lives.

    Args:
        install: The install; its ``service`` is the unit's name.
        name: The drop-in's file name.

    Returns:
        ``/etc/systemd/system/<unit>.d/<name>``.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    return f"/etc/systemd/system/{scriptable(install['service'], label='service')}.d/{name}"


def render_unit_file_lines(path: str, body: str, *, announcement: str) -> list[str]:
    """Bash lines that write one unit file only when its text differs.

    Args:
        path: Where the file is written; its directory is made when absent.
        body: The file's text, newline-terminated, with no line equal to
            :data:`UNIT_FILE_END`.
        announcement: What the lines print when they write the file.

    Returns:
        The lines.

    Raises:
        ValueError: When the path or the announcement cannot be embedded
            verbatim, or a line of the body would end the heredoc early.
    """
    scriptable(path, label="unit file path")
    scriptable(announcement, label="unit file announcement")
    if UNIT_FILE_END in body.splitlines():
        raise ValueError(f"the body for {path} holds a line {UNIT_FILE_END!r}")
    text = body.rstrip("\n")
    directory = path.rsplit("/", 1)[0]
    return [
        f'if [ "$(cat {path} 2>/dev/null)" != "$(cat <<\'{UNIT_FILE_END}\'',
        text,
        UNIT_FILE_END,
        ')" ]; then',
        f"    mkdir -p {directory}",
        f"    cat > {path} <<'{UNIT_FILE_END}'",
        text,
        UNIT_FILE_END,
        "    systemctl daemon-reload",
        f"    echo '{announcement}'",
        "fi",
    ]


def render_drop_in_lines(
    install: RunnerInstall, *, name: str, body: str, announcement: str
) -> list[str]:
    """Bash lines that give a runner unit one drop-in.

    Args:
        install: The install; its ``service`` is the unit's name.
        name: The drop-in's file name inside the unit's ``.d`` directory.
        body: The drop-in's text, as :func:`render_unit_file_lines` takes it.
        announcement: What the lines print when they write the file.

    Returns:
        The lines.

    Raises:
        ValueError: As :func:`render_unit_file_lines` and
            :func:`drop_in_path` describe.
    """
    return render_unit_file_lines(drop_in_path(install, name), body, announcement=announcement)


__all__ = ["UNIT_FILE_END", "drop_in_path", "render_drop_in_lines", "render_unit_file_lines"]
