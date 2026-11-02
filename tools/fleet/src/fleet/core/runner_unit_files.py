"""Where a runner unit's drop-ins live on a runner host.

A runner host carries two slice units (:mod:`fleet.core.runner_slice_render`)
and each runner unit five drop-ins, each its own fact owned by its own
module: the restart policy (:mod:`fleet.core.runner_recovery`), the kill mode
(:mod:`fleet.core.runner_reaper_render`), the job hooks
(:mod:`fleet.core.runner_job_scope_render`), the action cache
(:mod:`fleet.core.runner_action_cache`) and the listener slice. Where one of
them lives is one fact, written here once and read by the document the
host's NixOS generation carries (:mod:`fleet.core.runner_host_document`, MCPs
board task 9aba868d) and by the live host's cases.
"""

from __future__ import annotations

from fleet.contracts.runners import RunnerInstall
from fleet.core.script_values import scriptable


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


__all__ = ["drop_in_path"]
