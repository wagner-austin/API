"""Retiring a settled dispatch's directory on its node (MCPs board task bfca20e6).

A dispatch's export lives under ``<stage_root>/<run_id>`` on the node it ran
on, beside its transcript. Until this module nothing removed it: measured on
diphtheria 2026-09-27, the stage root held 130 exports in 293 GB and the disk
was at 96 percent, with 133 ``mkdir-`` scripts at the root outliving the trees
they made. Nothing reads an export again once its runner has read the
transcript's tail, so the runner retires it at that moment: the transcript
moves to ``<stage_root>/logs/<run_id>.log`` (:func:`fleet.core.names.retained_log_path`),
which is the path the verdict line names, and the export and the run's root
scripts are removed.

ONE RETIRE, THREE CALLERS, like :mod:`fleet.core.stop`: the node runner's
settle (a finished run, or one stopped past its lease), every stop that
closes a row (a cancel, by ``fleet-cancel`` or under a runner), and the hub's
``fleet-collect``. A failure propagates with the row still live, so the next
tick settles the run again; every step of the script tolerates what an
earlier attempt already did.

AND THE RUN'S VIRTUALENV GOES WITH IT (MCPs board task 7b07c5d2). Poetry
named it after the tree just removed, so right after the retire the node is
sent :mod:`fleet.core.venv_sweep`, which removes every virtualenv in the
node's fleet cache whose recorded source paths are all gone.
"""

from __future__ import annotations

from fleet.contracts.node import NodeConfig, NodePlatform
from fleet.core import dialect, names, remote, venv_sweep


def script_for(platform: NodePlatform, *, stage_root: str, run_id: str) -> str:
    """The retire script one dispatch's node is sent.

    The one place its arguments are derived from the run, so the script a
    node runs and the committed render the Pester suite executes
    (:mod:`fleet.core.rendered_powershell`) cannot be built two ways.

    Args:
        platform: The node's declared platform.
        stage_root: The node's declared stage root.
        run_id: The dispatch.

    Returns:
        The script's text.
    """
    spoken = dialect.for_platform(platform)
    return spoken.retire_script(
        target=names.dispatch_directory(stage_root, run_id),
        retained=names.retained_log_path(stage_root, run_id),
        scripts=tuple(
            spoken.script_path(stage_root, stem) for stem in names.root_script_stems(run_id)
        ),
    )


def retire_on_node(node: NodeConfig, *, run_id: str) -> str:
    """Keep one dispatch's transcript and remove its export and root scripts.

    Args:
        node: The node it was dispatched to.
        run_id: The dispatch.

    Returns:
        Where its transcript is now kept on the node.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            transport, the latter carrying the node's own error when a file
            could not be moved or removed.
    """
    stage_root = node["stage_root"]
    remote.run_script(
        node["host"],
        dialect.for_platform(node["platform"]).script_path(stage_root, names.retire_stem(run_id)),
        script_for(node["platform"], stage_root=stage_root, run_id=run_id),
        platform=node["platform"],
    )
    venv_sweep.sweep_on_node(node)
    return names.retained_log_path(stage_root, run_id)


__all__ = ["retire_on_node", "script_for"]
