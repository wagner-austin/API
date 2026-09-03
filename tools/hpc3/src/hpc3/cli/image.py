"""CLI: render an image spec into the files a build consumes.

Usage:
    hpc3-image --spec specs/abl-image.json \\
        --out-dir runs/abl-build-v23 --image-name abl.sif \\
        --project mi --name image-v23 --image-dir /pub/wagnera3/images/v23

THE JOB NAME IS DERIVED, NOT ACCEPTED. It used to be a free-text
``--job-name``, and a build rendered on 2026-08-28 carried
``img.abl-sif-v22`` -- a name whose project half is ``img``, which no
workspace declares. ``hpc3-image-build`` refuses exactly that, so the
malformed name pushed its author onto the raw ``sbatch`` path instead, which
records nothing. Taking ``--project`` and ``--name`` and composing them here
means the renderer and the submitter derive the same string from the same
rule, and a project the workspace does not declare is refused at render time
rather than after the files are written.

Writes four files: the Apptainer definition, the pinned requirements, the
in-image self-check, and the build script. All four are rendered from the one
spec, so a definition installing one torch while a self-check expects another
is not a mistake anyone can make by editing a file.

This command does not build. Rendering is pure and runs anywhere; building
pulls several gigabytes and belongs in a batch job on the cluster, which is
what the rendered script is for.

The wheels are NOT rendered or copied. They are built from the repository at
a known commit and staged into ``<out-dir>/wheels`` separately, because the
commit they came from is the provenance the image carries and this command
has no way to verify a claim about it.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.json_utils import load_json_str

from hpc3.cli import _fatal, _test_hooks
from hpc3.contracts.image_spec import decode_image_spec
from hpc3.contracts.layout import qualified_name
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.image_build import render_build_script
from hpc3.core.image_definition import render_definition, render_requirements
from hpc3.core.image_layout import (
    DEFINITION_NAME,
    REQUIREMENTS_NAME,
    SBATCH_NAME,
    SELFCHECK_NAME,
)
from hpc3.core.image_sbatch import render_build_sbatch
from hpc3.core.image_selfcheck import render_selfcheck

BUILD_SCRIPT_NAME = "build.sh"

_SPEC_FLAG = "--spec"
_OUT_DIR_FLAG = "--out-dir"
_IMAGE_NAME_FLAG = "--image-name"
_NAME_FLAG = "--name"
_IMAGE_DIR_FLAG = "--image-dir"
_FLAGS = (
    _SPEC_FLAG,
    _OUT_DIR_FLAG,
    _IMAGE_NAME_FLAG,
    _NAME_FLAG,
    _IMAGE_DIR_FLAG,
)


def require_build_name(raw: str) -> str:
    """Validate the job's own name within its project.

    Args:
        raw: The ``--name`` value.

    Returns:
        The name, unchanged.

    Raises:
        ValueError: If it is empty or contains a dot. The dot is the
            separator :func:`~hpc3.contracts.layout.qualified_name` relies on
            and :func:`~hpc3.contracts.layout.project_of` splits on, so a
            name carrying one makes the two disagree about where the project
            ends.
    """
    if raw == "":
        raise ValueError(f"{_NAME_FLAG} must not be empty")
    if "." in raw:
        raise ValueError(
            f"{_NAME_FLAG} must not contain a dot; it is the separator in "
            f"'<project>.<name>', and {raw!r} would be read as a project"
        )
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    """Render every file an image build needs.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when every file was written.

    Raises:
        ValueError: If a required flag is missing or an argument is unknown.
        JSONTypeError: If the spec is not a valid image spec -- a requirement
            without an exact pin, a wheel name carrying a path separator, an
            environment prefix under a bind-mounted root, an empty commit, or
            an empty assertion list. Nothing is caught: a spec that cannot be
            trusted must not produce a definition that looks buildable.
        OSError: If the output directory cannot be created or a file cannot
            be written. Propagated deliberately, because a partially rendered
            build directory whose definition and self-check disagree is worse
            than no build directory at all.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    spec_path = pathlib.Path(cli_args.require_flag(parsed, _SPEC_FLAG))
    out_dir = pathlib.Path(cli_args.require_flag(parsed, _OUT_DIR_FLAG))
    image_name = cli_args.require_flag(parsed, _IMAGE_NAME_FLAG)
    image_dir = cli_args.require_flag(parsed, _IMAGE_DIR_FLAG)

    raw = core_hooks.read_bytes(spec_path).decode("utf-8")
    spec = decode_image_spec(load_json_str(raw))

    # DERIVED, never accepted. This used to take a free-text --job-name, and
    # on 2026-08-28 a build was rendered as `img.abl-sif-v22` -- a name whose
    # project half is `img`, which no workspace declares. `submit_build`
    # refuses exactly that, but only when the build reaches the cluster
    # through it; the raw `sbatch` that name invited leaves no ledger row, and
    # twenty-two builds went that way.
    #
    # The project half now comes from the SPEC rather than from a --project
    # flag, and there is no registry lookup left here. Capture types the
    # project once and records it as a field; this renderer bakes that same
    # string into build.sbatch; `submit_build` refuses a label disagreeing
    # with the rendered script, before preflight, against the bytes that will
    # run. Agreement across artifacts is stronger than membership in a table,
    # because it also catches the render and the submission drifting apart --
    # and it holds while a project is being ONBOARDED, when the table cannot
    # answer, since registration needs the digest this build produces.
    project = spec["project"]
    build_name = require_build_name(cli_args.require_flag(parsed, _NAME_FLAG))
    job_name = qualified_name(project, build_name)

    core_hooks.make_dir(out_dir)
    rendered = (
        (DEFINITION_NAME, render_definition(spec)),
        (REQUIREMENTS_NAME, render_requirements(spec)),
        (SELFCHECK_NAME, render_selfcheck(spec)),
        (BUILD_SCRIPT_NAME, render_build_script(spec, image_name=image_name)),
        (
            SBATCH_NAME,
            render_build_sbatch(
                image_name=image_name,
                job_name=job_name,
                image_dir=image_dir,
                env_prefix=spec["env_prefix"],
                smoke_commands=spec["smoke_commands"],
            ),
        ),
    )
    for name, text in rendered:
        core_hooks.write_text(out_dir / name, text)
        _test_hooks.emit(f"rendered {out_dir / name}")

    _test_hooks.emit(f"commit {spec['git_commit']}")
    _test_hooks.emit(
        f"{len(spec['requirements'])} pinned requirement(s), "
        f"{len(spec['wheels'])} wheel(s) expected in {out_dir / 'wheels'}"
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = ["BUILD_SCRIPT_NAME", "entrypoint", "main", "require_build_name"]


if __name__ == "__main__":
    entrypoint()
