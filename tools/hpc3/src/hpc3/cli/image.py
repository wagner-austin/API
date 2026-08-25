"""CLI: render an image spec into the files a build consumes.

Usage:
    hpc3-image --spec runs/abl-image.json --out-dir runs/abl-build \\
        --image-name abl.sif

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
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.image_build import render_build_script
from hpc3.core.image_definition import render_definition, render_requirements
from hpc3.core.image_layout import (
    DEFINITION_NAME,
    REQUIREMENTS_NAME,
    SELFCHECK_NAME,
)
from hpc3.core.image_selfcheck import render_selfcheck

BUILD_SCRIPT_NAME = "build.sh"

_SPEC_FLAG = "--spec"
_OUT_DIR_FLAG = "--out-dir"
_IMAGE_NAME_FLAG = "--image-name"
_FLAGS = (_SPEC_FLAG, _OUT_DIR_FLAG, _IMAGE_NAME_FLAG)


def main(argv: Sequence[str] | None = None) -> int:
    """Render every file an image build needs.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when all four files were written.

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

    raw = core_hooks.read_bytes(spec_path).decode("utf-8")
    spec = decode_image_spec(load_json_str(raw))

    core_hooks.make_dir(out_dir)
    rendered = (
        (DEFINITION_NAME, render_definition(spec)),
        (REQUIREMENTS_NAME, render_requirements(spec)),
        (SELFCHECK_NAME, render_selfcheck(spec)),
        (BUILD_SCRIPT_NAME, render_build_script(spec, image_name=image_name)),
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


__all__ = ["BUILD_SCRIPT_NAME", "entrypoint", "main"]
