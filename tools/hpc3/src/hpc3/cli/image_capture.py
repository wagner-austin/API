"""CLI: turn a project's live environment into an image spec.

Usage:
    hpc3-image-capture --config hpc3.json --project abl \\
        --commit d11efacd231ef92426eaf92483c33a8504bd770f \\
        --base-image python:3.11.16-slim-bookworm \\
        --env-prefix /opt/env \\
        --first-party platform_core,platform_ml,model_trainer_server \\
        --symbols model_trainer.cluster.preflight:check_corpus_certified \\
        --extra-index-url https://download.pytorch.org/whl/cu124 \\
        --out specs/abl-image.json

This is the onboarding step for a project adopting an image, and it is one
command because the alternative is what produced the first spec by hand: run
pip list, delete the first-party lines, remember to drop pip, paste the rest
into JSON. That is unrepeatable and silently incomplete.

The project's own declarations are reused rather than restated. ``env_path``
says which environment to probe, and ``pinned_packages`` -- already the
versions that project says its environment must contain -- become the
version assertions the built image checks itself against. Declaring them
twice is how the two drift.

``--env-prefix`` is where the environment goes INSIDE the image, and is not
the path being probed. They are different filesystems, and the contract
refuses a prefix under a root the cluster bind-mounts.

What this does NOT decide: which symbols to assert. That is knowledge about
what the image is for, not about what is installed, so it is required rather
than guessed.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.json_utils import dump_json_str

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.contracts.image_spec import (
    ImageSpec,
    SymbolCheck,
    decode_image_spec,
    encode_image_spec,
)
from hpc3.contracts.workspace import require_project_config
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.env_probe import parse_installed, probe_command
from hpc3.core.image_capture import capture_layers
from hpc3.core.remote import run_remote

_PROJECT_FLAG = "--project"
_COMMIT_FLAG = "--commit"
_BASE_IMAGE_FLAG = "--base-image"
_ENV_PREFIX_FLAG = "--env-prefix"
_FIRST_PARTY_FLAG = "--first-party"
_SYMBOLS_FLAG = "--symbols"
_EXTRA_INDEX_FLAG = "--extra-index-url"
_OUT_FLAG = "--out"

_FLAGS = (
    _config.CONFIG_FLAG,
    _PROJECT_FLAG,
    _COMMIT_FLAG,
    _BASE_IMAGE_FLAG,
    _ENV_PREFIX_FLAG,
    _FIRST_PARTY_FLAG,
    _SYMBOLS_FLAG,
    _EXTRA_INDEX_FLAG,
    _OUT_FLAG,
)

SYMBOL_SEPARATOR = ":"
LIST_SEPARATOR = ","


def parse_symbols(raw: str) -> list[SymbolCheck]:
    """Parse the ``module:attribute`` list a caller declares.

    Args:
        raw: Comma-separated ``module:attribute`` pairs.

    Returns:
        The parsed checks, in the order given.

    Raises:
        ValueError: If an entry carries no separator, or either half is
            empty. An entry this cannot read would otherwise become an
            assertion that never runs, and an image asserting nothing about
            itself has its staleness discovered by a job that already waited
            for a GPU.
    """
    checks: list[SymbolCheck] = []
    for entry in raw.split(LIST_SEPARATOR):
        item = entry.strip()
        if item == "":
            continue
        module, separator, attribute = item.partition(SYMBOL_SEPARATOR)
        if separator == "" or module.strip() == "" or attribute.strip() == "":
            raise ValueError(
                f"{_SYMBOLS_FLAG} entry {item!r} must be 'module{SYMBOL_SEPARATOR}attribute'"
            )
        checks.append(SymbolCheck(module=module.strip(), attribute=attribute.strip()))
    if not checks:
        raise ValueError(f"{_SYMBOLS_FLAG} must name at least one module:attribute")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    """Capture a project's environment into an image spec.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when the spec was written.

    Raises:
        ValueError: If a required flag is missing, an argument is unknown, or
            a ``--symbols`` entry is malformed.
        JSONTypeError: If the workspace is invalid, or the captured spec does
            not satisfy the image contract -- which is checked here, before
            writing, so an unusable spec never reaches a build.
        AppError: If the project is not declared, the environment cannot be
            probed, or it does not contain a named first-party distribution.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)
    project_name = cli_args.require_flag(parsed, _PROJECT_FLAG)
    config = require_project_config(workspace, project_name)

    first_party = frozenset(
        name.strip()
        for name in cli_args.require_flag(parsed, _FIRST_PARTY_FLAG).split(LIST_SEPARATOR)
        if name.strip() != ""
    )
    symbols = parse_symbols(cli_args.require_flag(parsed, _SYMBOLS_FLAG))

    output = run_remote(workspace["host"], probe_command(config["env_path"]))
    installed = parse_installed(output)
    requirements, wheels = capture_layers(installed, first_party)
    _test_hooks.emit(
        f"probed {config['env_path']}: {len(installed)} distribution(s), "
        f"{len(requirements)} requirement(s), {len(wheels)} wheel(s)"
    )

    spec = ImageSpec(
        base_image=cli_args.require_flag(parsed, _BASE_IMAGE_FLAG),
        env_prefix=cli_args.require_flag(parsed, _ENV_PREFIX_FLAG),
        git_commit=cli_args.require_flag(parsed, _COMMIT_FLAG),
        extra_index_urls=[cli_args.require_flag(parsed, _EXTRA_INDEX_FLAG)],
        requirements=requirements,
        wheels=wheels,
        # The project already declares which versions its environment must
        # contain. Reusing them means the image asserts what preflight
        # asserts, rather than a second list that drifts from the first.
        expected_versions=dict(config["pinned_packages"]),
        required_symbols=symbols,
        # Empty, and not because emptiness is a sensible default. Capture
        # reads what an environment CONTAINS; a smoke command states what the
        # image must be able to DO, which is a decision about the experiment
        # and cannot be probed off a package list. Writing a guess here would
        # put an assertion into an image that no author chose.
        smoke_commands=[],
        labels={
            "org.corvis.project": project_name,
            "org.corvis.env-source": config["env_path"],
        },
    )

    # Decoded before writing: a spec that cannot be read back is one a build
    # would fail on later, with the wheels already staged and a GPU reserved.
    encoded = encode_image_spec(spec)
    _ = decode_image_spec(encoded)

    out_path = pathlib.Path(cli_args.require_flag(parsed, _OUT_FLAG))
    core_hooks.make_dir(out_path.parent)
    core_hooks.write_text(out_path, dump_json_str(encoded, indent=2) + "\n")
    _test_hooks.emit(f"wrote {out_path}")
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = ["LIST_SEPARATOR", "SYMBOL_SEPARATOR", "entrypoint", "main", "parse_symbols"]
