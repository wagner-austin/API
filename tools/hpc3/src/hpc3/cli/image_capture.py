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

Usage, ONBOARDING a project that is not registered yet -- same flags plus
``--env-path``, which names the host directory to probe:
    hpc3-image-capture --config hpc3.json --project newcomer \
        --env-path /pub/wagnera3/envs/newcomer ... --out specs/newcomer-image.json

It is one command because the alternative is what produced the first spec by
hand: run pip list, delete the first-party lines, remember to drop pip, paste
the rest into JSON. That is unrepeatable and silently incomplete.

TWO ROUTES, AND WHICH ONE RUNS IS DECIDED BY ``--env-path``.

Without it the project must be REGISTERED, and its own declarations are
reused rather than restated: ``env_path`` says which environment to probe --
a path inside the image, since every registered project has one -- and
``pinned_packages`` become the version assertions the built image checks
itself against. Declaring them twice is how the two drift. This is the
version bump, the recurring job.

With it the project is being ONBOARDED and is not in the registry at all,
because it cannot be: registration requires an image digest, and producing
that digest is what this command starts. Only the workspace's CONNECTION is
read, the named path is probed on the host filesystem, and the spec asserts
no versions -- a project that is not registered has declared none, and
inventing assertions nobody wrote is worse than asserting nothing.

That split is not a check and a way around it. A spec registers nothing, so
neither route can put an unimaged project into the registry. What the
onboarding route avoids is DECODING a registry it does not need: capture used
to read the whole workspace to reach one string, so a single unimaged project
refused the read for the very command whose output would have fixed it.

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
from typing_extensions import TypedDict

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.contracts.image import ImageReference
from hpc3.contracts.image_spec import (
    ImageSpec,
    SymbolCheck,
    decode_image_spec,
    encode_image_spec,
)
from hpc3.contracts.workspace import require_project_config
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.env_probe import InstalledDistribution, parse_installed, probe_command
from hpc3.core.image_capture import capture_layers, third_party_versions
from hpc3.core.image_exec import run_inside_image
from hpc3.core.remote import run_remote

_PROJECT_FLAG = "--project"
_ENV_PATH_FLAG = "--env-path"
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
    _ENV_PATH_FLAG,
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


class ProbeSource(TypedDict):
    """The environment capture will probe, and how to reach it.

    Attributes:
        host: SSH destination the probe runs on.
        env_path: Environment to probe, on the host or inside the image.
        image: Image to run the probe inside, or None to probe the host
            filesystem directly. None ONLY for onboarding, where no image
            exists yet -- which is the whole reason onboarding needs a path
            that does not go through the registry.
        declared_versions: Versions the project has DECLARED its environment
            must contain. Empty when onboarding, since an unregistered project
            has declared none -- and empty for a registered project that pins
            nothing, which ``rusted`` legitimately does. What the built image
            ends up asserting is decided by :func:`_versions_to_assert`.
    """

    host: str
    env_path: str
    image: ImageReference | None
    declared_versions: dict[str, str]


def _environment_to_probe(parsed: dict[str, str], project_name: str) -> ProbeSource:
    """Decide which environment to probe, and by which of the two routes.

    ``--env-path`` selects ONBOARDING: the project is not in the registry, so
    only the workspace's connection is read and the probe runs on the host
    filesystem. Without it the project must be registered, and the probe runs
    inside the image that registration guarantees.

    The two are not a check and a way around it. Registration's rule is
    untouched either way -- this command writes a spec, and a spec registers
    nothing. What ``--env-path`` avoids is decoding a registry the caller does
    not need, which is what made the first image unbuildable: capture read the
    whole workspace to reach one string, so one unimaged project refused the
    read for the command whose output would have fixed it.

    Args:
        parsed: Flags already read from the command line.
        project_name: The project being captured, registered or not.

    Returns:
        Where to probe, and what the built image should assert.

    Raises:
        ValueError: If ``--config`` was not given.
        JSONTypeError: If the document is invalid.
        AppError: If onboarding is not selected and the project is not
            declared, or a declared project is invalid.
    """
    onboarding_env = parsed.get(_ENV_PATH_FLAG)
    if onboarding_env is not None:
        connection = _config.load_workspace_connection(parsed)
        return ProbeSource(
            host=connection["host"],
            env_path=onboarding_env,
            image=None,
            declared_versions={},
        )
    workspace = _config.load_workspace(parsed)
    config = require_project_config(workspace, project_name)
    return ProbeSource(
        host=workspace["host"],
        env_path=config["env_path"],
        image=config["image"],
        declared_versions=dict(config["pinned_packages"]),
    )


def _versions_to_assert(
    declared: dict[str, str],
    installed: dict[str, InstalledDistribution],
    first_party: frozenset[str],
) -> dict[str, str]:
    """Decide which versions the built image must report about itself.

    A project's own pins are the authority when it has any, so the image
    asserts exactly what preflight asserts and the two cannot drift. When it
    has none, the image asserts the third-party versions actually captured --
    which is a real check that the build reproduced the environment it was
    taken from, not a restatement of a declaration.

    THE SPEC CONTRACT REFUSES AN EMPTY MAPPING, deliberately: an image that
    asserts no versions cannot detect its own staleness, and discovers it
    instead in a job that already waited for a GPU. So "assert nothing" is not
    available, and falling back to it would only move the refusal later.

    This is not only the onboarding case. ``pinned_packages`` may legitimately
    be empty for a registered project whose payload is a compiled binary --
    ``rusted`` declares exactly that -- and capturing one would have produced
    an unusable spec.

    Args:
        declared: The project's declared pins; empty if it declares none.
        installed: Everything the probe reported.
        first_party: Distributions that ship as wheels rather than being
            installed from an index.

    Returns:
        The versions to assert.
    """
    if declared != {}:
        return declared
    return third_party_versions(installed, first_party)


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
    project_name = cli_args.require_flag(parsed, _PROJECT_FLAG)
    source = _environment_to_probe(parsed, project_name)

    first_party = frozenset(
        name.strip()
        for name in cli_args.require_flag(parsed, _FIRST_PARTY_FLAG).split(LIST_SEPARATOR)
        if name.strip() != ""
    )
    symbols = parse_symbols(cli_args.require_flag(parsed, _SYMBOLS_FLAG))

    # THE PROBE ALWAYS RUNS INSIDE THE IMAGE, exactly as
    # :func:`~hpc3.core.preflight.check_env_path` does, and for the same
    # reason: an imaged project's ``env_path`` is a CONTAINER path.
    # ``/opt/env`` exists only inside the ``.sif`` and nowhere on the cluster
    # filesystem, so probing the host would fail.
    #
    # There is no host branch any more, and its absence is the point. Every
    # REGISTERED project declares an image
    # (:func:`~hpc3.contracts.project._require_project_image`), so a project
    # this command reads out of a workspace can never be imageless.
    #
    # THAT LEAVES THE FIRST IMAGE UNSERVED BY THIS PATH, and the gap is real
    # rather than theoretical: the flow is capture -> render -> scp -> build,
    # so the spec ``hpc3-image-build`` consumes is the one capture writes, and
    # hand-authoring 38 pinned distributions is not a workflow anybody runs.
    # A project being ONBOARDED is not yet in the registry -- it cannot be,
    # since registration now requires the digest the build produces -- so its
    # environment has to be named directly rather than looked up. That is
    # tracked as its own change; this path is the version bump, which was
    # always the recurring job and was being done by hand-editing
    # ``git_commit`` in the generated spec.
    probe = probe_command(source["env_path"])
    image = source["image"]
    if image is not None:
        probe = run_inside_image(image, probe)
    installed = parse_installed(run_remote(source["host"], probe))
    requirements, wheels = capture_layers(installed, first_party)
    _test_hooks.emit(
        f"probed {source['env_path']}: {len(installed)} distribution(s), "
        f"{len(requirements)} requirement(s), {len(wheels)} wheel(s)"
    )

    spec = ImageSpec(
        base_image=cli_args.require_flag(parsed, _BASE_IMAGE_FLAG),
        env_prefix=cli_args.require_flag(parsed, _ENV_PREFIX_FLAG),
        git_commit=cli_args.require_flag(parsed, _COMMIT_FLAG),
        # Empty for the same reason smoke_commands is, and not because
        # emptiness is a sensible default. Capture probes a PYTHON
        # environment: it asks importlib.metadata what distributions are
        # installed, which cannot see a JVM or an X server. Guessing the
        # operating-system layer from a pip listing would record packages
        # nobody chose, so the layer is declared by hand or not at all.
        system_packages=[],
        extra_index_urls=[cli_args.require_flag(parsed, _EXTRA_INDEX_FLAG)],
        requirements=requirements,
        wheels=wheels,
        # A REGISTERED project already declares which versions its environment
        # must contain. Reusing them means the image asserts what preflight
        # asserts, rather than a second list that drifts from the first. Empty
        # while onboarding, where the project has declared none yet.
        expected_versions=_versions_to_assert(source["declared_versions"], installed, first_party),
        required_symbols=symbols,
        # Empty, and not because emptiness is a sensible default. Capture
        # reads what an environment CONTAINS; a smoke command states what the
        # image must be able to DO, which is a decision about the experiment
        # and cannot be probed off a package list. Writing a guess here would
        # put an assertion into an image that no author chose.
        smoke_commands=[],
        # The project is a FIELD, not a label. It used to live only here,
        # among free-form metadata nothing validates, which is why every later
        # command re-took it from its own flag. One source, typed once.
        labels={"org.corvis.env-source": source["env_path"]},
        project=project_name,
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


if __name__ == "__main__":
    entrypoint()
