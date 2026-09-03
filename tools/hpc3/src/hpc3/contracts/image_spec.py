"""What an image IS, as a document a build can be produced from.

The build-time half of the image contract; :mod:`hpc3.contracts.image` holds
the run-time half. A directory pinned by convention cannot say what it held
when a run used it -- ``/pub/wagnera3/envs/abl-pinned`` is the concrete case,
and every artifact HPC3 has produced carries ``git_commit`` null as a result.

The rules below are the ones that make the description worth trusting:

* Every requirement carries an exact ``==`` pin. An unpinned requirement
  resolves at build time, reintroducing the drift the image exists to remove
  -- silently, because the build still succeeds.
* Wheels are bare filenames, joined onto a directory inside the build, so a
  separator would reach outside it.
* The commit is non-empty. The trainer reads an empty ``GIT_COMMIT`` as "not
  stamped" and records null, so an image stamped with an empty string claims
  provenance it does not have -- worse than admitting none.
* At least one version assertion and one symbol assertion. An image that
  checks nothing about itself is one whose staleness is discovered by a job
  that already waited for a GPU.

What this deliberately does NOT describe: corpora, artifacts, GPU model or
driver. Data is bind-mounted and identified by digest
(:mod:`hpc3.contracts.stage`); hardware is an axis containers do not control.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_list,
)
from typing_extensions import TypedDict

from hpc3.contracts.image import require_json_object
from hpc3.contracts.image_spec_fields import (
    require_bare_filenames,
    require_container_dir,
    require_digest_pinned_image,
    require_non_empty_str,
    require_pinned_requirements,
    require_pinned_system_packages,
    require_str_list,
    require_str_map,
)
from hpc3.contracts.layout import require_project

#: How a distribution's package manager spells an exact pin. One ``=``, not
#: two: ``apt-get install xvfb=2:21.1.4-2ubuntu1.7`` is the syntax, and
#: writing pip's separator here installs nothing and reports success.
#: Characters that would turn one package specification into something else
#: once interpolated into the build script.


class SymbolCheck(TypedDict):
    """One attribute whose presence proves a fresh wheel was baked.

    Attributes:
        module: Importable dotted module path, evaluated inside the image.
        attribute: Attribute that must exist on that module. Absence means a
            stale wheel was installed, which is precisely the failure an
            image is supposed to make impossible to ship unnoticed.
    """

    module: str
    attribute: str


class ImageSpec(TypedDict):
    """Everything needed to render a reproducible image definition.

    Attributes:
        base_image: Docker reference the build bootstraps from, which MUST
            carry an ``@sha256:`` digest. The CUDA runtime arrives as
            ``nvidia-*-cu12`` wheels in ``requirements``, so this does not
            need to be a CUDA base image and should not be one.

            A TAG IS NOT A PIN, and this document exists to pin things. A tag
            is a mutable pointer the publisher can move, so two builds of the
            same spec a week apart can start from different bytes and neither
            says so -- the identical argument the ``system_packages`` field
            already makes about an unpinned ``apt-get install``.

            It is not hypothetical either. ``rusted`` pinned
            ``python:3.11-slim-bookworm@sha256:0bee7276...``; that tag now
            resolves to ``sha256:528257d4...``. The tag moved under four
            specs that named it bare, and nothing in the workspace noticed.

            The digest is required rather than resolved here, because
            resolving means a network call to a registry at decode time --
            which would make reading a document depend on the internet, and
            would silently re-pin a spec every time the tag moved, which is
            the failure being prevented. Get it with
            ``docker buildx imagetools inspect <ref>``, or from the
            registry's Docker-Content-Digest header.
        env_prefix: Absolute POSIX directory receiving the virtualenv, inside
            the image. Never under a host-bound path. HPC3 does NOT auto-bind
            ``/pub`` -- that claim was here and is false; ``/pub`` is a host
            symlink to ``/dfs6b/pub`` and does not exist inside a container
            at all, which is why a job that needs it must declare the bind.
            The rule stands for the other reason: a job DOES bind its data
            roots, and an env under one of them would be shadowed at runtime
            by the host directory mounted over it, so the image's own
            interpreter would disappear behind the copy it was meant to
            replace.
        git_commit: Commit the wheels were built from. Written into the image
            so the trainer can stamp it into every manifest.
        extra_index_urls: Additional package indexes, in order. Present
            because a local version such as ``torch==2.6.0+cu124`` is
            published only on the PyTorch index.
        system_packages: Operating-system packages the image installs before
            anything else, every entry an exact ``=`` pin in the distribution's
            own syntax (``xvfb=2:21.1.4-2ubuntu1.7``).

            Present because not every dependency is a wheel. A JVM, an X
            server and a software OpenGL stack cannot be pip-installed, and an
            image that could only describe its Python layer forced the
            alternative: a hand-built base image nobody could reproduce, which
            is the drift this whole document exists to remove. Empty is the
            ordinary case and is a recorded decision, not an omission.

            Pinned for exactly the reason the pip layer is: an unpinned
            ``apt-get install`` resolves at build time against whatever the
            distribution is serving that day, and it does so successfully, so
            two images built a week apart differ with nothing to say they do.
        requirements: Third-party layer, every line an exact ``==`` pin.
        wheels: First-party wheel filenames, installed with dependencies
            already pinned above.
        expected_versions: Package name to exact version the built image must
            report. Checked inside the image, at build time.
        required_symbols: Attributes that must exist in the built image.
        smoke_commands: Commands run INSIDE the built image after the
            self-check, each of which must exit 0. Importing a symbol is not
            computing with it: v5 carried a probe whose module-level entry
            point silently did nothing, and every symbol assertion passed.
            Declared even when empty, so "this image asserts no behaviour" is
            a recorded decision rather than an omission.

            They run with NO binds, because at build time the image is the
            only thing that exists -- there is no job, no data root and
            nothing declaring what to mount. A command that writes must
            therefore write inside the container (``/tmp``). Reaching for a
            host path here fails with a read-only filesystem error that names
            the path and not the cause, which is exactly how the first
            rendered build job failed.
        labels: Free-form metadata recorded on the image.
        project: The project this image is for, validated as a project name.

            FIRST-CLASS RATHER THAN A LABEL, and that is the whole point.
            Capture used to record it as ``org.corvis.project`` among the
            free-form labels, which nothing requires and nothing validates, so
            every later step re-took the project from ITS OWN ``--project``
            flag. Three commands each retyping one string is three chances to
            typo it, and the defence against that was to look the name up in
            the workspace registry -- which a project mid-onboarding is not in,
            because registration needs the digest this build produces.

            Naming it once here removes both problems at once. The renderer
            reads it instead of being told it, so the job name it bakes into
            ``build.sbatch`` cannot disagree with the spec; the submitter
            already refuses a label that disagrees with the rendered script
            (``check_name_agrees``, before preflight, against the bytes that
            will run). Agreement across artifacts is a stronger check than
            membership in a table, and it holds during onboarding, when the
            table cannot answer.
    """

    base_image: str
    env_prefix: str
    git_commit: str
    system_packages: list[str]
    extra_index_urls: list[str]
    requirements: list[str]
    wheels: list[str]
    expected_versions: dict[str, str]
    required_symbols: list[SymbolCheck]
    smoke_commands: list[str]
    labels: dict[str, str]
    project: str


def require_symbol_check(value: JSONValue, where: str) -> SymbolCheck:
    """Decode one symbol assertion.

    Args:
        value: Value produced by the JSON loader.
        where: Location used in error messages.

    Returns:
        The validated symbol check.

    Raises:
        JSONTypeError: If the value is not an object, or either field is
            missing or empty.
    """
    obj = require_json_object(value, where)
    return SymbolCheck(
        module=require_non_empty_str(obj, "module"),
        attribute=require_non_empty_str(obj, "attribute"),
    )


def _require_symbol_checks(obj: JSONObject, key: str) -> list[SymbolCheck]:
    """Read the required non-empty list of symbol assertions.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The symbol checks, in order.

    Raises:
        JSONTypeError: If the field is missing, not a list, empty, or holds a
            malformed entry. Emptiness is refused because an image that
            asserts nothing about its own contents cannot detect a stale
            wheel, and detecting that at build time is why the check exists.
    """
    raw = require_list(obj, key)
    if not raw:
        raise JSONTypeError(f"Field '{key}' must not be empty")
    return [require_symbol_check(item, f"{key}[{index}]") for index, item in enumerate(raw)]


def encode_symbol_check(check: SymbolCheck) -> JSONObject:
    """Encode a symbol assertion to a JSON object.

    Args:
        check: Assertion to encode.

    Returns:
        The JSON object form.
    """
    return {"module": check["module"], "attribute": check["attribute"]}


def encode_image_spec(spec: ImageSpec) -> JSONObject:
    """Encode an image spec to a JSON object.

    Args:
        spec: Spec to encode.

    Returns:
        The JSON object form, round-trippable through
        :func:`decode_image_spec`.
    """
    return {
        "base_image": spec["base_image"],
        "env_prefix": spec["env_prefix"],
        "git_commit": spec["git_commit"],
        "system_packages": list(spec["system_packages"]),
        "extra_index_urls": list(spec["extra_index_urls"]),
        "requirements": list(spec["requirements"]),
        "wheels": list(spec["wheels"]),
        "expected_versions": dict(spec["expected_versions"]),
        "required_symbols": [encode_symbol_check(c) for c in spec["required_symbols"]],
        "smoke_commands": list(spec["smoke_commands"]),
        "labels": dict(spec["labels"]),
        "project": spec["project"],
    }


def decode_image_spec(value: JSONValue) -> ImageSpec:
    """Decode and validate an image spec.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated spec.

    Raises:
        JSONTypeError: If the document is not an object, a required field is
            missing or of the wrong type, a requirement is not exactly
            pinned, a wheel name carries a path separator, the env prefix
            sits under a host-bound root, or either assertion list is empty.
            Nothing is defaulted: a field absent from the document is a field
            the author did not decide, and deciding it here would put a value
            into an image that no document records.
    """
    obj = require_json_object(value, "image spec")
    return ImageSpec(
        base_image=require_digest_pinned_image(obj, "base_image"),
        env_prefix=require_container_dir(obj, "env_prefix"),
        git_commit=require_non_empty_str(obj, "git_commit"),
        system_packages=require_pinned_system_packages(obj, "system_packages"),
        extra_index_urls=require_str_list(obj, "extra_index_urls"),
        requirements=require_pinned_requirements(obj, "requirements"),
        wheels=require_bare_filenames(obj, "wheels"),
        expected_versions=require_str_map(obj, "expected_versions", allow_empty=False),
        required_symbols=_require_symbol_checks(obj, "required_symbols"),
        smoke_commands=require_str_list(obj, "smoke_commands"),
        labels=require_str_map(obj, "labels", allow_empty=True),
        project=require_project(obj, "project"),
    )


__all__ = [
    "ImageSpec",
    "SymbolCheck",
    "decode_image_spec",
    "encode_image_spec",
    "encode_symbol_check",
    "require_symbol_check",
]
