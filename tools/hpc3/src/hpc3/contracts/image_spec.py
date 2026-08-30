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
    require_dict,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from hpc3.contracts.image import HOST_BOUND_ROOTS, require_json_object

_PIN_SEPARATOR = "=="

#: How a distribution's package manager spells an exact pin. One ``=``, not
#: two: ``apt-get install xvfb=2:21.1.4-2ubuntu1.7`` is the syntax, and
#: writing pip's separator here installs nothing and reports success.
_SYSTEM_PIN_SEPARATOR = "="

#: Characters that would turn one package specification into something else
#: once interpolated into the build script.
_SHELL_METACHARACTERS = (" ", "\t", ";", "&", "|", "$", "`", "(", ")", "<", ">", "\n")


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
        base_image: Docker reference the build bootstraps from, digest- or
            tag-qualified. The CUDA runtime arrives as ``nvidia-*-cu12``
            wheels in ``requirements``, so this does not need to be a CUDA
            base image and should not be one.
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


def _require_non_empty_str(obj: JSONObject, key: str) -> str:
    """Read a required string field that must carry characters.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, or empty. An
            empty value here is never a usable default -- it is a field
            somebody forgot to fill, and accepting it defers the failure to
            a build or a run.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    return value


def _require_container_dir(obj: JSONObject, key: str) -> str:
    """Read a required absolute POSIX directory that lives inside the image.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value, without a trailing slash.

    Raises:
        JSONTypeError: If the field is missing, not a string, not absolute,
            backslashed, holds a ``..`` segment, or sits under a root a job
            binds. The last is the load-bearing one: a job mounts its data
            directories at the same paths inside the image, so an
            environment under one of them is replaced at runtime by the host
            directory and the image's interpreter ceases to exist inside its
            own image.

            An earlier version of this docstring said the CLUSTER
            auto-mounts ``/pub``. That was wrong, measured 2026-08-25: HPC3
            binds the BeeGFS mounts but ``/pub`` there is a symlink to
            ``/dfs6b/pub`` and the symlink is not carried, so an unbound
            ``/pub/...`` does not resolve inside a container at all. The
            rule is unchanged and the shadowing is real -- it is the job's
            own ``binds`` that do it, not the cluster.
    """
    value = require_str(obj, key)
    if not value.startswith("/"):
        raise JSONTypeError(f"Field '{key}' must be an absolute POSIX path, got {value!r}")
    if "\\" in value:
        raise JSONTypeError(f"Field '{key}' must be forward-slashed, got {value!r}")
    if ".." in value.split("/"):
        raise JSONTypeError(f"Field '{key}' must not contain '..', got {value!r}")
    trimmed = value.rstrip("/")
    if trimmed == "":
        raise JSONTypeError(f"Field '{key}' must not be the filesystem root")
    first_segment = trimmed.split("/")[1]
    if first_segment in HOST_BOUND_ROOTS:
        raise JSONTypeError(
            f"Field '{key}' is under /{first_segment}, which the cluster bind-mounts over; "
            f"the image's own copy would be shadowed at runtime, got {value!r}"
        )
    return trimmed


def _require_pinned_requirements(obj: JSONObject, key: str) -> list[str]:
    """Read the third-party layer, refusing anything not exactly pinned.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The requirement lines, in order.

    Raises:
        JSONTypeError: If the field is missing, not a list, empty, holds a
            non-string, holds a blank line, or holds a requirement without an
            exact ``==`` pin. The pin check is the point of this validator: a
            requirement resolved at build time reintroduces exactly the drift
            the image exists to remove, and it does so without failing.
    """
    raw = require_list(obj, key)
    if not raw:
        raise JSONTypeError(f"Field '{key}' must not be empty")
    lines: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise JSONTypeError(
                f"Field '{key}[{index}]' must be a string, got {type(item).__name__}"
            )
        line = item.strip()
        if line == "":
            raise JSONTypeError(f"Field '{key}[{index}]' must not be blank")
        if _PIN_SEPARATOR not in line:
            raise JSONTypeError(
                f"Field '{key}[{index}]' must pin an exact version with "
                f"'{_PIN_SEPARATOR}', got {line!r}"
            )
        lines.append(line)
    return lines


def _require_pinned_system_packages(obj: JSONObject, key: str) -> list[str]:
    """Read the operating-system layer, refusing anything not exactly pinned.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The package specifications, in order.

    Raises:
        JSONTypeError: If the field is missing, not a list, holds a
            non-string, holds a blank entry, holds an entry without an exact
            ``=`` pin, or holds a shell metacharacter.

            The pin check is the same argument as the pip layer's: an
            unpinned ``apt-get install`` resolves against whatever the
            distribution serves that day and SUCCEEDS, so two images built a
            week apart differ with nothing recording that they do.

            The metacharacter check is not about a hostile spec -- these are
            written by whoever builds the image -- but about a spec that is
            wrong in a way the shell would act on. These names are
            interpolated into the build script, so a stray space or semicolon
            becomes a second command rather than a package that does not
            exist, and the build reports success having installed nothing.

    Note:
        Empty is permitted: an image whose whole dependency set is wheels is
        the ordinary case, and requiring a package would force one to be
        invented.
    """
    raw = require_list(obj, key)
    packages: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise JSONTypeError(
                f"Field '{key}[{index}]' must be a string, got {type(item).__name__}"
            )
        entry = item.strip()
        if entry == "":
            raise JSONTypeError(f"Field '{key}[{index}]' must not be blank")
        if _SYSTEM_PIN_SEPARATOR not in entry:
            raise JSONTypeError(
                f"Field '{key}[{index}]' must pin an exact version with "
                f"'{_SYSTEM_PIN_SEPARATOR}', got {entry!r}"
            )
        found = [character for character in _SHELL_METACHARACTERS if character in entry]
        if found:
            raise JSONTypeError(
                f"Field '{key}[{index}]' must not contain {''.join(found)!r}: the entry is "
                f"interpolated into the build script, got {entry!r}"
            )
        packages.append(entry)
    return packages


def _require_bare_filenames(obj: JSONObject, key: str) -> list[str]:
    """Read a required non-empty list of filenames carrying no separator.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The filenames, in order.

    Raises:
        JSONTypeError: If the field is missing, not a list, empty, holds a
            non-string, or holds a name with a path separator or a
            relative-navigation component. These are joined onto a directory
            inside the build, so a separator would reach outside it.
    """
    raw = require_list(obj, key)
    if not raw:
        raise JSONTypeError(f"Field '{key}' must not be empty")
    names: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise JSONTypeError(
                f"Field '{key}[{index}]' must be a string, got {type(item).__name__}"
            )
        if item == "":
            raise JSONTypeError(f"Field '{key}[{index}]' must not be empty")
        if "/" in item or "\\" in item:
            raise JSONTypeError(
                f"Field '{key}[{index}]' must not contain a path separator, got {item!r}"
            )
        if item in (".", ".."):
            raise JSONTypeError(f"Field '{key}[{index}]' must name a file, got {item!r}")
        names.append(item)
    return names


def _require_str_list(obj: JSONObject, key: str) -> list[str]:
    """Read a required list of strings, which may be empty.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The values, in order.

    Raises:
        JSONTypeError: If the field is missing, not a list, or holds a
            non-string. Emptiness is permitted here because a build with no
            extra index is the ordinary case.
    """
    raw = require_list(obj, key)
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise JSONTypeError(
                f"Field '{key}[{index}]' must be a string, got {type(item).__name__}"
            )
        values.append(item)
    return values


def _require_str_map(obj: JSONObject, key: str, *, allow_empty: bool) -> dict[str, str]:
    """Read a required object whose every value is a string.

    Args:
        obj: Object being decoded.
        key: Field name.
        allow_empty: Whether an empty mapping is acceptable.

    Returns:
        The mapping, with insertion order preserved.

    Raises:
        JSONTypeError: If the field is missing, not an object, holds a
            non-string value, or is empty when ``allow_empty`` is False.
    """
    raw = require_dict(obj, key)
    if not allow_empty and not raw:
        raise JSONTypeError(f"Field '{key}' must not be empty")
    mapping: dict[str, str] = {}
    for name, value in raw.items():
        if not isinstance(value, str):
            raise JSONTypeError(
                f"Field '{key}[{name!r}]' must be a string, got {type(value).__name__}"
            )
        if value == "":
            raise JSONTypeError(f"Field '{key}[{name!r}]' must not be empty")
        mapping[name] = value
    return mapping


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
        module=_require_non_empty_str(obj, "module"),
        attribute=_require_non_empty_str(obj, "attribute"),
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
        base_image=_require_non_empty_str(obj, "base_image"),
        env_prefix=_require_container_dir(obj, "env_prefix"),
        git_commit=_require_non_empty_str(obj, "git_commit"),
        system_packages=_require_pinned_system_packages(obj, "system_packages"),
        extra_index_urls=_require_str_list(obj, "extra_index_urls"),
        requirements=_require_pinned_requirements(obj, "requirements"),
        wheels=_require_bare_filenames(obj, "wheels"),
        expected_versions=_require_str_map(obj, "expected_versions", allow_empty=False),
        required_symbols=_require_symbol_checks(obj, "required_symbols"),
        smoke_commands=_require_str_list(obj, "smoke_commands"),
        labels=_require_str_map(obj, "labels", allow_empty=True),
    )


__all__ = [
    "ImageSpec",
    "SymbolCheck",
    "decode_image_spec",
    "encode_image_spec",
    "encode_symbol_check",
    "require_symbol_check",
]
