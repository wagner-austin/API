"""The image contract: what a research environment is, as a document.

A directory pinned by convention cannot say what it held when a run used it.
``/pub/wagnera3/envs/abl-pinned`` is the concrete case: nothing prevented a
later install from mutating it, nothing recorded which commit built it, and
every artifact HPC3 has produced carries ``git_commit`` null as a result. The
published arms of the extraction ablation additionally straddle two torch
major versions, which no lock file records because a lock states intent and a
manifest states fact.

An image answers that by being content-addressed. This module is the document
that describes one, and the rules below are the ones that make the description
worth trusting:

* Every requirement carries an exact ``==`` pin. An unpinned requirement
  resolves at build time, which reintroduces the drift the image exists to
  remove -- silently, because the build still succeeds.
* Wheels are bare filenames. They are joined onto a directory inside the
  build, so a separator would reach outside it.
* The commit is non-empty. The trainer reads an empty ``GIT_COMMIT`` as "not
  stamped" and records null, so an image stamped with an empty string claims
  provenance it does not have -- worse than admitting none.
* At least one version assertion and one symbol assertion. An image that
  checks nothing about itself is one whose staleness is discovered by a job
  that already waited for a GPU.

What this document deliberately does NOT describe: corpora, artifacts, GPU
model, or driver. Data is bind-mounted and identified by digest
(:mod:`hpc3.contracts.stage`); hardware is an axis containers do not control,
and belongs to comparability rather than to the image.
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

_PIN_SEPARATOR = "=="


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
            the image. Never under a host-bound path: HPC3 auto-binds
            ``/pub``, so an env at ``/pub/...`` would be shadowed at runtime
            by the host filesystem and the image's own interpreter would
            disappear behind it.
        git_commit: Commit the wheels were built from. Written into the image
            so the trainer can stamp it into every manifest.
        extra_index_urls: Additional package indexes, in order. Present
            because a local version such as ``torch==2.6.0+cu124`` is
            published only on the PyTorch index.
        requirements: Third-party layer, every line an exact ``==`` pin.
        wheels: First-party wheel filenames, installed with dependencies
            already pinned above.
        expected_versions: Package name to exact version the built image must
            report. Checked inside the image, at build time.
        required_symbols: Attributes that must exist in the built image.
        labels: Free-form metadata recorded on the image.
    """

    base_image: str
    env_prefix: str
    git_commit: str
    extra_index_urls: list[str]
    requirements: list[str]
    wheels: list[str]
    expected_versions: dict[str, str]
    required_symbols: list[SymbolCheck]
    labels: dict[str, str]


def _require_object(value: JSONValue, what: str) -> JSONObject:
    """Narrow a decoded JSON value to an object.

    Args:
        value: Value produced by the JSON loader.
        what: Name of the thing being decoded, used in the error message.

    Returns:
        The value as a JSON object.

    Raises:
        JSONTypeError: If the value is not a JSON object.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{what} must be a JSON object, got {type(value).__name__}")
    return value


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
            backslashed, holds a ``..`` segment, or sits under a path the
            cluster bind-mounts. The last is the load-bearing one: HPC3
            binds ``/pub`` into every container, so an env prefix there is
            replaced at runtime by the host directory and the image's
            interpreter ceases to exist inside its own image.
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


HOST_BOUND_ROOTS = frozenset({"pub", "dfs6b", "data", "tmp", "home"})


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
    obj = _require_object(value, where)
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
        "extra_index_urls": list(spec["extra_index_urls"]),
        "requirements": list(spec["requirements"]),
        "wheels": list(spec["wheels"]),
        "expected_versions": dict(spec["expected_versions"]),
        "required_symbols": [encode_symbol_check(c) for c in spec["required_symbols"]],
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
    obj = _require_object(value, "image spec")
    return ImageSpec(
        base_image=_require_non_empty_str(obj, "base_image"),
        env_prefix=_require_container_dir(obj, "env_prefix"),
        git_commit=_require_non_empty_str(obj, "git_commit"),
        extra_index_urls=_require_str_list(obj, "extra_index_urls"),
        requirements=_require_pinned_requirements(obj, "requirements"),
        wheels=_require_bare_filenames(obj, "wheels"),
        expected_versions=_require_str_map(obj, "expected_versions", allow_empty=False),
        required_symbols=_require_symbol_checks(obj, "required_symbols"),
        labels=_require_str_map(obj, "labels", allow_empty=True),
    )


__all__ = [
    "HOST_BOUND_ROOTS",
    "ImageSpec",
    "SymbolCheck",
    "decode_image_spec",
    "encode_image_spec",
    "encode_symbol_check",
    "require_symbol_check",
]
