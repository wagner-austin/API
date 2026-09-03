"""How one image spec's fields are read, and what each one refuses.

Split from :mod:`hpc3.contracts.image_spec` when that module passed the
600-line ceiling. The seam is by role rather than by size: this module says
how a field is READ and why a value is refused; its neighbour says what a
spec IS and how one round-trips.

Every reader here refuses rather than repairs. A spec is the document a build
is reproduced from, so a value this cannot read is a build nobody can repeat,
and guessing what was meant is how two images built a week apart differ with
nothing to say they do.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_dict,
    require_list,
    require_str,
)

from hpc3.contracts.image import HOST_BOUND_ROOTS, SHA256_HEX_LENGTH, is_sha256_hex

_PIN_SEPARATOR = "=="

_SYSTEM_PIN_SEPARATOR = "="

_SHELL_METACHARACTERS = (" ", "\t", ";", "&", "|", "$", "`", "(", ")", "<", ">", "\n")

DIGEST_SEPARATOR = "@sha256:"
"""What separates a base image reference from the digest that pins it."""


def require_non_empty_str(obj: JSONObject, key: str) -> str:
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


def require_container_dir(obj: JSONObject, key: str) -> str:
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


def require_pinned_requirements(obj: JSONObject, key: str) -> list[str]:
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


def require_pinned_system_packages(obj: JSONObject, key: str) -> list[str]:
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


def require_bare_filenames(obj: JSONObject, key: str) -> list[str]:
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


def require_str_list(obj: JSONObject, key: str) -> list[str]:
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


def require_str_map(obj: JSONObject, key: str, *, allow_empty: bool) -> dict[str, str]:
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


def require_digest_pinned_image(obj: JSONObject, key: str) -> str:
    """Read a base image reference that names an immutable digest.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The reference, unchanged.

    Raises:
        JSONTypeError: If the field is missing, empty, carries no
            ``@sha256:`` digest, or the digest is not 64 lowercase hex
            characters. Refused rather than resolved: see
            :class:`ImageSpec.base_image` for why a tag is not a pin and why
            resolving one here would be worse than refusing it.
    """
    value = require_non_empty_str(obj, key)
    reference, separator, digest = value.partition(DIGEST_SEPARATOR)
    if separator == "" or reference == "":
        raise JSONTypeError(
            f"Field '{key}' must pin a digest as '<image>{DIGEST_SEPARATOR}<64 hex>', "
            f"got {value!r}. A tag is a mutable pointer: two builds of this spec can "
            "start from different bytes and neither would say so."
        )
    if not is_sha256_hex(digest):
        raise JSONTypeError(
            f"Field '{key}' digest must be {SHA256_HEX_LENGTH} lowercase hex "
            f"characters, got {digest!r}"
        )
    return value


__all__ = [
    "DIGEST_SEPARATOR",
    "require_bare_filenames",
    "require_container_dir",
    "require_digest_pinned_image",
    "require_non_empty_str",
    "require_pinned_requirements",
    "require_pinned_system_packages",
    "require_str_list",
    "require_str_map",
]
