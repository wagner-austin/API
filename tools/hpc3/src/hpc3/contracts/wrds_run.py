"""Explicit, digest-bound Python experiments for WRDS Grid Engine.

Local source paths locate bytes; they are not part of the experiment identity.
The identity covers their destination names and digests, every remote input,
the interpreter and package pins, arguments, seed, resources and outputs.
"""

from __future__ import annotations

import hashlib
import pathlib
import re

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    narrow_json_to_dict,
    narrow_json_to_str,
    require_dict,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from hpc3.contracts.pins import encode_pinned_packages, require_pinned_packages


class WrdsFile(TypedDict):
    """One local code file, or remote input, with its expected contents."""

    path: str
    name: str
    sha256: str


class WrdsRuntime(TypedDict):
    """Required interpreter bytes, version, and installed package versions."""

    python: str
    sha256: str
    version: str
    packages: dict[str, str]


class WrdsRun(TypedDict):
    """One experiment, independent of the attempt that executes it."""

    project: str
    name: str
    host: str
    root: str
    runtime: WrdsRuntime
    slots: int
    memory_gb_per_slot: int
    seconds: int
    seed: int
    code: list[WrdsFile]
    inputs: list[WrdsFile]
    entrypoint: str
    arguments: list[str]
    outputs: list[str]


def safe_label(value: str) -> str:
    """Validate a label used as one path component.

    Args:
        value: Project, run or attempt label.

    Returns:
        The validated label.

    Raises:
        ValueError: For an empty, traversing or shell-shaped label.
    """
    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value) is None:
        raise ValueError(f"WRDS_LABEL: invalid label {value!r}")
    return value


def relative_path(value: str) -> str:
    """Require a nonempty normalized relative file path.

    Args:
        value: Name beneath a code or output directory.

    Returns:
        The validated path.

    Raises:
        ValueError: For traversal, absolute paths or noncanonical spelling.
    """
    path = pathlib.PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ValueError(f"WRDS_PATH: invalid relative path {value!r}")
    if any(character in value for character in ("\n", "\r", "\x00", "\\")):
        raise ValueError(f"WRDS_PATH: unsupported path characters {value!r}")
    return value


def absolute_path(value: str) -> str:
    """Require a normalized absolute POSIX path.

    Args:
        value: A remote directory, executable or input file.

    Returns:
        Validated path.

    Raises:
        ValueError: When the path is relative or traverses upward.
    """
    if not value.startswith("/"):
        raise ValueError(f"WRDS_PATH: expected absolute path, got {value!r}")
    relative_path(value[1:])
    return value


def digest(value: str) -> str:
    """Validate a SHA256 digest.

    Args:
        value: Lowercase hexadecimal digest.

    Returns:
        Validated digest.

    Raises:
        ValueError: When the digest is malformed.
    """
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("WRDS_DIGEST: expected 64 lowercase hex characters")
    return value


def decode_wrds_file(value: JSONValue) -> WrdsFile:
    """Decode a named file pin.

    Args:
        value: JSON file object.

    Returns:
        A validated pin.

    Raises:
        ValueError: When a name, source or digest is invalid.
    """
    obj = narrow_json_to_dict(value)
    path = require_str(obj, "path")
    if not path:
        raise ValueError("WRDS_PATH: source path cannot be empty")
    return WrdsFile(
        path=path,
        name=relative_path(require_str(obj, "name")),
        sha256=digest(require_str(obj, "sha256")),
    )


def encode_wrds_file(value: WrdsFile) -> JSONObject:
    """Encode a file pin.

    Args:
        value: Validated file pin.

    Returns:
        JSON object.
    """
    return {"path": value["path"], "name": value["name"], "sha256": value["sha256"]}


def decode_wrds_runtime(value: JSONValue) -> WrdsRuntime:
    """Decode a complete environment pin.

    Args:
        value: JSON runtime object.

    Returns:
        The expected runtime.

    Raises:
        ValueError: When its version or executable identity is absent.
    """
    obj = narrow_json_to_dict(value)
    version = require_str(obj, "version")
    if not version:
        raise ValueError("WRDS_RUNTIME: Python version is required")
    return WrdsRuntime(
        python=absolute_path(require_str(obj, "python")),
        sha256=digest(require_str(obj, "sha256")),
        version=version,
        packages=require_pinned_packages(obj, "packages"),
    )


def encode_wrds_runtime(value: WrdsRuntime) -> JSONObject:
    """Encode a runtime pin.

    Args:
        value: Expected or measured runtime.

    Returns:
        JSON object with normalized package names.
    """
    return {
        "python": value["python"],
        "sha256": value["sha256"],
        "version": value["version"],
        "packages": encode_pinned_packages(value["packages"]),
    }


def _unique_paths(names: list[str]) -> None:
    """Refuse duplicate paths and file/directory collisions.

    Args:
        names: Relative paths in one namespace.

    Raises:
        ValueError: When two files claim the same location or an ancestor.
    """
    if len(set(names)) != len(names):
        raise ValueError("WRDS_DUPLICATE: duplicate file names")
    for name in names:
        if any(str(parent) in names for parent in pathlib.PurePosixPath(name).parents):
            raise ValueError("WRDS_DUPLICATE: file/directory collision")


def decode_wrds_run(value: JSONValue) -> WrdsRun:
    """Decode a run and enforce the measured WRDS resource envelope.

    Args:
        value: Run document.

    Returns:
        A complete validated experiment.

    Raises:
        ValueError: For unsafe paths, missing entrypoints, inconsistent file
            declarations, unknown fields or excessive resources.
    """
    obj = narrow_json_to_dict(value)
    expected = {
        "project",
        "name",
        "host",
        "root",
        "runtime",
        "slots",
        "memory_gb_per_slot",
        "seconds",
        "seed",
        "code",
        "inputs",
        "entrypoint",
        "arguments",
        "outputs",
    }
    if set(obj) != expected:
        raise ValueError(f"WRDS_FIELDS: required fields differ: {sorted(set(obj) ^ expected)}")
    slots, memory = require_int(obj, "slots"), require_int(obj, "memory_gb_per_slot")
    seconds, seed = require_int(obj, "seconds"), require_int(obj, "seed")
    if not (1 <= slots <= 8 and memory >= 1 and slots * memory <= 48):
        raise ValueError("WRDS_RESOURCES: require 1..8 slots and at most 48 GiB total")
    if not 1 <= seconds <= 604800 or not 0 <= seed <= 4294967295:
        raise ValueError("WRDS_RESOURCES: runtime must be 1..604800 seconds and seed uint32")
    root = absolute_path(require_str(obj, "root"))
    if not root.startswith("/home/"):
        raise ValueError("WRDS_ROOT: durable run directories must be beneath /home")
    code = [decode_wrds_file(item) for item in require_list(obj, "code")]
    inputs = [decode_wrds_file(item) for item in require_list(obj, "inputs")]
    for item in inputs:
        absolute_path(item["path"])
    outputs = [relative_path(narrow_json_to_str(item)) for item in require_list(obj, "outputs")]
    for names in ([item["name"] for item in code], [item["name"] for item in inputs], outputs):
        _unique_paths(names)
    entrypoint = relative_path(require_str(obj, "entrypoint"))
    if entrypoint not in [item["name"] for item in code] or "metrics.json" not in outputs:
        raise ValueError(
            "WRDS_FILES: code must contain entrypoint; outputs must contain metrics.json"
        )
    return WrdsRun(
        project=safe_label(require_str(obj, "project")),
        name=safe_label(require_str(obj, "name")),
        host=safe_label(require_str(obj, "host")),
        root=root,
        runtime=decode_wrds_runtime(require_dict(obj, "runtime")),
        slots=slots,
        memory_gb_per_slot=memory,
        seconds=seconds,
        seed=seed,
        code=code,
        inputs=inputs,
        entrypoint=entrypoint,
        arguments=[narrow_json_to_str(item) for item in require_list(obj, "arguments")],
        outputs=outputs,
    )


def encode_wrds_run(run: WrdsRun) -> JSONObject:
    """Encode a run with stable field and file ordering.

    Args:
        run: Validated experiment.

    Returns:
        Its complete JSON representation.
    """
    return {
        "project": run["project"],
        "name": run["name"],
        "host": run["host"],
        "root": run["root"],
        "runtime": encode_wrds_runtime(run["runtime"]),
        "slots": run["slots"],
        "memory_gb_per_slot": run["memory_gb_per_slot"],
        "seconds": run["seconds"],
        "seed": run["seed"],
        "code": [encode_wrds_file(item) for item in run["code"]],
        "inputs": [encode_wrds_file(item) for item in run["inputs"]],
        "entrypoint": run["entrypoint"],
        "arguments": list(run["arguments"]),
        "outputs": list(run["outputs"]),
    }


def execution_document(run: WrdsRun) -> JSONObject:
    """Remove machine-local source locations from the portable execution spec.

    Args:
        run: Local run document.

    Returns:
        The remote document, also used for identity hashing.
    """
    value = encode_wrds_run(run)
    value["code"] = [
        {"path": item["name"], "name": item["name"], "sha256": item["sha256"]}
        for item in run["code"]
    ]
    return value


def run_identity(run: WrdsRun) -> str:
    """Hash the portable experiment definition.

    Args:
        run: Validated experiment.

    Returns:
        SHA256 of the exact execution document.
    """
    return hashlib.sha256(dump_json_str(execution_document(run)).encode()).hexdigest()


__all__ = [
    "WrdsFile",
    "WrdsRun",
    "WrdsRuntime",
    "absolute_path",
    "decode_wrds_file",
    "decode_wrds_run",
    "decode_wrds_runtime",
    "digest",
    "encode_wrds_file",
    "encode_wrds_run",
    "encode_wrds_runtime",
    "execution_document",
    "relative_path",
    "run_identity",
    "safe_label",
]
