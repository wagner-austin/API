"""What a node declares for serving catalog models, and the catalog row a model job carries.

MCPs board task b9235198. The dispatch queue's ``model-load`` and
``model-unload`` jobs (MCPs mig 896, board task 62b1bb0a) name a model of
the catalog committed in MCPs corvis-db (``fleet-dispatch/model-catalog.ts``)
and are claimed on the ``models`` lane by the runner of the node the queue
placed them on (:mod:`fleet.cli.node_models`). The claim hands the job its
catalog row as ``model``, so this package keeps no catalog of its own:
:class:`CatalogModel` is that row, decoded strictly.

A NODE SAYS WHERE ITS SERVER IS. A machine file's ``dispatch`` object
declares ``models``: null for a node that serves none, or a
:class:`NodeModels` naming the llama.cpp server binary, the backend it was
built for, the directory holding the catalog's files, the ports its models
are served on and how long a model may go unused before it is stopped.
Null is spelled out, as ``gpu`` is (:mod:`fleet.contracts.node`): a node
whose file says nothing about a server has none, and a reinstall writes null
(MCPs maketools ``node_capacity.dispatch_object``), since it leaves no build
behind.
"""

from __future__ import annotations

import re
from enum import StrEnum

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_str,
)
from platform_core.members import find_member, require_member
from typing_extensions import TypedDict

from fleet.contracts.queue_answer import malformed
from fleet.contracts.queue_answer import require_str as require_answer_str


class ModelBackend(StrEnum):
    """The llama.cpp GPU backends a catalog model is served on.

    MCPs corvis-db's ``ModelBackend`` names the same two; colossus's build
    is Vulkan's since its NixOS reinstall (board task b9235198).
    """

    CUDA = "cuda"
    VULKAN = "vulkan"


#: The ports a node's models may be served on: unprivileged, and a range no
#: wider than this, since each one is a tailnet grant (MCPs fleet-mcp
#: ``tailnet-policy.source.hujson``).
_PORT_FLOOR = 1024
_PORT_CEILING = 65535
_MAX_PORTS = 16


class NodeModels(TypedDict):
    """How a node serves catalog models.

    Attributes:
        server: Absolute path of ``llama-server`` on the node.
        backend: The backend that binary was built for, which a model's
            catalog row must list.
        cache: Absolute directory holding the catalog's GGUF files, each
            under its catalog file name.
        first_port: The lowest port a model is served on.
        last_port: The highest; each loaded model takes the lowest free one,
            and a node serves at most this many models at once.
        idle_seconds: How long a served model may go without a completion
            before its runner stops it with no unload job.
    """

    server: str
    backend: ModelBackend
    cache: str
    first_port: int
    last_port: int
    idle_seconds: int


class CatalogModel(TypedDict):
    """One catalog model, as a claimed model job carries it.

    Attributes:
        model_id: The catalog id, the job's project after ``models/``.
        file: The GGUF file name under the node's model cache.
        sha256: The digest that file must hash to, lowercase hex.
        size_bytes: The file's size.
        gpu_bytes: The GPU memory it takes once served at its context.
        context_tokens: The context it is served at.
        backends: The backends ``gpu_bytes`` was measured on.
    """

    model_id: str
    file: str
    sha256: str
    size_bytes: int
    gpu_bytes: int
    context_tokens: int
    backends: tuple[ModelBackend, ...]


def _absolute(value: JSONObject, key: str) -> str:
    """Read a field that must be an absolute path with no quote or space.

    The path is rendered into a node script, quoted (:func:`shlex.quote`),
    so a quote could not inject anything; it is refused all the same,
    because a declared path holding one was not written by hand on purpose.

    Args:
        value: The ``models`` object.
        key: The field.

    Returns:
        The path.

    Raises:
        JSONTypeError: When it is missing, not a string, or not such a path.
    """
    path = require_str(value, key)
    if re.fullmatch(r"/[A-Za-z0-9._/-]+", path) is None:
        raise JSONTypeError(
            f"models.{key} must be an absolute path of letters, digits, '.', '_', '-' and '/', "
            f"got {path!r}"
        )
    return path


def decode_node_models(value: JSONValue) -> NodeModels | None:
    """Decode a ``dispatch`` object's ``models``.

    Args:
        value: The field's value: null, or the object.

    Returns:
        The declaration, or None for a node that serves no model.

    Raises:
        JSONTypeError: When it is neither null nor an object, a field is
            missing or mistyped, the backend is not one of
            :class:`ModelBackend`, the ports are not an unprivileged range of
            at most :data:`_MAX_PORTS`, or the idle span is not positive.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise JSONTypeError(f"models must be null or a JSON object, got {type(value).__name__}")
    first_port = require_int(value, "first_port")
    last_port = require_int(value, "last_port")
    ceiling = min(_PORT_CEILING, first_port + _MAX_PORTS - 1)
    if not _PORT_FLOOR <= first_port <= last_port <= ceiling:
        raise JSONTypeError(
            f"models ports must run from first_port to last_port within {_PORT_FLOOR} to "
            f"{_PORT_CEILING}, at most {_MAX_PORTS} of them, got {first_port} to {last_port}"
        )
    idle_seconds = require_int(value, "idle_seconds")
    if idle_seconds < 1:
        raise JSONTypeError(f"models.idle_seconds must be positive, got {idle_seconds}")
    return NodeModels(
        server=_absolute(value, "server"),
        backend=require_member(value, "backend", ModelBackend),
        cache=_absolute(value, "cache"),
        first_port=first_port,
        last_port=last_port,
        idle_seconds=idle_seconds,
    )


def encode_node_models(models: NodeModels | None) -> JSONValue:
    """Encode a node's ``models``.

    Args:
        models: The declaration, or None.

    Returns:
        Null, or the object its file carries.
    """
    if models is None:
        return None
    return {
        "server": models["server"],
        "backend": models["backend"].value,
        "cache": models["cache"],
        "first_port": models["first_port"],
        "last_port": models["last_port"],
        "idle_seconds": models["idle_seconds"],
    }


def _require_count(row: dict[str, JSONValue], key: str, *, answer: str) -> int:
    """Read a catalog field that must be a positive whole number.

    Args:
        row: The ``model`` object.
        key: The field.
        answer: The whole answer, for the error message.

    Returns:
        The number.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when it is not one.
    """
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise malformed(f"model field {key!r} is {value!r}, not a positive integer", answer=answer)
    return value


def decode_catalog_model(value: JSONValue, *, answer: str) -> CatalogModel:
    """Decode the ``model`` a model job's wire row carries.

    Args:
        value: The field's value.
        answer: The whole answer, for the error message.

    Returns:
        The catalog row.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when it is not an object, a
            field is missing or mistyped, the id or file is not a plain
            name, the digest is not 64 lowercase hex, or a backend is not
            one of :class:`ModelBackend`: the corvis-db catalog and this
            vocabulary are twins, so a stranger means one moved alone.
    """
    if not isinstance(value, dict):
        raise malformed(
            f"a model job's model is {type(value).__name__}, not an object", answer=answer
        )
    model_id = require_answer_str(value, "id", answer=answer)
    if re.fullmatch(r"[a-z0-9][a-z0-9.-]*", model_id) is None:
        raise malformed(f"model id {model_id!r} is not a catalog id", answer=answer)
    file = require_answer_str(value, "file", answer=answer)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.gguf", file) is None:
        raise malformed(f"model file {file!r} is not a plain GGUF file name", answer=answer)
    sha256 = require_answer_str(value, "sha256", answer=answer)
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise malformed(f"model sha256 {sha256!r} is not 64 lowercase hex", answer=answer)
    raw_backends = value.get("backends")
    if not isinstance(raw_backends, list) or not raw_backends:
        raise malformed(f"model backends is {raw_backends!r}, not a non-empty array", answer=answer)
    backends: list[ModelBackend] = []
    for entry in raw_backends:
        backend = find_member(entry, ModelBackend) if isinstance(entry, str) else None
        if backend is None:
            raise malformed(
                f"model backend {entry!r} is not one of {', '.join(ModelBackend)}", answer=answer
            )
        backends.append(backend)
    return CatalogModel(
        model_id=model_id,
        file=file,
        sha256=sha256,
        size_bytes=_require_count(value, "sizeBytes", answer=answer),
        gpu_bytes=_require_count(value, "gpuBytes", answer=answer),
        context_tokens=_require_count(value, "contextTokens", answer=answer),
        backends=tuple(backends),
    )


__all__ = [
    "CatalogModel",
    "ModelBackend",
    "NodeModels",
    "decode_catalog_model",
    "decode_node_models",
    "encode_node_models",
]
