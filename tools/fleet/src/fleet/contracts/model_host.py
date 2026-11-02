"""What a node declares for serving catalog models, and the catalog row a model job carries.

MCPs board task b9235198. The dispatch queue's ``model-load`` and
``model-unload`` jobs (MCPs mig 896, board task 62b1bb0a) name a model of
the catalog committed in MCPs corvis-db (``fleet-dispatch/model-catalog.ts``)
and are claimed on the ``models`` lane by the runner of the node the queue
placed them on (:mod:`fleet.cli.node_models`). The claim hands the job its
catalog row as ``model``, so this package keeps no catalog of its own:
:class:`CatalogModel` is that row, decoded strictly.

A NODE SAYS WHERE ITS SERVERS ARE. A machine file's ``dispatch`` object
declares ``models``: null for a node that serves none, or a
:class:`NodeModels` naming each server binary it has built, the backend
they were built for, the directory holding the catalog's files, the ports
its models are served on and how long a model may go unused before it is
stopped. Null is spelled out, as ``gpu`` is (:mod:`fleet.contracts.node`): a
node whose file says nothing about a server has none, and a reinstall writes
null (MCPs maketools ``node_capacity.dispatch_object``), since it leaves no
build behind.

TWO SERVERS (MCPs board task 228aa097). A catalog row names the server that
runs it, :class:`ModelServer`: llama.cpp's ``llama-server`` for a GGUF chat
model, served at the row's context, and whisper.cpp's ``whisper-server``
for a ggml speech-to-text model, which takes no context, so its row's
``contextTokens`` is null. A node declares a binary per server it has built
(``servers``), and a load of a model whose server the node has not built is
refused before anything starts.
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


class ModelServer(StrEnum):
    """The server a catalog model runs in, by its binary's name.

    MCPs corvis-db's ``ModelServer`` names the same two.
    """

    LLAMA = "llama-server"
    WHISPER = "whisper-server"


#: The extension of each server's model file: llama.cpp reads GGUF, and
#: whisper.cpp's published models are ggml ``.bin`` files.
_FILE_SUFFIX: dict[ModelServer, str] = {
    ModelServer.LLAMA: ".gguf",
    ModelServer.WHISPER: ".bin",
}


#: The ports a node's models may be served on: unprivileged, and a range no
#: wider than this, since each one is a tailnet grant (MCPs fleet-mcp
#: ``tailnet-policy.source.hujson``).
_PORT_FLOOR = 1024
_PORT_CEILING = 65535
_MAX_PORTS = 16


class NodeModels(TypedDict):
    """How a node serves catalog models.

    Attributes:
        servers: Absolute path of each server binary the node has built,
            by the server it is; at least one.
        backend: The backend those binaries were built for, which a
            model's catalog row must list.
        cache: Absolute directory holding the catalog's model files, each
            under its catalog file name.
        first_port: The lowest port a model is served on.
        last_port: The highest; each loaded model takes the lowest free one,
            and a node serves at most this many models at once.
        idle_seconds: How long a served model may go without a request
            before its runner stops it with no unload job.
    """

    servers: dict[ModelServer, str]
    backend: ModelBackend
    cache: str
    first_port: int
    last_port: int
    idle_seconds: int


class CatalogModel(TypedDict):
    """One catalog model, as a claimed model job carries it.

    Attributes:
        model_id: The catalog id, the job's project after ``models/``.
        server: The server that runs it.
        file: The model file name under the node's model cache: GGUF for
            llama-server, ggml ``.bin`` for whisper-server.
        sha256: The digest that file must hash to, lowercase hex.
        size_bytes: The file's size.
        gpu_bytes: The GPU memory it takes once served.
        context_tokens: The context a llama-server model is served at;
            None for a whisper-server model, which takes none.
        backends: The backends ``gpu_bytes`` was measured on.
    """

    model_id: str
    server: ModelServer
    file: str
    sha256: str
    size_bytes: int
    gpu_bytes: int
    context_tokens: int | None
    backends: tuple[ModelBackend, ...]


def _absolute(path: str, field: str) -> str:
    """Check a declared path is absolute, with no quote or space.

    The path is rendered into a node script, quoted (:func:`shlex.quote`),
    so a quote could not inject anything; it is refused all the same,
    because a declared path holding one was not written by hand on purpose.

    Args:
        path: The declared path.
        field: Its field under ``models``, for the error.

    Returns:
        The path.

    Raises:
        JSONTypeError: When it is not such a path.
    """
    if re.fullmatch(r"/[A-Za-z0-9._/-]+", path) is None:
        raise JSONTypeError(
            f"models.{field} must be an absolute path of letters, digits, '.', '_', '-' and '/', "
            f"got {path!r}"
        )
    return path


def _servers(value: JSONObject) -> dict[ModelServer, str]:
    """Read a declaration's ``servers``: each built server's binary.

    Args:
        value: The ``models`` object.

    Returns:
        Each binary's absolute path, by its server.

    Raises:
        JSONTypeError: When it is not a non-empty object, a key is not one
            of :class:`ModelServer`, or a path is not a string or not plain
            and absolute.
    """
    raw = value.get("servers")
    if not isinstance(raw, dict) or not raw:
        raise JSONTypeError(f"models.servers must be a non-empty JSON object, got {raw!r}")
    servers: dict[ModelServer, str] = {}
    for name, path in raw.items():
        server = find_member(name, ModelServer)
        if server is None:
            raise JSONTypeError(
                f"models.servers names {name!r}, which is not one of {', '.join(ModelServer)}"
            )
        if not isinstance(path, str):
            raise JSONTypeError(f"models.servers.{name} must be a string, got {path!r}")
        servers[server] = _absolute(path, f"servers.{name}")
    return servers


def decode_node_models(value: JSONValue) -> NodeModels | None:
    """Decode a ``dispatch`` object's ``models``.

    Args:
        value: The field's value: null, or the object.

    Returns:
        The declaration, or None for a node that serves no model.

    Raises:
        JSONTypeError: When it is neither null nor an object, a field is
            missing or mistyped, a server is not one of :class:`ModelServer`,
            the backend is not one of :class:`ModelBackend`, the ports are
            not an unprivileged range of at most :data:`_MAX_PORTS`, or the
            idle span is not positive.
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
        servers=_servers(value),
        backend=require_member(value, "backend", ModelBackend),
        cache=_absolute(require_str(value, "cache"), "cache"),
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
        "servers": {server.value: path for server, path in models["servers"].items()},
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


def _require_server(row: dict[str, JSONValue], *, answer: str) -> ModelServer:
    """Read the server a catalog row names.

    Args:
        row: The ``model`` object.
        answer: The whole answer, for the error message.

    Returns:
        The server.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when it is not one of
            :class:`ModelServer`.
    """
    raw = row.get("server")
    server = find_member(raw, ModelServer) if isinstance(raw, str) else None
    if server is None:
        raise malformed(
            f"model server {raw!r} is not one of {', '.join(ModelServer)}", answer=answer
        )
    return server


def _require_context(row: dict[str, JSONValue], server: ModelServer, *, answer: str) -> int | None:
    """Read a catalog row's context, which its server decides.

    Args:
        row: The ``model`` object.
        server: The server it names.
        answer: The whole answer, for the error message.

    Returns:
        A positive count for llama-server; None for whisper-server.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when it is not that.
    """
    if server is ModelServer.LLAMA:
        return _require_count(row, "contextTokens", answer=answer)
    if "contextTokens" not in row or row["contextTokens"] is not None:
        raise malformed(
            f"model field 'contextTokens' is {row.get('contextTokens')!r}, not null, "
            f"for a {server} model, which takes no context",
            answer=answer,
        )
    return None


def decode_catalog_model(value: JSONValue, *, answer: str) -> CatalogModel:
    """Decode the ``model`` a model job's wire row carries.

    Args:
        value: The field's value.
        answer: The whole answer, for the error message.

    Returns:
        The catalog row.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when it is not an object, a
            field is missing or mistyped, the server is not one of
            :class:`ModelServer`, the id or file is not a plain name or the
            file not its server's kind, the digest is not 64 lowercase hex,
            a backend is not one of :class:`ModelBackend`, or the context is
            not a positive count for llama-server and null for
            whisper-server: the corvis-db catalog and this vocabulary are
            twins, so a stranger means one moved alone.
    """
    if not isinstance(value, dict):
        raise malformed(
            f"a model job's model is {type(value).__name__}, not an object", answer=answer
        )
    model_id = require_answer_str(value, "id", answer=answer)
    if re.fullmatch(r"[a-z0-9][a-z0-9.-]*", model_id) is None:
        raise malformed(f"model id {model_id!r} is not a catalog id", answer=answer)
    server = _require_server(value, answer=answer)
    file = require_answer_str(value, "file", answer=answer)
    suffix = _FILE_SUFFIX[server]
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*" + re.escape(suffix), file) is None:
        raise malformed(
            f"model file {file!r} is not a plain {suffix} file name, which {server} reads",
            answer=answer,
        )
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
    context_tokens = _require_context(value, server, answer=answer)
    return CatalogModel(
        model_id=model_id,
        server=server,
        file=file,
        sha256=sha256,
        size_bytes=_require_count(value, "sizeBytes", answer=answer),
        gpu_bytes=_require_count(value, "gpuBytes", answer=answer),
        context_tokens=context_tokens,
        backends=tuple(backends),
    )


__all__ = [
    "CatalogModel",
    "ModelBackend",
    "ModelServer",
    "NodeModels",
    "decode_catalog_model",
    "decode_node_models",
    "encode_node_models",
]
