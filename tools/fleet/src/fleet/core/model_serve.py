"""Serving, stopping and watching a catalog model on a node, read back into verdicts.

MCPs board task b9235198. The scripts are :mod:`fleet.core.model_scripts`'s;
this module sends each to the node (:mod:`fleet.core.remote`) and reads the
lines it prints into what the queue is told, so every word a node can print
has one meaning here and a line it cannot print is a fault, never a guess.

WHAT A LOAD BECOMES. A file that is absent, the wrong size or the wrong
sha256 is refused before anything starts, naming the file and both values
(A2): the catalog's digest is the only thing that says the bytes on the node
are the model the job asked for. A server that exits while loading, does
not answer ``/health`` in :data:`fleet.core.model_scripts.HEALTH_SECONDS`, or
is not wholly on the GPU fails, and is stopped. Only a server that answered
on the GPU passes, with the url it answers on, its node's tailnet address
and port: a llama-server with every layer there, or a whisper-server that
logged the GPU device it uses (MCPs board task 228aa097).

WHAT AN UNLOAD READS. The node's GPU memory, by the tick's own probe lines
(:data:`fleet.core.linux_gpu_probe.GPU_PROBE_PART`), before the stop and
after it (:func:`gpu_free_bytes`), so the unload's own record says what the
stop freed: the tick keeps only each runner's latest row, which a later
stop overwrites (audit 7b709022 of board task b9235198).
"""

from __future__ import annotations

import re
from typing import Final

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.dispatch import ClosingStatus
from fleet.contracts.model_host import CatalogModel, ModelServer, NodeModels
from fleet.contracts.node import NodeConfig
from fleet.core import dialect, remote
from fleet.core.linux_gpu_probe import GPU_PROBE_PART, read_gpus
from fleet.core.model_scripts import (
    HEALTH_SECONDS,
    forget_script,
    load_script,
    scan_script,
    unload_script,
)
from fleet.core.probe import read_fields

#: The deadline for a whole load: hashing the largest catalog file, 21.7 GB,
#: well inside five minutes on an NVMe node, then the health wait.
LOAD_DEADLINE_SECONDS: Final[int] = 300 + HEALTH_SECONDS

#: An IPv4 address, as ``tailscale ip -4`` prints the node's.
_ADDRESS = re.compile(r"[0-9]{1,3}(\.[0-9]{1,3}){3}")


class LoadOutcome(TypedDict):
    """What a load on a node came to.

    Attributes:
        status: ``passed`` once it answers, ``refused`` when the file was
            not the catalog's or no port was free, so nothing started, and
            ``failed`` when a server started and did not serve.
        endpoint: The url it answers on, set exactly when it passed.
        detail: The one line the queue keeps.
    """

    status: ClosingStatus
    endpoint: str | None
    detail: str


class ServedModel(TypedDict):
    """A model a node is serving, as its scan read it.

    Attributes:
        model_id: The catalog id.
        job_id: The load that started it.
        port: The port it answers on.
        idle_seconds: How long since its last use.
    """

    model_id: str
    job_id: str
    port: int
    idle_seconds: int


class EndedModel(TypedDict):
    """A model whose server stopped with no unload job.

    Attributes:
        model_id: The catalog id.
        job_id: The load that started it.
        why: Idle for how long, or how its server exited.
    """

    model_id: str
    job_id: str
    why: str


class ModelScan(TypedDict):
    """Every model with state on a node.

    Attributes:
        serving: The ones still served.
        ended: The ones whose end the queue has yet to record.
    """

    serving: tuple[ServedModel, ...]
    ended: tuple[EndedModel, ...]


def _odd(host: str, what: str, output: str) -> AppError[FleetErrorCode]:
    """The fault for a line a model script cannot print.

    Args:
        host: The node.
        what: Which script.
        output: What it printed.

    Returns:
        A ``DISPATCH_FAILED`` naming both.
    """
    return AppError(
        FleetErrorCode.DISPATCH_FAILED,
        f"the model {what} script on {host} printed {output!r}, which it cannot print",
    )


def _transcribing(
    host: str, model: CatalogModel, address: str, port: str, device: str
) -> LoadOutcome:
    """Judge a whisper-server that answered ``/health`` by its GPU line.

    Args:
        host: The node.
        model: The catalog row.
        address: The node's tailnet address.
        port: The port it answers on.
        device: The GPU device its log names, or ``none``.

    Returns:
        ``passed`` with the url when it uses a GPU, else ``failed``.
    """
    if device == "none":
        return LoadOutcome(
            status=ClosingStatus.FAILED,
            endpoint=None,
            detail=(
                f"MODEL_NOT_ON_GPU: {model['model_id']} answered on {host} with no "
                "whisper_backend_init_gpu device in its log, so it runs on the CPU; "
                "whisper-server stopped"
            ),
        )
    return LoadOutcome(
        status=ClosingStatus.PASSED,
        endpoint=f"http://{address}:{port}",
        detail=(
            f"whisper-server serves {model['model_id']} ({model['file']}, sha256 "
            f"{model['sha256'][:12]}) on {host} at {address}:{port} on its GPU {device}, "
            "and answered /health"
        ),
    )


def _passed(host: str, model: CatalogModel, address: str, port: str, offload: str) -> LoadOutcome:
    """Judge a server that answered ``/health`` by where its log says it was placed.

    Args:
        host: The node.
        model: The catalog row.
        address: The node's tailnet address.
        port: The port it answers on.
        offload: llama-server's ``N/M`` or whisper-server's GPU device, or
            ``none``.

    Returns:
        ``passed`` with the url when it is wholly on the GPU, else
        ``failed`` naming where it was.

    Raises:
        AppError: ``DISPATCH_FAILED`` for an address, port or placement
            that is not one.
    """
    layers = re.fullmatch(r"([0-9]+)/([0-9]+)", offload)
    if _ADDRESS.fullmatch(address) is None or re.fullmatch(r"[0-9]{1,5}", port) is None:
        raise _odd(host, "load", f"loaded {address} {port} {offload}")
    if model["server"] is ModelServer.WHISPER:
        if re.fullmatch(r"[A-Za-z0-9_]+", offload) is None:
            raise _odd(host, "load", f"loaded {address} {port} {offload}")
        return _transcribing(host, model, address, port, offload)
    if offload != "none" and layers is None:
        raise _odd(host, "load", f"loaded {address} {port} {offload}")
    counts = offload.split("/")
    if layers is None or counts[0] != counts[1]:
        placed = "no offload line in its log" if layers is None else f"{offload} layers on the GPU"
        return LoadOutcome(
            status=ClosingStatus.FAILED,
            endpoint=None,
            detail=(
                f"MODEL_PARTIAL_OFFLOAD: {model['model_id']} answered on {host} with {placed}, "
                "not every layer on the GPU; llama-server stopped"
            ),
        )
    return LoadOutcome(
        status=ClosingStatus.PASSED,
        endpoint=f"http://{address}:{port}",
        detail=(
            f"llama-server serves {model['model_id']} ({model['file']}, sha256 "
            f"{model['sha256'][:12]}) on {host} at {address}:{port} with {offload} layers on "
            f"the GPU and a {model['context_tokens']}-token context, and answered /health"
        ),
    )


def read_load(output: str, *, host: str, models: NodeModels, model: CatalogModel) -> LoadOutcome:
    """Read what a load script printed into its outcome.

    Args:
        output: The script's output.
        host: The node.
        models: The node's declaration.
        model: The catalog row.

    Returns:
        The outcome; see :func:`fleet.core.model_scripts.load_script` for
        each line.

    Raises:
        AppError: ``DISPATCH_FAILED`` for anything else.
    """
    words = output.strip().split(" ", 1)
    verdict = words[0]
    rest = words[1] if len(words) == 2 else ""
    path = f"{models['cache']}/{model['file']}"
    if verdict == "absent":
        return LoadOutcome(
            status=ClosingStatus.REFUSED,
            endpoint=None,
            detail=f"MODEL_FILE_ABSENT: {model['model_id']}'s file {path} is not on {host}",
        )
    if verdict == "size":
        return LoadOutcome(
            status=ClosingStatus.REFUSED,
            endpoint=None,
            detail=(
                f"MODEL_FILE_MISMATCH: {path} on {host} is {rest} bytes, not the "
                f"{model['size_bytes']} the catalog's {model['model_id']} is; nothing was served"
            ),
        )
    if verdict == "sha256":
        return LoadOutcome(
            status=ClosingStatus.REFUSED,
            endpoint=None,
            detail=(
                f"MODEL_FILE_MISMATCH: {path} on {host} hashes to sha256 {rest}, not the "
                f"catalog's {model['sha256']} for {model['model_id']}; nothing was served"
            ),
        )
    if verdict == "ports":
        return LoadOutcome(
            status=ClosingStatus.REFUSED,
            endpoint=None,
            detail=f"MODEL_PORTS_BUSY: every model port of {host}, {rest}, is taken",
        )
    if verdict == "exited":
        return LoadOutcome(
            status=ClosingStatus.FAILED,
            endpoint=None,
            detail=(
                f"{model['server']} of {model['model_id']} on {host} exited while loading; "
                f"its status and its log's last line: {rest}"
            ),
        )
    if verdict == "unhealthy":
        return LoadOutcome(
            status=ClosingStatus.FAILED,
            endpoint=None,
            detail=(
                f"{model['server']} of {model['model_id']} on {host} did not answer /health in "
                f"{rest} s; stopped"
            ),
        )
    fields = rest.split(" ")
    if verdict == "loaded" and len(fields) == 3:
        return _passed(host, model, fields[0], fields[1], fields[2])
    raise _odd(host, "load", output)


def _script_path(node: NodeConfig, action: str, model_id: str) -> str:
    """Where a model script is written on the node.

    Args:
        node: The node.
        action: ``load``, ``unload``, ``scan``, ``forget`` or ``gpus``.
        model_id: The catalog id, or ``all`` for a scan, a forget or a GPU
            reading.

    Returns:
        A path under its stage root, named so two actions never share one.
    """
    return dialect.script_path(node["stage_root"], f"model-{action}-{model_id}")


def stop_model(node: NodeConfig, model_id: str) -> str | None:
    """Stop a model's server on its node and forget its state.

    Args:
        node: The node.
        model_id: The catalog id.

    Returns:
        The load that started the server it stopped, or None when none of
        that model was active.

    Raises:
        AppError: ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the node,
            and ``DISPATCH_FAILED`` for a line the script cannot print.
    """
    output = remote.run_script(
        node["host"], _script_path(node, "unload", model_id), unload_script(model_id)
    ).strip()
    if output == "absent":
        return None
    stopped = re.fullmatch(r"stopped ([0-9a-f-]{36})", output)
    if stopped is None:
        raise _odd(node["host"], "unload", output)
    return stopped.group(1)


def gpu_free_bytes(node: NodeConfig) -> int:
    """The free GPU memory the tick's probe reads on a node, summed over its GPUs.

    Args:
        node: The node.

    Returns:
        The free bytes of every GPU whose memory a counter holds.

    Raises:
        AppError: ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the node,
            and ``DISPATCH_FAILED`` when the probe's lines cannot be read or
            no GPU's memory can, since a stop that cannot be measured has
            not been shown to free anything.
    """
    output = remote.run_script(
        node["host"], _script_path(node, "gpus", "all"), f"{dialect.PROLOGUE}{GPU_PROBE_PART}"
    )
    gpus, problem = read_gpus(node["host"], read_fields(output))
    if gpus is None:
        raise AppError(FleetErrorCode.DISPATCH_FAILED, problem)
    memories = [gpu["memory"] for gpu in gpus]
    free = [memory["free_bytes"] for memory in memories if memory is not None]
    if not free:
        raise AppError(
            FleetErrorCode.DISPATCH_FAILED,
            f"{node['host']} read {len(gpus)} GPU(s) and no GPU memory counter",
        )
    return sum(free)


def serve_model(
    node: NodeConfig, models: NodeModels, model: CatalogModel, *, job_id: str, binary: str
) -> LoadOutcome:
    """Check a model's file on its node and serve it there.

    Args:
        node: The node.
        models: Its declaration.
        model: The catalog row.
        job_id: The load.
        binary: The node's binary of the model's server.

    Returns:
        The outcome. A server that started and did not pass is stopped
        before this returns, so a failed load holds no GPU memory.

    Raises:
        AppError: ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the node,
            and ``DISPATCH_FAILED`` for a line the script cannot print.
    """
    outcome = remote.attempt_script(
        node["host"],
        _script_path(node, "load", model["model_id"]),
        load_script(models=models, model=model, job_id=job_id, binary=binary),
        timeout_seconds=LOAD_DEADLINE_SECONDS,
    )
    failure = outcome["failure"]
    if failure is not None:
        raise AppError(failure["code"], failure["message"])
    read = read_load(outcome["output"], host=node["host"], models=models, model=model)
    if read["status"] is ClosingStatus.FAILED:
        stop_model(node, model["model_id"])
    return read


#: A scan's line for a model still served, and for one that ended.
_SERVING = re.compile(r"serving ([a-z0-9][a-z0-9.-]*) ([0-9a-f-]{36}) ([0-9]{1,5}) ([0-9]+)")
_ENDED = re.compile(r"ended ([a-z0-9][a-z0-9.-]*) ([0-9a-f-]{36}) (.+)")


def scan_models(node: NodeConfig, models: NodeModels) -> ModelScan:
    """Read every model a node serves, stopping each one idle past its span.

    Args:
        node: The node.
        models: Its declaration.

    Returns:
        What it serves, and what ended with no unload job.

    Raises:
        AppError: ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the node,
            and ``DISPATCH_FAILED`` for a line the scan cannot print.
    """
    output = remote.run_script(node["host"], _script_path(node, "scan", "all"), scan_script(models))
    serving: list[ServedModel] = []
    ended: list[EndedModel] = []
    for line in output.splitlines():
        served = _SERVING.fullmatch(line)
        stopped = _ENDED.fullmatch(line)
        if served is not None:
            serving.append(
                ServedModel(
                    model_id=served.group(1),
                    job_id=served.group(2),
                    port=int(served.group(3)),
                    idle_seconds=int(served.group(4)),
                )
            )
        elif stopped is not None:
            ended.append(
                EndedModel(model_id=stopped.group(1), job_id=stopped.group(2), why=stopped.group(3))
            )
        else:
            raise _odd(node["host"], "scan", line)
    return ModelScan(serving=tuple(serving), ended=tuple(ended))


def forget_models(node: NodeConfig, ended: tuple[EndedModel, ...]) -> tuple[str, ...]:
    """Remove the node's state of models whose end the queue has recorded.

    Args:
        node: The node.
        ended: The models.

    Returns:
        The ids whose state was removed.

    Raises:
        AppError: ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the node,
            and ``DISPATCH_FAILED`` for a line the script cannot print.
    """
    pairs = tuple((model["model_id"], model["job_id"]) for model in ended)
    output = remote.run_script(
        node["host"], _script_path(node, "forget", "all"), forget_script(pairs)
    )
    forgot: list[str] = []
    for line in output.splitlines():
        removed = re.fullmatch(r"forgot ([a-z0-9][a-z0-9.-]*)", line)
        if removed is None:
            raise _odd(node["host"], "forget", line)
        forgot.append(removed.group(1))
    return tuple(forgot)


__all__ = [
    "LOAD_DEADLINE_SECONDS",
    "EndedModel",
    "LoadOutcome",
    "ModelScan",
    "ServedModel",
    "forget_models",
    "read_load",
    "scan_models",
    "serve_model",
    "stop_model",
]
