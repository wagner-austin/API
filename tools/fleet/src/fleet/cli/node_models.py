"""The models half of a node's serve: load and unload catalog models, and stop idle ones.

MCPs board task b9235198. A node whose machine file declares ``models``
(:mod:`fleet.contracts.model_host`) is the one the queue places a catalog
model's load on (MCPs mig 896, board task 62b1bb0a), and its runner, the
serve of :mod:`fleet.cli.node_agent`, drains the queue's ``models`` lane for
it beside the node lane's checks. Two passes, each run where the serve
already runs one:

    THE MODELS PASS, before every fill pass. It claims each model job naming
    this node, reports it started, and runs it to its close in the same
    pass: a load is checked against its catalog row and served
    (:func:`fleet.core.model_serve.serve_model`), then passed with the url
    it answers on (``dispatch_report action='loaded'``); an unload stops the
    model's server and passes, its close naming the GPU free bytes the
    tick's own probe read before the stop and after it
    (:func:`fleet.core.model_serve.gpu_free_bytes`). That record is the
    unload's: the tick the fill pass after it records
    (:mod:`fleet.cli.node_ready`) keeps only the runner's latest row, which
    the next stop overwrites (audit 7b709022).

    THE IDLE PASS, after every collect pass, so at every fire boundary. It
    reads every model the node serves (:func:`fleet.core.model_serve.scan_models`),
    which stops one that nobody has asked anything for its idle span, and
    tells the queue each one that stopped with no unload job, idle or exited
    (``dispatch_report action='ended'``), then forgets it on the node.

A load runs inside its pass, so the serve's loop waits on it for up to
:data:`fleet.core.model_serve.LOAD_DEADLINE_SECONDS`, the hash and the
model's start; checks already launched are watched on the watch's own
thread meanwhile (:mod:`fleet.cli.node_serve`).

A NODE THAT FAULTS A MODEL JOB REFUSES IT, as every runner here reports a
local refusal (:mod:`fleet.cli.agent`): an unreachable node or a script
that failed closes the job ``refused`` with the fault's code and message,
and the serve goes on. Only a queue call's own failure propagates.
"""

from __future__ import annotations

from typing import Final

from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import job_close
from fleet.contracts.dispatch import (
    ClosingStatus,
    DispatchCommand,
    DispatchJob,
    DispatchLane,
    encode_job_line,
)
from fleet.contracts.model_host import CatalogModel, NodeModels
from fleet.contracts.node import NodeConfig
from fleet.core import queue, queue_models
from fleet.core.model_scripts import model_unit
from fleet.core.model_serve import (
    LOAD_DEADLINE_SECONDS,
    EndedModel,
    forget_models,
    gpu_free_bytes,
    scan_models,
    serve_model,
    stop_model,
)
from fleet.core.queue_transport import unanswered

_log = get_logger(__name__)

#: The lease a model job is claimed and started under: its whole load, with
#: room for the reports either side.
MODEL_LEASE_SECONDS: Final[int] = LOAD_DEADLINE_SECONDS + 120


def _require_model(job: DispatchJob) -> CatalogModel:
    """The catalog row a model job carries.

    Args:
        job: A job the models lane handed out.

    Returns:
        Its model.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` for a job without one, which
            the decoder gives every model command, so the lane handed out
            another command.
    """
    model = job["model"]
    if model is None:
        raise AppError(
            FleetErrorCode.QUEUE_ANSWER_MALFORMED,
            f"the models lane handed out {encode_job_line(job)}, which names no catalog model",
        )
    return model


def _unload(
    credentials: McpCredentials,
    identity: JSONObject,
    job: DispatchJob,
    *,
    alias: str,
    node: NodeConfig,
    model: CatalogModel,
) -> None:
    """Stop a model's server and pass its unload.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        job: The unload, reported started.
        alias: This node's name.
        node: Its declaration.
        model: The catalog row.

    Raises:
        AppError: Only from the queue calls.
    """
    try:
        before = gpu_free_bytes(node)
        stopped = stop_model(node, model["model_id"])
        after = gpu_free_bytes(node)
    except AppError as fault:
        job_close.close(
            credentials,
            job,
            identity,
            status=ClosingStatus.REFUSED,
            exit_code=None,
            detail=f"{fault.code}: {fault.message}",
        )
        return
    said = (
        f"{model['model_id']} was not being served on {alias}, so there was nothing to stop"
        if stopped is None
        else (
            f"stopped {model['server']} of {model['model_id']} on {alias}, "
            f"which load {stopped} started"
        )
    )
    detail = (
        f"{said}; the tick's GPU probe read {before} free bytes before the stop "
        f"and {after} after it"
    )
    job_close.close(
        credentials, job, identity, status=ClosingStatus.PASSED, exit_code=0, detail=detail
    )


def _load(
    credentials: McpCredentials,
    identity: JSONObject,
    job: DispatchJob,
    *,
    alias: str,
    node: NodeConfig,
    models: NodeModels,
    model: CatalogModel,
) -> None:
    """Serve a model and pass its load with the url it answers on, or say why not.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        job: The load, reported started.
        alias: This node's name.
        node: Its declaration.
        models: How it serves models.
        model: The catalog row.

    Raises:
        AppError: Only from the queue calls.
    """
    if models["backend"] not in model["backends"]:
        job_close.close(
            credentials,
            job,
            identity,
            status=ClosingStatus.REFUSED,
            exit_code=None,
            detail=(
                f"MODEL_BACKEND: {alias} serves models on {models['backend']}, and "
                f"{model['model_id']} is measured on {', '.join(model['backends'])} only"
            ),
        )
        return
    binary = models["servers"].get(model["server"])
    if binary is None:
        job_close.close(
            credentials,
            job,
            identity,
            status=ClosingStatus.REFUSED,
            exit_code=None,
            detail=(
                f"MODEL_SERVER_ABSENT: {model['model_id']} runs in {model['server']}, which "
                f"{alias} has not built; its machine file declares "
                f"{', '.join(models['servers'])}"
            ),
        )
        return
    try:
        outcome = serve_model(node, models, model, job_id=job["job_id"], binary=binary)
    except AppError as fault:
        job_close.close(
            credentials,
            job,
            identity,
            status=ClosingStatus.REFUSED,
            exit_code=None,
            detail=f"{fault.code}: {fault.message}",
        )
        return
    endpoint = outcome["endpoint"]
    if endpoint is not None:
        queue_models.report_loaded(
            credentials,
            job_id=job["job_id"],
            endpoint=endpoint,
            detail=outcome["detail"],
            identity=identity,
        )
        _log.info("%s: %s", encode_job_line(job), outcome["detail"])
        return
    job_close.close(
        credentials,
        job,
        identity,
        status=outcome["status"],
        exit_code=None if outcome["status"] is ClosingStatus.REFUSED else 1,
        detail=outcome["detail"],
    )


def models_pass(
    credentials: McpCredentials, identity: JSONObject, *, alias: str, node: NodeConfig
) -> tuple[str, ...]:
    """Claim and run to its close every model job naming this node.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        alias: This node's name.
        node: Its declaration.

    Returns:
        The jobs claimed, in order; none for a node that serves no model,
        which never asks the lane.

    Raises:
        AppError: Only from the queue calls.
    """
    models = node["models"]
    if models is None:
        return ()
    handled: list[str] = []
    while (
        job := queue.claim_next(
            credentials,
            lane=DispatchLane.MODELS,
            tags=(),
            node=alias,
            projects=None,
            lease_seconds=MODEL_LEASE_SECONDS,
            identity=identity,
        )
    ) is not None:
        model = _require_model(job)
        _log.info("claimed %s", encode_job_line(job))
        queue.report_start(
            credentials,
            job_id=job["job_id"],
            node=alias,
            run_id=model_unit(model["model_id"]),
            lease_seconds=MODEL_LEASE_SECONDS,
            identity=identity,
        )
        if job["command"] is DispatchCommand.MODEL_UNLOAD:
            _unload(credentials, identity, job, alias=alias, node=node, model=model)
        else:
            _load(credentials, identity, job, alias=alias, node=node, models=models, model=model)
        handled.append(job["job_id"])
    return tuple(handled)


def _record_end(
    credentials: McpCredentials, identity: JSONObject, ended: EndedModel, *, alias: str
) -> None:
    """Tell the queue a model's server stopped with no unload job.

    A refusal the queue ANSWERED is final, a load that ended already or one
    that never passed, so it is logged and the model is forgotten all the
    same; a queue that did not answer propagates, and the end is said again
    at the next scan, since the node still holds it.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        ended: The model.
        alias: This node's name.

    Raises:
        AppError: ``QUEUE_UNANSWERED``.
    """
    try:
        queue_models.report_ended(
            credentials, job_id=ended["job_id"], detail=ended["why"], identity=identity
        )
    except AppError as refusal:
        if unanswered(refusal):
            raise
        _log.info(
            "%s: the queue refused the end of %s's load %s, so it is forgotten: %s",
            alias,
            ended["model_id"],
            ended["job_id"],
            refusal.message,
        )
        return
    _log.info("%s: %s's load %s ended: %s", alias, ended["model_id"], ended["job_id"], ended["why"])


def idle_pass(
    credentials: McpCredentials, identity: JSONObject, *, alias: str, node: NodeConfig
) -> tuple[str, ...]:
    """Stop every model idle past its span, and record each that stopped.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        alias: This node's name.
        node: Its declaration.

    Returns:
        The models whose end was recorded and whose state was forgotten;
        none for a node that serves no model, or one that did not answer,
        which is read again at the next pass.

    Raises:
        AppError: From the queue calls, and ``DISPATCH_FAILED`` when the
            node answered and a model script failed there.
    """
    models = node["models"]
    if models is None:
        return ()
    try:
        scan = scan_models(node, models)
    except AppError as fault:
        if fault.code is not FleetErrorCode.NODE_UNREACHABLE:
            raise
        _log.info("%s: its served models were not read: %s", alias, fault.message)
        return ()
    for served in scan["serving"]:
        _log.info(
            "%s: serves %s for load %s on port %d, idle %d s",
            alias,
            served["model_id"],
            served["job_id"],
            served["port"],
            served["idle_seconds"],
        )
    if not scan["ended"]:
        return ()
    for ended in scan["ended"]:
        _record_end(credentials, identity, ended, alias=alias)
    return forget_models(node, scan["ended"])


__all__ = ["MODEL_LEASE_SECONDS", "idle_pass", "models_pass"]
