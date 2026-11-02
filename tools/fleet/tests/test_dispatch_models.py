"""Model jobs remain readable by existing fleet consumers (MCPs mig 896, board task 62b1bb0a),
and each carries its catalog row to the runner that serves it (board task b9235198)."""

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.contracts.dispatch import (
    MODEL_COMMANDS,
    DispatchCommand,
    DispatchLane,
    decode_claim,
    decode_listing,
    encode_job_line,
)
from fleet.contracts.model_host import CatalogModel, ModelBackend, ModelServer
from tests._queue_fakes import queue_job

#: The catalog row the queue hands a model job's runner, as fleet-mcp renders it.
GLIMMER: JSONObject = {
    "id": "glimmer-30b",
    "server": "llama-server",
    "file": "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
    "sha256": "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e",
    "sizeBytes": 16756683904,
    "gpuBytes": 17016778752,
    "contextTokens": 32768,
    "backends": ["vulkan"],
}


def test_a_listing_with_model_jobs_decodes_every_row_and_names_each_model() -> None:
    """A load and an unload must not poison a listing holding a check beside them."""
    rows = decode_listing(
        dump_json_str(
            {
                "jobs": [
                    queue_job(model=None),
                    queue_job(
                        command="model-load",
                        project="models/glimmer-30b",
                        status="passed",
                        requestedNode="colossus",
                        node="colossus",
                        sha=None,
                        model=GLIMMER,
                        modelEndpoint="http://colossus:8091",
                    ),
                    queue_job(
                        command="model-unload",
                        project="models/glimmer-30b",
                        requestedNode="colossus",
                        sha=None,
                        model=GLIMMER,
                        modelEndpoint=None,
                    ),
                ]
            }
        )
    )
    assert tuple(row["command"] for row in rows) == (
        DispatchCommand.CHECK,
        DispatchCommand.MODEL_LOAD,
        DispatchCommand.MODEL_UNLOAD,
    )
    assert rows[0]["model"] is None
    glimmer = CatalogModel(
        model_id="glimmer-30b",
        server=ModelServer.LLAMA,
        file="Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        sha256="4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e",
        size_bytes=16756683904,
        gpu_bytes=17016778752,
        context_tokens=32768,
        backends=(ModelBackend.VULKAN,),
    )
    assert (rows[1]["model"], rows[2]["model"]) == (glimmer, glimmer)
    assert encode_job_line(rows[1]) == (
        "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa passed model-load models/glimmer-30b @colossus"
    )
    assert encode_job_line(rows[2]) == (
        "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa queued model-unload models/glimmer-30b @colossus"
    )
    assert DispatchLane.MODELS.value == "models"
    assert frozenset({DispatchCommand.MODEL_LOAD, DispatchCommand.MODEL_UNLOAD}) == MODEL_COMMANDS


def test_a_model_job_whose_row_names_no_model_is_malformed() -> None:
    """The catalog row is what the runner serves from, so a load without one is refused."""
    with pytest.raises(AppError, match="a model job's model is NoneType") as refused:
        decode_claim(
            dump_json_str(
                {
                    "claimed": queue_job(
                        command="model-load",
                        project="models/glimmer-30b",
                        requestedNode="colossus",
                        sha=None,
                        model=None,
                    )
                }
            )
        )
    assert refused.value.code is FleetErrorCode.QUEUE_ANSWER_MALFORMED
