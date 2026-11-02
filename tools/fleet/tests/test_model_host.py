"""A node's model declaration and a model job's catalog row, decoded strictly.

MCPs board task b9235198.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
)

from fleet.contracts.model_host import (
    CatalogModel,
    ModelBackend,
    ModelServer,
    NodeModels,
    decode_catalog_model,
    decode_node_models,
    encode_node_models,
)
from tests._contract_fixtures import COLOSSUS_MODELS as COLOSSUS_DECLARED

LLAMA_BINARY = "/home/corvis/build/llama.cpp-b11457/build/bin/llama-server"
WHISPER_BINARY = "/home/corvis/build/whisper.cpp-v1.9.5/build/bin/whisper-server"

#: colossus's declaration, as its machine file carries it.
COLOSSUS_MODELS: JSONObject = {
    "servers": {"llama-server": LLAMA_BINARY, "whisper-server": WHISPER_BINARY},
    "backend": "vulkan",
    "cache": "/home/corvis/models",
    "first_port": 8091,
    "last_port": 8094,
    "idle_seconds": 1800,
}

#: The catalog's Ornith 9B, as fleet-mcp renders it on a claimed job.
ORNITH_9B_WIRE: JSONObject = {
    "id": "ornith-1.5-9b",
    "server": "llama-server",
    "file": "Ornith-1.5-9B-Q4_K_M.gguf",
    "sha256": "70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6",
    "sizeBytes": 5780090816,
    "gpuBytes": 6499381248,
    "contextTokens": 32768,
    "backends": ["vulkan"],
}

#: The catalog's whisper model, as fleet-mcp renders it (MCPs board task 228aa097).
WHISPER_WIRE: JSONObject = {
    "id": "whisper-large-v3-turbo",
    "server": "whisper-server",
    "file": "ggml-large-v3-turbo.bin",
    "sha256": "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    "sizeBytes": 1624555275,
    "gpuBytes": 1932406784,
    "contextTokens": None,
    "backends": ["vulkan"],
}


def _catalog(value: JSONValue) -> CatalogModel:
    """Decode a ``model`` as a claim answer carrying it would.

    Args:
        value: The field's value.

    Returns:
        The row.
    """
    return decode_catalog_model(value, answer=dump_json_str({"model": value}))


class TestNodeModels:
    """A machine file's ``dispatch.models``."""

    def test_colossus_s_declaration_round_trips(self) -> None:
        decoded = decode_node_models(COLOSSUS_MODELS)
        assert decoded == COLOSSUS_DECLARED
        assert encode_node_models(decoded) == COLOSSUS_MODELS

    def test_colossus_s_machine_file_declares_both_servers(self) -> None:
        machine = pathlib.Path(__file__).parents[1] / "registry" / "machines" / "colossus.json"
        declared = load_json_str(machine.read_text(encoding="utf-8"))
        assert isinstance(declared, dict)
        dispatch = declared["dispatch"]
        assert isinstance(dispatch, dict)
        assert dispatch["models"] == COLOSSUS_MODELS

    @pytest.mark.parametrize(
        ("servers", "message"),
        [
            ({}, r"models.servers must be a non-empty JSON object, got \{\}"),
            (LLAMA_BINARY, "models.servers must be a non-empty JSON object, got '/home"),
            (
                {"vllm": LLAMA_BINARY},
                "models.servers names 'vllm', which is not one of llama-server, whisper-server",
            ),
            ({"whisper-server": 7}, "models.servers.whisper-server must be a string, got 7"),
            ({"whisper-server": "whisper-server"}, "models.servers.whisper-server must be an"),
        ],
    )
    def test_servers_that_are_not_built_binaries_are_refused(
        self, servers: JSONValue, message: str
    ) -> None:
        with pytest.raises(JSONTypeError, match=message):
            decode_node_models({**COLOSSUS_MODELS, "servers": servers})

    def test_null_is_a_node_that_serves_none(self) -> None:
        assert decode_node_models(None) is None
        assert encode_node_models(None) is None

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="models must be null or a JSON object, got str"):
            decode_node_models("colossus")

    @pytest.mark.parametrize(
        ("first", "last"),
        [(80, 81), (8094, 8091), (8091, 8107), (65535, 65536)],
    )
    def test_ports_outside_an_unprivileged_range_of_sixteen_are_refused(
        self, first: int, last: int
    ) -> None:
        with pytest.raises(JSONTypeError, match=f"got {first} to {last}"):
            decode_node_models({**COLOSSUS_MODELS, "first_port": first, "last_port": last})

    def test_sixteen_ports_are_admitted(self) -> None:
        assert decode_node_models({**COLOSSUS_MODELS, "last_port": 8106}) == NodeModels(
            servers={ModelServer.LLAMA: LLAMA_BINARY, ModelServer.WHISPER: WHISPER_BINARY},
            backend=ModelBackend.VULKAN,
            cache="/home/corvis/models",
            first_port=8091,
            last_port=8106,
            idle_seconds=1800,
        )

    def test_an_idle_span_that_is_not_positive_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="idle_seconds must be positive, got 0"):
            decode_node_models({**COLOSSUS_MODELS, "idle_seconds": 0})

    def test_a_backend_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="sycl"):
            decode_node_models({**COLOSSUS_MODELS, "backend": "sycl"})

    def test_a_path_that_is_not_plain_and_absolute_is_refused(self) -> None:
        for path in ("models", "/home/corvis/my models", "/home/'x'"):
            with pytest.raises(JSONTypeError, match=r"models\.cache must be an absolute path"):
                decode_node_models({**COLOSSUS_MODELS, "cache": path})
            with pytest.raises(
                JSONTypeError, match=r"models\.servers\.llama-server must be an absolute path"
            ):
                decode_node_models({**COLOSSUS_MODELS, "servers": {"llama-server": path}})


class TestCatalogModel:
    """The ``model`` a claimed model job carries."""

    def test_ornith_9b_decodes_field_for_field(self) -> None:
        assert _catalog(ORNITH_9B_WIRE) == CatalogModel(
            model_id="ornith-1.5-9b",
            server=ModelServer.LLAMA,
            file="Ornith-1.5-9B-Q4_K_M.gguf",
            sha256="70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6",
            size_bytes=5780090816,
            gpu_bytes=6499381248,
            context_tokens=32768,
            backends=(ModelBackend.VULKAN,),
        )

    def test_the_whisper_model_decodes_with_no_context(self) -> None:
        assert _catalog(WHISPER_WIRE) == CatalogModel(
            model_id="whisper-large-v3-turbo",
            server=ModelServer.WHISPER,
            file="ggml-large-v3-turbo.bin",
            sha256="1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
            size_bytes=1624555275,
            gpu_bytes=1932406784,
            context_tokens=None,
            backends=(ModelBackend.VULKAN,),
        )

    @pytest.mark.parametrize(
        ("wire", "message"),
        [
            (
                {**WHISPER_WIRE, "contextTokens": 448},
                "model field 'contextTokens' is 448, not null, for a whisper-server model",
            ),
            (
                {key: value for key, value in WHISPER_WIRE.items() if key != "contextTokens"},
                "model field 'contextTokens' is None, not null, for a whisper-server model",
            ),
            (
                {**WHISPER_WIRE, "file": "ggml-large-v3-turbo.gguf"},
                "model file 'ggml-large-v3-turbo.gguf' is not a plain .bin file name, which "
                "whisper-server reads",
            ),
            ({**ORNITH_9B_WIRE, "contextTokens": None}, "model field 'contextTokens' is None"),
        ],
    )
    def test_a_context_or_file_that_is_not_its_server_s_is_malformed(
        self, wire: JSONObject, message: str
    ) -> None:
        with pytest.raises(AppError, match=message) as refused:
            _catalog(wire)
        assert refused.value.code is FleetErrorCode.QUEUE_ANSWER_MALFORMED

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("id", "Ornith 9B", "model id 'Ornith 9B' is not a catalog id"),
            ("server", "vllm", "model server 'vllm' is not one of llama-server, whisper-server"),
            ("server", None, "model server None is not one of llama-server, whisper-server"),
            ("file", "../Ornith.gguf", "model file '../Ornith.gguf' is not a plain .gguf file"),
            ("file", "Ornith-1.5-9B.bin", "model file 'Ornith-1.5-9B.bin' is not a plain .gguf"),
            ("sha256", "70C1", "model sha256 '70C1' is not 64 lowercase hex"),
            ("backends", [], r"model backends is \[\], not a non-empty array"),
            ("backends", "vulkan", "model backends is 'vulkan', not a non-empty array"),
            ("backends", ["sycl"], "model backend 'sycl' is not one of cuda, vulkan"),
            ("backends", [7], "model backend 7 is not one of cuda, vulkan"),
            ("sizeBytes", 0, "model field 'sizeBytes' is 0, not a positive integer"),
            ("gpuBytes", True, "model field 'gpuBytes' is True, not a positive integer"),
            ("contextTokens", "32768", "model field 'contextTokens' is '32768', not a positive"),
        ],
    )
    def test_a_field_the_catalog_cannot_hold_is_malformed(
        self, field: str, value: JSONValue, message: str
    ) -> None:
        with pytest.raises(AppError, match=message) as refused:
            _catalog({**ORNITH_9B_WIRE, field: value})
        assert refused.value.code is FleetErrorCode.QUEUE_ANSWER_MALFORMED

    def test_a_model_that_is_not_an_object_is_malformed(self) -> None:
        with pytest.raises(AppError, match="a model job's model is NoneType, not an object"):
            _catalog(None)
