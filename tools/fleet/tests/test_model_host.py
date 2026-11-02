"""A node's model declaration and a model job's catalog row, decoded strictly.

MCPs board task b9235198.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, dump_json_str

from fleet.contracts.model_host import (
    CatalogModel,
    ModelBackend,
    NodeModels,
    decode_catalog_model,
    decode_node_models,
    encode_node_models,
)

#: colossus's declaration, as its machine file carries it.
COLOSSUS_MODELS: JSONObject = {
    "server": "/home/corvis/build/llama.cpp-b11457/build/bin/llama-server",
    "backend": "vulkan",
    "cache": "/home/corvis/models",
    "first_port": 8091,
    "last_port": 8094,
    "idle_seconds": 1800,
}

#: The catalog's Ornith 9B, as fleet-mcp renders it on a claimed job.
ORNITH_9B_WIRE: JSONObject = {
    "id": "ornith-1.5-9b",
    "file": "Ornith-1.5-9B-Q4_K_M.gguf",
    "sha256": "70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6",
    "sizeBytes": 5780090816,
    "gpuBytes": 6499381248,
    "contextTokens": 32768,
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
        assert decoded == NodeModels(
            server="/home/corvis/build/llama.cpp-b11457/build/bin/llama-server",
            backend=ModelBackend.VULKAN,
            cache="/home/corvis/models",
            first_port=8091,
            last_port=8094,
            idle_seconds=1800,
        )
        assert encode_node_models(decoded) == COLOSSUS_MODELS

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
        decoded = decode_node_models({**COLOSSUS_MODELS, "last_port": 8106})
        assert decoded is not None
        assert decoded["last_port"] == 8106

    def test_an_idle_span_that_is_not_positive_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="idle_seconds must be positive, got 0"):
            decode_node_models({**COLOSSUS_MODELS, "idle_seconds": 0})

    def test_a_backend_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="sycl"):
            decode_node_models({**COLOSSUS_MODELS, "backend": "sycl"})

    @pytest.mark.parametrize("key", ["server", "cache"])
    def test_a_path_that_is_not_plain_and_absolute_is_refused(self, key: str) -> None:
        for path in ("models", "/home/corvis/my models", "/home/'x'"):
            with pytest.raises(JSONTypeError, match=f"models.{key} must be an absolute path"):
                decode_node_models({**COLOSSUS_MODELS, key: path})


class TestCatalogModel:
    """The ``model`` a claimed model job carries."""

    def test_ornith_9b_decodes_field_for_field(self) -> None:
        assert _catalog(ORNITH_9B_WIRE) == CatalogModel(
            model_id="ornith-1.5-9b",
            file="Ornith-1.5-9B-Q4_K_M.gguf",
            sha256="70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6",
            size_bytes=5780090816,
            gpu_bytes=6499381248,
            context_tokens=32768,
            backends=(ModelBackend.VULKAN,),
        )

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("id", "Ornith 9B", "model id 'Ornith 9B' is not a catalog id"),
            ("file", "../Ornith.gguf", "model file '../Ornith.gguf' is not a plain GGUF file name"),
            ("file", "Ornith-1.5-9B.bin", "model file 'Ornith-1.5-9B.bin' is not a plain GGUF"),
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
