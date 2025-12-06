from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str

from music_wrapped_api.app import create_app


def test_schema_endpoint_returns_json_schema() -> None:
    client = TestClient(create_app())
    r = client.get("/v1/wrapped/schema")
    assert r.status_code == 200
    schema = load_json_str(r.text)
    if not isinstance(schema, dict):
        raise AssertionError("schema must be an object")
    assert schema.get("type") == "object"
    props: JSONValue | None = schema.get("properties")
    if not isinstance(props, dict):
        raise AssertionError("properties must be an object")
    assert "top_songs" in props and "top_artists" in props and "top_by_month" in props
