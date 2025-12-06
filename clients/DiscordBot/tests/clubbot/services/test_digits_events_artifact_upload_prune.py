from __future__ import annotations

import logging

import pytest
from platform_core.digits_metrics_events import (
    is_digits_artifact,
    is_digits_prune,
    is_digits_upload,
    try_decode_digits_event,
)
from platform_core.json_utils import dump_json_str


def test_decode_artifact_valid_complete_payload() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.artifact.v1",
            "job_id": "r1",
            "user_id": 42,
            "model_id": "mnist",
            "path": "/artifacts/digits/models/mnist_resnet18_v1",
        }
    )
    evt = try_decode_digits_event(payload)
    if evt is None:
        raise AssertionError("expected decoded event")
    assert is_digits_artifact(evt)
    assert evt["job_id"] == "r1"
    assert evt["user_id"] == 42
    assert evt["model_id"] == "mnist"
    assert evt["path"] == "/artifacts/digits/models/mnist_resnet18_v1"


def test_decode_artifact_with_null_run_id() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.artifact.v1",
            "job_id": "r2",
            "user_id": 1,
            "model_id": "m",
            "path": "/path/to/artifact",
        }
    )
    evt = try_decode_digits_event(payload)
    if evt is None:
        raise AssertionError("expected decoded event")
    assert evt["type"] == "digits.metrics.artifact.v1"


def test_decode_artifact_missing_path_field_raises() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.artifact.v1",
            "job_id": "r",
            "user_id": 1,
            "model_id": "m",
            # missing path
        }
    )
    with pytest.raises(ValueError, match="artifact event missing required fields"):
        try_decode_digits_event(payload)


def test_decode_artifact_path_wrong_type_raises() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.artifact.v1",
            "job_id": "r",
            "user_id": 1,
            "model_id": "m",
            "path": 123,  # int instead of string
        }
    )
    with pytest.raises(ValueError, match="artifact event missing required fields"):
        try_decode_digits_event(payload)


def test_decode_upload_valid_complete_payload() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.upload.v1",
            "job_id": "r1",
            "user_id": 42,
            "model_id": "mnist",
            "status": 200,
            "model_bytes": 45678901,
            "manifest_bytes": 1234,
            "file_id": "fid",
            "file_sha256": "sha",
        }
    )
    evt = try_decode_digits_event(payload)
    if evt is None:
        raise AssertionError("expected decoded event")
    assert is_digits_upload(evt)
    assert evt["job_id"] == "r1"
    assert evt["user_id"] == 42
    assert evt["model_id"] == "mnist"
    assert evt["status"] == 200
    assert evt["model_bytes"] == 45678901
    assert evt["manifest_bytes"] == 1234


def test_decode_upload_with_null_run_id() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.upload.v1",
            "job_id": "r2",
            "user_id": 1,
            "model_id": "m",
            "status": 500,
            "model_bytes": 100,
            "manifest_bytes": 50,
            "file_id": "fid",
            "file_sha256": "sha",
        }
    )
    evt = try_decode_digits_event(payload)
    if evt is None:
        raise AssertionError("expected decoded event")
    assert evt["type"] == "digits.metrics.upload.v1"


def test_decode_upload_missing_status_field_raises() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.upload.v1",
            "job_id": "r",
            "user_id": 1,
            "model_id": "m",
            # missing status
            "model_bytes": 100,
            "manifest_bytes": 50,
        }
    )
    with pytest.raises(ValueError, match="upload event missing required fields"):
        try_decode_digits_event(payload)


def test_decode_upload_status_wrong_type_raises() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.upload.v1",
            "job_id": "r",
            "user_id": 1,
            "model_id": "m",
            "status": "200",  # string instead of int
            "model_bytes": 100,
            "manifest_bytes": 50,
        }
    )
    with pytest.raises(ValueError, match="upload event missing required fields"):
        try_decode_digits_event(payload)


def test_decode_upload_model_bytes_wrong_type_raises() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.upload.v1",
            "job_id": "r",
            "user_id": 1,
            "model_id": "m",
            "status": 200,
            "model_bytes": "100",  # string instead of int
            "manifest_bytes": 50,
        }
    )
    with pytest.raises(ValueError, match="upload event missing required fields"):
        try_decode_digits_event(payload)


def test_decode_prune_valid_complete_payload() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.prune.v1",
            "job_id": "r1",
            "user_id": 42,
            "model_id": "mnist",
            "deleted_count": 3,
        }
    )
    evt = try_decode_digits_event(payload)
    if evt is None:
        raise AssertionError("expected decoded event")
    assert is_digits_prune(evt)
    assert evt["job_id"] == "r1"
    assert evt["user_id"] == 42
    assert evt["model_id"] == "mnist"
    assert evt["deleted_count"] == 3


def test_decode_prune_with_null_run_id() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.prune.v1",
            "job_id": "r2",
            "user_id": 1,
            "model_id": "m",
            "deleted_count": 0,
        }
    )
    evt = try_decode_digits_event(payload)
    if evt is None:
        raise AssertionError("expected decoded event")
    assert evt["type"] == "digits.metrics.prune.v1"


def test_decode_prune_missing_deleted_count_field_raises() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.prune.v1",
            "job_id": "r",
            "user_id": 1,
            "model_id": "m",
            # missing deleted_count
        }
    )
    with pytest.raises(ValueError, match="prune event missing required fields"):
        try_decode_digits_event(payload)


def test_decode_prune_deleted_count_wrong_type_raises() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.prune.v1",
            "job_id": "r",
            "user_id": 1,
            "model_id": "m",
            "deleted_count": "3",  # string instead of int
        }
    )
    with pytest.raises(ValueError, match="prune event missing required fields"):
        try_decode_digits_event(payload)


logger = logging.getLogger(__name__)
