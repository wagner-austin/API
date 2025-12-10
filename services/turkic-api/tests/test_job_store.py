from __future__ import annotations

from datetime import datetime

import pytest
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.json_utils import JSONTypeError
from platform_core.turkic_jobs import TurkicJobStatus, turkic_job_key
from platform_workers.testing import FakeRedis

from turkic_api.api.job_store import TurkicJobStore


def test_job_store_roundtrip() -> None:
    r = FakeRedis()
    store = TurkicJobStore(r)
    now = datetime.utcnow()
    status: TurkicJobStatus = {
        "job_id": "abc",
        "user_id": 42,
        "status": "queued",
        "progress": 0,
        "message": None,
        "result_url": None,
        "file_id": None,
        "upload_status": None,
        "created_at": now,
        "updated_at": now,
        "error": None,
    }
    store.save(status)
    loaded = store.load("abc")
    if loaded is None:
        pytest.fail("expected loaded job")
    assert loaded["job_id"] == "abc"
    assert loaded["user_id"] == 42
    assert loaded["status"] == "queued"
    assert turkic_job_key("abc") in r._hashes
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_invalid_status_raises() -> None:
    r = FakeRedis()
    key = turkic_job_key("bad")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "unknown",
            "progress": "0",
            "created_at": now,
            "updated_at": now,
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="invalid status"):
        store.load("bad")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_handles_failed_and_non_numeric_progress() -> None:
    r = FakeRedis()
    key = turkic_job_key("f1")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "failed",
            "progress": "not-a-number",
            "created_at": now,
            "updated_at": now,
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="invalid progress"):
        store.load("f1")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_upload_metadata_roundtrip() -> None:
    r = FakeRedis()
    store = TurkicJobStore(r)
    meta: FileUploadResponse = {
        "file_id": "fid",
        "size": 123,
        "sha256": "abc123",
        "content_type": "text/plain",
        "created_at": "2024-01-01T00:00:00Z",
    }
    store.save_upload_metadata("job1", meta)
    loaded = store.load_upload_metadata("job1")
    assert loaded == meta
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_upload_metadata_allows_null_created_at() -> None:
    r = FakeRedis()
    store = TurkicJobStore(r)
    meta: FileUploadResponse = {
        "file_id": "fid2",
        "size": 5,
        "sha256": "def",
        "content_type": "text/plain",
        "created_at": None,
    }
    store.save_upload_metadata("job2", meta)
    loaded = store.load_upload_metadata("job2")
    assert loaded["created_at"] is None
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_missing_progress_raises() -> None:
    r = FakeRedis()
    key = turkic_job_key("mp")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "queued",
            "created_at": now,
            "updated_at": now,
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="missing progress"):
        store.load("mp")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_missing_created_at_raises() -> None:
    r = FakeRedis()
    key = turkic_job_key("mc")
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "queued",
            "progress": "0",
            "updated_at": "2024-01-01T00:00:00",
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="missing created_at"):
        store.load("mc")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_invalid_upload_status_raises() -> None:
    r = FakeRedis()
    key = turkic_job_key("us")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "processing",
            "progress": "10",
            "created_at": now,
            "updated_at": now,
            "upload_status": "pending",
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="invalid upload_status"):
        store.load("us")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_load_upload_metadata_missing() -> None:
    r = FakeRedis()
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="upload metadata missing"):
        store.load_upload_metadata("missing")
    r.assert_only_called({"hgetall"})


def test_job_store_load_upload_metadata_invalid_size() -> None:
    r = FakeRedis()
    key = f"{turkic_job_key('bad')}:file"
    r.hset(
        key,
        {
            "file_id": "fid",
            "size": "NaN",
            "sha256": "abc",
            "content_type": "text/plain",
            "created_at": "2024-01-01T00:00:00Z",
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="invalid size"):
        store.load_upload_metadata("bad")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_upload_metadata_invalid_file_id() -> None:
    r = FakeRedis()
    key = f"{turkic_job_key('badfid')}:file"
    r.hset(
        key,
        {
            "file_id": "",
            "size": "1",
            "sha256": "abc",
            "content_type": "text/plain",
            "created_at": "2024-01-01T00:00:00Z",
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="invalid file_id"):
        store.load_upload_metadata("badfid")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_missing_status_raises() -> None:
    r = FakeRedis()
    key = turkic_job_key("nostat")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "progress": "0",
            "created_at": now,
            "updated_at": now,
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="missing status"):
        store.load("nostat")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_progress_wrong_type_raises() -> None:
    r = FakeRedis()
    key = turkic_job_key("ptype")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "queued",
            "progress": "abc",
            "created_at": now,
            "updated_at": now,
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="invalid progress"):
        store.load("ptype")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_optional_field_wrong_type_raises() -> None:
    r = FakeRedis()
    key = turkic_job_key("optbad")
    now = datetime.utcnow().isoformat()
    r.hset(
        key,
        {
            "user_id": "42",
            "status": "queued",
            "progress": "0",
            "created_at": now,
            "updated_at": now,
            "error": "   ",
        },
    )
    store = TurkicJobStore(r)
    loaded = store.load("optbad")
    if loaded is None:
        pytest.fail("expected loaded job")
    assert loaded["error"] is None
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_upload_metadata_missing_content_type() -> None:
    r = FakeRedis()
    key = f"{turkic_job_key('nocontent')}:file"
    r.hset(
        key,
        {
            "file_id": "fid",
            "size": "1",
            "sha256": "abc",
            "content_type": "",
            "created_at": "2024-01-01T00:00:00Z",
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="invalid content_type"):
        store.load_upload_metadata("nocontent")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_upload_metadata_missing_created_at() -> None:
    r = FakeRedis()
    key = f"{turkic_job_key('nocreated')}:file"
    r.hset(
        key,
        {
            "file_id": "fid",
            "size": "1",
            "sha256": "abc",
            "content_type": "text/plain",
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="missing created_at"):
        store.load_upload_metadata("nocreated")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_upload_metadata_invalid_sha256() -> None:
    r = FakeRedis()
    key = f"{turkic_job_key('badsha')}:file"
    r.hset(
        key,
        {
            "file_id": "fid",
            "size": "1",
            "sha256": "",
            "content_type": "text/plain",
            "created_at": "2024-01-01T00:00:00Z",
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="invalid sha256"):
        store.load_upload_metadata("badsha")
    r.assert_only_called({"hset", "expire", "hgetall"})


def test_job_store_upload_metadata_invalid_created_at_type() -> None:
    r = FakeRedis()
    key = f"{turkic_job_key('badcat')}:file"
    r.hset(
        key,
        {
            "file_id": "fid",
            "size": "1",
            "sha256": "abc",
            "content_type": "text/plain",
            "created_at": "   ",
        },
    )
    store = TurkicJobStore(r)
    with pytest.raises(JSONTypeError, match="invalid created_at"):
        store.load_upload_metadata("badcat")
    r.assert_only_called({"hset", "expire", "hgetall"})
