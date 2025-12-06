from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from turkic_api.api import models
from turkic_api.api.jobs import _decode_job_params
from turkic_api.api.main import _to_json_simple
from turkic_api.api.types import JsonDict, UnknownJson
from turkic_api.core import transliteval as te
from turkic_api.core.corpus import CorpusService, LocalCorpusService
from turkic_api.core.models import ProcessSpec


def test_to_json_simple_supports_primitives_and_datetime() -> None:
    dt = datetime(2024, 1, 1, tzinfo=UTC)
    payload: dict[str, str | int | float | bool | None | datetime] = {
        "a": 1,
        "b": True,
        "c": None,
        "d": 3.5,
        "ts": dt,
    }
    out = _to_json_simple(payload)
    assert out["ts"] == dt.isoformat()


def test_decode_job_params_branches() -> None:
    good = _decode_job_params(
        {
            "user_id": 42,
            "source": "oscar",
            "language": "kk",
            "script": "latn",
            "max_sentences": 10,
            "transliterate": True,
            "confidence_threshold": 0.4,
        }
    )
    assert good["script"] == "Latn"

    blank = _decode_job_params(
        {
            "user_id": 42,
            "source": "oscar",
            "language": "kk",
            "script": "   ",
            "max_sentences": 10,
            "transliterate": True,
            "confidence_threshold": 0.4,
        }
    )
    assert blank["script"] is None

    with pytest.raises(ValueError, match="Invalid script"):
        _decode_job_params(
            {
                "user_id": 42,
                "source": "oscar",
                "language": "kk",
                "script": "bad",
                "max_sentences": 10,
                "transliterate": True,
                "confidence_threshold": 0.4,
            }
        )
    with pytest.raises(ValueError, match="Invalid source or language"):
        _decode_job_params(
            {
                "user_id": 42,
                "source": "bad",
                "language": "kk",
                "max_sentences": 10,
                "transliterate": True,
                "confidence_threshold": 0.4,
            }
        )
    payload_bad: dict[str, UnknownJson] = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "max_sentences": "x",
        "transliterate": True,
        "confidence_threshold": 0.4,
    }
    with pytest.raises(TypeError):
        _decode_job_params(payload_bad)


def test_models_parse_job_create_and_json_helpers() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    base_payload: JsonDict = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "script": None,
        "max_sentences": 10,
        "transliterate": True,
        "confidence_threshold": 0.5,
    }
    parsed = models.parse_job_create(base_payload)
    assert parsed["language"] == "kk"

    defaults = models.parse_job_create({"user_id": 42, "source": "wikipedia", "language": "uz"})
    assert defaults["max_sentences"] == 1000
    assert defaults["transliterate"] is True
    assert defaults["confidence_threshold"] == 0.95

    http_exc_type: type[Exception] = AppError
    with pytest.raises(http_exc_type, match="source is required"):
        models.parse_job_create({"user_id": 42, "language": "kk"})

    job_resp_data: dict[str, str | int] = {
        "job_id": "1",
        "user_id": 42,
        "status": "queued",
        "created_at": now.isoformat(),
    }
    job_resp_json = dump_json_str(job_resp_data)
    job_resp = models.parse_job_response_json(job_resp_json)
    assert job_resp["status"] == "queued"

    job_status_data: dict[str, str | int | None] = {
        "job_id": "1",
        "user_id": 42,
        "status": "completed",
        "progress": 100,
        "message": "ok",
        "result_url": "u",
        "file_id": "f",
        "upload_status": "uploaded",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "error": None,
    }
    job_status_json = dump_json_str(job_status_data)
    job_status = models.parse_job_status_json(job_status_json)
    assert job_status["upload_status"] == "uploaded"


def test_models_literal_and_conversion_paths() -> None:
    base: dict[str, UnknownJson] = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "max_sentences": 5,
        "transliterate": False,
        "confidence_threshold": 0.5,
    }
    for script in (None, "Latn", "Cyrl", "Arab"):
        payload = dict(base)
        payload["script"] = script
        decoded_job = models._decode_job_create_from_unknown(payload)
        assert decoded_job["script"] == script

    payload2: JsonDict = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "script": "Latn",
        "max_sentences": 3,
        "transliterate": True,
        "confidence_threshold": 0.3,
    }
    parsed2 = models.parse_job_create(payload2)
    assert parsed2["max_sentences"] == 3

    ts = datetime.now(UTC).isoformat()
    for status in ("queued", "processing", "completed", "failed"):
        response_payload = {"job_id": "x", "user_id": 42, "status": status, "created_at": ts}
        job_response = models.parse_job_response_json(dump_json_str(response_payload))
        assert job_response["status"] == status

    base_status: dict[str, str | int | None] = {
        "job_id": "id1",
        "user_id": 42,
        "progress": 1,
        "message": None,
        "result_url": None,
        "file_id": None,
        "upload_status": None,
        "created_at": ts,
        "updated_at": ts,
        "error": None,
    }
    for status in ("queued", "processing", "completed", "failed"):
        status_payload: dict[str, str | int | None] = {
            "job_id": base_status["job_id"],
            "user_id": base_status["user_id"],
            "progress": base_status["progress"],
            "message": base_status["message"],
            "result_url": base_status["result_url"],
            "file_id": base_status["file_id"],
            "upload_status": base_status["upload_status"],
            "created_at": base_status["created_at"],
            "updated_at": base_status["updated_at"],
            "error": base_status["error"],
            "status": status,
        }
        parsed_status = models.parse_job_status_json(dump_json_str(status_payload))
        assert parsed_status["status"] == status


def test_corpus_service_base_raises() -> None:
    svc = CorpusService()
    spec: ProcessSpec = {
        "source": "oscar",
        "language": "kk",
        "max_sentences": 1,
        "transliterate": True,
        "confidence_threshold": 0.0,
    }
    with pytest.raises(NotImplementedError):
        list(svc.stream(spec))


def test_local_corpus_service_stream_and_errors(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    corpus_dir = data_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    corpus_file = corpus_dir / "oscar_kk.txt"
    corpus_file.write_text("line1\n\nline2\nline3\n", encoding="utf-8")

    svc = LocalCorpusService(str(data_dir))
    spec: ProcessSpec = {
        "source": "oscar",
        "language": "kk",
        "max_sentences": 2,
        "transliterate": True,
        "confidence_threshold": 0.0,
    }
    collected = list(svc.stream(spec))
    assert collected == ["line1", "line2"]

    missing = LocalCorpusService(str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        list(missing.stream(spec))


def test_transliteval_edge_cases_and_rule_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rule lhs must be non-empty"):
        te._rule("", "x")

    match = re.match(r"^(?P<x>\d+)?$", "")
    if match is None:
        pytest.fail("expected match")
    with pytest.raises(te.RuleParseError):
        te._group(match, "x", 1, "")

    with pytest.raises(te.RuleParseError):
        te._parse_rule_stmt("$$$", {}, 1)

    bad_path = te._RULES_DIR / "unclosed_macro_test.rules"
    bad_path.write_text("$A = [abc\n", encoding="utf-8")
    try:
        with pytest.raises(te.RuleParseError, match="macro definition missing closing"):
            te.load_rules("unclosed_macro_test.rules")
    finally:
        bad_path.unlink(missing_ok=True)

    # Test _truncate_output: simple case where entire last item is removed
    out: list[str] = ["abcd"]
    te._truncate_output(out, 2)
    assert out == ["ab"]

    # Test _truncate_output: case where last item is partially truncated (lines 288-289)
    out2: list[str] = ["abc", "defgh"]
    te._truncate_output(out2, 2)
    assert out2 == ["abc", "def"]

    # Verify all production rule files parse and apply correctly.
    # Production files follow naming convention: {lang}_{fmt}.rules (e.g., kk_ipa.rules).
    # The format part is either "ipa" or "lat". This filter excludes temp test files
    # created during parallel test execution (e.g., bad_macro.rules, delete.rules).
    rule_dir = te._RULES_DIR
    production_files: tuple[Path, ...] = tuple(
        p
        for p in rule_dir.glob("*.rules")
        if "_" in p.stem and p.stem.split("_")[-1] in ("ipa", "lat")
    )
    for path in production_files:
        rules = te.load_rules(path.name)
        te.apply_rules("test", rules)
