"""Inference worker: outcomes and errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import load_json_str
from platform_core.trainer_keys import artifact_file_id_key, generate_key, score_key
from platform_workers.testing import FakeRedis
from tests.worker._inference_worker_support import (
    _FakeArtifactStore,
    _FakeBackendNoTopk,
    _FakeBackendWithTopk,
    _FakeServiceContainer,
    _FakeTokenizerHandle,
    _make_manifest,
    _SettingsFactory,
)

from model_trainer.core import _test_hooks
from model_trainer.core._hook_protocols import ArtifactStoreProto, ServiceContainerProto
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import GenerateJobPayload, ScoreJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.worker import generate_job, score_job


class TestProcessScoreJob:
    def test_score_job_success(
        self,
        tmp_path: Path,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test successful score job execution."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)
        run_dir = models_dir / "run123"
        run_dir.mkdir(parents=True)

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(_make_manifest("char_lstm", "tok123"), encoding="utf-8")

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run123"), "file123")

        # Override hooks
        _test_hooks.load_settings = lambda: settings
        _test_hooks.kv_store_factory = lambda url: fake_redis

        def _fake_load_tokenizer(settings: Settings, tokenizer_id: str) -> TokenizerHandle:
            return _FakeTokenizerHandle()

        _test_hooks.load_tokenizer_for_training = _fake_load_tokenizer

        backend = _FakeBackendWithTopk()

        def _fake_service_container(settings: Settings) -> ServiceContainerProto:
            return _FakeServiceContainer(settings, fake_redis, backend)

        _test_hooks.service_container_from_settings = _fake_service_container

        payload: ScoreJobPayload = {
            "run_id": "run123",
            "request_id": "req123",
            "text": "hello",
            "path": None,
            "detail_level": "per_char",
            "top_k": 5,
            "seed": 42,
        }

        score_job.process_score_job(payload)

        cached = fake_redis.get(score_key("run123", "req123"))
        assert isinstance(cached, str) and len(cached) > 0
        obj = load_json_str(cached)
        assert isinstance(obj, dict) and obj.get("status") == "completed"
        assert obj.get("loss") == 1.5
        assert obj.get("perplexity") == 4.5
        fake_redis.assert_only_called({"set", "get"})

    def test_score_job_no_artifact_pointer_fails(
        self,
        tmp_path: Path,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test score job fails when artifact pointer is missing."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()

        _test_hooks.load_settings = lambda: settings
        _test_hooks.kv_store_factory = lambda url: fake_redis

        payload: ScoreJobPayload = {
            "run_id": "run123",
            "request_id": "req123",
            "text": "hello",
            "path": None,
            "detail_level": "summary",
            "top_k": None,
            "seed": None,
        }

        with pytest.raises(AppError, match="artifact pointer not found"):
            score_job.process_score_job(payload)

        cached = fake_redis.get(score_key("run123", "req123"))
        assert isinstance(cached, str) and len(cached) > 0
        obj = load_json_str(cached)
        assert isinstance(obj, dict) and obj.get("status") == "failed"
        fake_redis.assert_only_called({"set", "get"})

    def test_score_job_no_topk_or_surprisal(
        self,
        tmp_path: Path,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test score job with no topk or surprisal in result."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)
        run_dir = models_dir / "run123"
        run_dir.mkdir(parents=True)

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(_make_manifest("char_lstm", "tok123"), encoding="utf-8")

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run123"), "file123")

        _test_hooks.load_settings = lambda: settings
        _test_hooks.kv_store_factory = lambda url: fake_redis

        def _fake_load_tokenizer(settings: Settings, tokenizer_id: str) -> TokenizerHandle:
            return _FakeTokenizerHandle()

        _test_hooks.load_tokenizer_for_training = _fake_load_tokenizer

        backend = _FakeBackendNoTopk()

        def _fake_service_container(settings: Settings) -> ServiceContainerProto:
            return _FakeServiceContainer(settings, fake_redis, backend)

        _test_hooks.service_container_from_settings = _fake_service_container

        payload: ScoreJobPayload = {
            "run_id": "run123",
            "request_id": "req123",
            "text": "hello",
            "path": None,
            "detail_level": "summary",
            "top_k": None,
            "seed": None,
        }

        score_job.process_score_job(payload)

        cached = fake_redis.get(score_key("run123", "req123"))
        assert isinstance(cached, str) and len(cached) > 0
        obj = load_json_str(cached)
        assert isinstance(obj, dict) and obj.get("status") == "completed"
        assert obj.get("surprisal") is None
        assert obj.get("topk") is None
        assert obj.get("tokens") is None
        fake_redis.assert_only_called({"set", "get"})


class TestProcessGenerateJob:
    def test_generate_job_success(
        self,
        tmp_path: Path,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test successful generate job execution."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)
        run_dir = models_dir / "run123"
        run_dir.mkdir(parents=True)

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(_make_manifest("char_lstm", "tok123"), encoding="utf-8")

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run123"), "file123")

        _test_hooks.load_settings = lambda: settings
        _test_hooks.kv_store_factory = lambda url: fake_redis

        def _fake_load_tokenizer(settings: Settings, tokenizer_id: str) -> TokenizerHandle:
            return _FakeTokenizerHandle()

        _test_hooks.load_tokenizer_for_training = _fake_load_tokenizer

        backend = _FakeBackendWithTopk()

        def _fake_service_container(settings: Settings) -> ServiceContainerProto:
            return _FakeServiceContainer(settings, fake_redis, backend)

        _test_hooks.service_container_from_settings = _fake_service_container

        payload: GenerateJobPayload = {
            "run_id": "run123",
            "request_id": "req123",
            "prompt_text": "Hello",
            "prompt_path": None,
            "max_new_tokens": 10,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "stop_on_eos": True,
            "stop_sequences": [],
            "seed": 42,
            "num_return_sequences": 1,
        }

        generate_job.process_generate_job(payload)

        cached = fake_redis.get(generate_key("run123", "req123"))
        assert isinstance(cached, str) and len(cached) > 0
        obj = load_json_str(cached)
        assert isinstance(obj, dict) and obj.get("status") == "completed"
        assert obj.get("outputs") == ["generated text here"]
        assert obj.get("steps") == 10
        assert obj.get("eos_terminated") == [True]
        fake_redis.assert_only_called({"set", "get"})

    def test_generate_job_no_artifact_pointer_fails(
        self,
        tmp_path: Path,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test generate job fails when artifact pointer is missing."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()

        _test_hooks.load_settings = lambda: settings
        _test_hooks.kv_store_factory = lambda url: fake_redis

        payload: GenerateJobPayload = {
            "run_id": "run123",
            "request_id": "req123",
            "prompt_text": "Hello",
            "prompt_path": None,
            "max_new_tokens": 10,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "stop_on_eos": True,
            "stop_sequences": [],
            "seed": None,
            "num_return_sequences": 1,
        }

        with pytest.raises(AppError, match="artifact pointer not found"):
            generate_job.process_generate_job(payload)

        cached = fake_redis.get(generate_key("run123", "req123"))
        assert isinstance(cached, str) and len(cached) > 0
        obj = load_json_str(cached)
        assert isinstance(obj, dict) and obj.get("status") == "failed"
        fake_redis.assert_only_called({"set", "get"})


class TestArtifactDownloadPaths:
    """Tests for artifact download scenarios covering lines 846-852, 857, 958-964, 969."""

    def test_score_job_with_download(
        self,
        tmp_path: Path,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test score job when artifact needs to be downloaded (covers lines 846-852)."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)
        # NOTE: We do NOT create run_dir - it will be "downloaded"

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run_download"), "file_download_123")

        _test_hooks.load_settings = lambda: settings
        _test_hooks.kv_store_factory = lambda url: fake_redis

        def _fake_load_tokenizer(settings: Settings, tokenizer_id: str) -> TokenizerHandle:
            return _FakeTokenizerHandle()

        _test_hooks.load_tokenizer_for_training = _fake_load_tokenizer

        backend = _FakeBackendWithTopk()

        def _fake_service_container(settings: Settings) -> ServiceContainerProto:
            return _FakeServiceContainer(settings, fake_redis, backend)

        _test_hooks.service_container_from_settings = _fake_service_container

        # Create factory function that captures manifest content
        manifest_content = _make_manifest("char_lstm", "tok123")

        def _make_fake_store(
            base_url: str, api_key: str, *, timeout_seconds: float = 600.0
        ) -> ArtifactStoreProto:
            return _FakeArtifactStore(
                base_url,
                api_key,
                timeout_seconds=timeout_seconds,
                include_manifest=True,
                manifest_content=manifest_content,
            )

        _test_hooks.artifact_store_factory = _make_fake_store

        payload: ScoreJobPayload = {
            "run_id": "run_download",
            "request_id": "req_download",
            "text": "hello",
            "path": None,
            "detail_level": "per_char",
            "top_k": 5,
            "seed": 42,
        }

        score_job.process_score_job(payload)

        cached = fake_redis.get(score_key("run_download", "req_download"))
        assert isinstance(cached, str) and len(cached) > 0
        result = load_json_str(cached)
        assert isinstance(result, dict) and result.get("status") == "completed"
        fake_redis.assert_only_called({"set", "get"})

    def test_score_job_missing_manifest_after_download(
        self,
        tmp_path: Path,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test score job when manifest is missing after download (covers line 857)."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run_no_manifest"), "file_no_manifest_123")

        _test_hooks.load_settings = lambda: settings
        _test_hooks.kv_store_factory = lambda url: fake_redis

        # Create factory without manifest
        def _make_fake_store_no_manifest(
            base_url: str, api_key: str, *, timeout_seconds: float = 600.0
        ) -> ArtifactStoreProto:
            return _FakeArtifactStore(
                base_url,
                api_key,
                timeout_seconds=timeout_seconds,
                include_manifest=False,
                manifest_content=None,
            )

        _test_hooks.artifact_store_factory = _make_fake_store_no_manifest

        payload: ScoreJobPayload = {
            "run_id": "run_no_manifest",
            "request_id": "req_no_manifest",
            "text": "hello",
            "path": None,
            "detail_level": "summary",
            "top_k": None,
            "seed": None,
        }

        with pytest.raises(AppError, match="manifest missing"):
            score_job.process_score_job(payload)
        fake_redis.assert_only_called({"set", "get"})

    def test_generate_job_with_download(
        self,
        tmp_path: Path,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test generate job when artifact needs to be downloaded (covers lines 958-964)."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run_gen_download"), "file_gen_download_123")

        _test_hooks.load_settings = lambda: settings
        _test_hooks.kv_store_factory = lambda url: fake_redis

        def _fake_load_tokenizer(settings: Settings, tokenizer_id: str) -> TokenizerHandle:
            return _FakeTokenizerHandle()

        _test_hooks.load_tokenizer_for_training = _fake_load_tokenizer

        backend = _FakeBackendWithTopk()

        def _fake_service_container(settings: Settings) -> ServiceContainerProto:
            return _FakeServiceContainer(settings, fake_redis, backend)

        _test_hooks.service_container_from_settings = _fake_service_container

        manifest_content = _make_manifest("char_lstm", "tok123")

        def _make_fake_store(
            base_url: str, api_key: str, *, timeout_seconds: float = 600.0
        ) -> ArtifactStoreProto:
            return _FakeArtifactStore(
                base_url,
                api_key,
                timeout_seconds=timeout_seconds,
                include_manifest=True,
                manifest_content=manifest_content,
            )

        _test_hooks.artifact_store_factory = _make_fake_store

        payload: GenerateJobPayload = {
            "run_id": "run_gen_download",
            "request_id": "req_gen_download",
            "prompt_text": "Hello",
            "prompt_path": None,
            "max_new_tokens": 10,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "stop_on_eos": True,
            "stop_sequences": [],
            "seed": None,
            "num_return_sequences": 1,
        }

        generate_job.process_generate_job(payload)

        cached = fake_redis.get(generate_key("run_gen_download", "req_gen_download"))
        assert isinstance(cached, str) and len(cached) > 0
        result = load_json_str(cached)
        assert isinstance(result, dict) and result.get("status") == "completed"
        fake_redis.assert_only_called({"set", "get"})

    def test_generate_job_missing_manifest_after_download(
        self,
        tmp_path: Path,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test generate job when manifest is missing after download (covers line 969)."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run_gen_no_manifest"), "file_gen_no_manifest_123")

        _test_hooks.load_settings = lambda: settings
        _test_hooks.kv_store_factory = lambda url: fake_redis

        def _make_fake_store_no_manifest(
            base_url: str, api_key: str, *, timeout_seconds: float = 600.0
        ) -> ArtifactStoreProto:
            return _FakeArtifactStore(
                base_url,
                api_key,
                timeout_seconds=timeout_seconds,
                include_manifest=False,
                manifest_content=None,
            )

        _test_hooks.artifact_store_factory = _make_fake_store_no_manifest

        payload: GenerateJobPayload = {
            "run_id": "run_gen_no_manifest",
            "request_id": "req_gen_no_manifest",
            "prompt_text": "Hello",
            "prompt_path": None,
            "max_new_tokens": 10,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "stop_on_eos": True,
            "stop_sequences": [],
            "seed": None,
            "num_return_sequences": 1,
        }

        with pytest.raises(AppError, match="manifest missing"):
            generate_job.process_generate_job(payload)
        fake_redis.assert_only_called({"set", "get"})
