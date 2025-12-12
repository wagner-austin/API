"""Shared test fixtures for handwriting-ai tests."""

from __future__ import annotations

import gzip
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from platform_core.config import _test_hooks as config_test_hooks
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.testing import make_fake_env
from platform_workers.rq_harness import _RedisBytesClient
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient

from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import ArtifactStoreFactoryProtocol, ArtifactStoreProtocol
from handwriting_ai.inference.types import PredictOutput

# =============================================================================
# Fake ArtifactStore for testing
# =============================================================================


class FakeArtifactStore:
    """Fake artifact store that records calls and returns canned responses."""

    def __init__(
        self,
        upload_response: FileUploadResponse | None = None,
        download_error: Exception | None = None,
        download_result: Path | None = None,
    ) -> None:
        self._upload_response: FileUploadResponse = upload_response or {
            "file_id": "fake-file-id",
            "size": 1024,
            "sha256": "fake-sha256-hash",
            "content_type": "application/gzip",
            "created_at": "2024-01-01T00:00:00Z",
        }
        self._download_error = download_error
        self._download_result = download_result
        self.upload_calls: list[tuple[Path, str, str]] = []
        self.download_calls: list[tuple[str, Path, str, str]] = []

    def upload_artifact(
        self,
        dir_path: Path,
        *,
        artifact_name: str,
        request_id: str,
    ) -> FileUploadResponse:
        self.upload_calls.append((dir_path, artifact_name, request_id))
        return self._upload_response

    def download_artifact(
        self,
        file_id: str,
        *,
        dest_dir: Path,
        request_id: str,
        expected_root: str,
    ) -> Path:
        self.download_calls.append((file_id, dest_dir, request_id, expected_root))
        if self._download_error is not None:
            raise self._download_error
        if self._download_result is not None:
            return self._download_result
        return dest_dir / expected_root


def make_fake_artifact_store(
    upload_response: FileUploadResponse | None = None,
    download_error: Exception | None = None,
    download_result: Path | None = None,
) -> ArtifactStoreFactoryProtocol:
    """Create a factory that returns a FakeArtifactStore."""
    store = FakeArtifactStore(upload_response, download_error, download_result)

    def factory(api_url: str, api_key: str) -> ArtifactStoreProtocol:
        _ = (api_url, api_key)  # Unused in fake
        return store

    return factory


# =============================================================================
# Hook reset fixture
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_test_hooks() -> Generator[None, None, None]:
    """Reset all test hooks to their production defaults after each test.

    This ensures tests don't leak hook state to subsequent tests.
    """
    # Store original values - platform hooks
    original_platform_get_env = config_test_hooks.get_env

    # Store original values - handwriting-ai hooks
    original_test_runner = _test_hooks.test_runner
    original_redis_factory = _test_hooks.redis_factory
    original_rq_conn = _test_hooks.rq_conn
    original_rq_queue_factory = _test_hooks.rq_queue_factory
    original_artifact_store_factory = _test_hooks.artifact_store_factory
    original_run_worker = _test_hooks.run_worker
    original_guard_find = _test_hooks.guard_find_monorepo_root
    original_guard_load = _test_hooks.guard_load_orchestrator
    original_pil_image_open = _test_hooks.pil_image_open
    original_import_module = _test_hooks.import_module
    original_thread_factory = _test_hooks.thread_factory
    original_event_factory = _test_hooks.event_factory
    original_load_state_dict_file = _test_hooks.load_state_dict_file
    original_validate_state_dict = _test_hooks.validate_state_dict
    original_submit_predict_override = _test_hooks.submit_predict_override
    original_download_remote_override = _test_hooks.download_remote_override
    original_run_preprocess = _test_hooks.run_preprocess
    original_preprocess_signature = _test_hooks.preprocess_signature

    # Store original values - monitoring hooks
    original_psutil_process = _test_hooks.psutil_process
    original_psutil_virtual_memory = _test_hooks.psutil_virtual_memory
    original_psutil_cpu_count = _test_hooks.psutil_cpu_count
    original_os_getpid = _test_hooks.os_getpid
    original_cgroup_mem_current = _test_hooks.cgroup_mem_current
    original_cgroup_mem_max = _test_hooks.cgroup_mem_max
    original_cgroup_mem_stat = _test_hooks.cgroup_mem_stat

    # Store original values - resource detection hooks
    original_read_text_file = _test_hooks.read_text_file
    original_cgroup_cpu_max = _test_hooks.cgroup_cpu_max
    original_os_cpu_count = _test_hooks.os_cpu_count
    original_detect_resource_limits = _test_hooks.detect_resource_limits

    # Store original values - preprocess hooks
    original_principal_angle_confidence = _test_hooks.principal_angle_confidence

    # Store original values - torch hooks
    original_torch_set_interop_threads = _test_hooks.torch_set_interop_threads
    original_torch_has_set_num_interop_threads = _test_hooks.torch_has_set_num_interop_threads
    original_torch_has_get_num_interop_threads = _test_hooks.torch_has_get_num_interop_threads
    original_torch_get_num_interop_threads = _test_hooks.torch_get_num_interop_threads
    original_torch_cuda_is_available = _test_hooks.torch_cuda_is_available
    original_torch_cuda_current_device = _test_hooks.torch_cuda_current_device
    original_torch_cuda_memory_allocated = _test_hooks.torch_cuda_memory_allocated
    original_torch_cuda_memory_reserved = _test_hooks.torch_cuda_memory_reserved
    original_torch_cuda_max_memory_allocated = _test_hooks.torch_cuda_max_memory_allocated

    # Store original values - safety/monitoring hooks
    original_get_memory_snapshot = _test_hooks.get_memory_snapshot
    original_check_memory_pressure = _test_hooks.check_memory_pressure
    original_is_cgroup_available = _test_hooks.is_cgroup_available
    original_on_batch_check = _test_hooks.on_batch_check
    original_get_logger = _test_hooks.get_logger

    # Store original values - time and os hooks
    original_perf_counter = _test_hooks.perf_counter
    original_os_access = _test_hooks.os_access
    original_os_name = _test_hooks.os_name

    # Store original values - progress hooks
    original_emit_batch = _test_hooks.emit_batch

    # Store original values - inference hooks
    original_build_model = _test_hooks.build_model

    # Store original values - random hooks
    original_random_random = _test_hooks.random_random
    original_random_randint = _test_hooks.random_randint
    original_random_uniform = _test_hooks.random_uniform

    # Store original values - calibration hooks
    original_orchestrator_factory = _test_hooks.orchestrator_factory
    original_safe_loader = _test_hooks.safe_loader
    original_measure_training = _test_hooks.measure_training
    original_gc_collect = _test_hooks.gc_collect
    original_torch_cuda_empty_cache = _test_hooks.torch_cuda_empty_cache
    original_interop_configured_getter = _test_hooks.interop_configured_getter
    original_interop_configured_setter = _test_hooks.interop_configured_setter

    # Store original values - multiprocessing hooks
    original_mp_active_children = _test_hooks.mp_active_children
    original_mp_get_all_start_methods = _test_hooks.mp_get_all_start_methods
    original_mp_get_context = _test_hooks.mp_get_context

    # Store original values - PIL preprocessing hooks
    original_exif_transpose = _test_hooks.exif_transpose
    original_principal_angle = _test_hooks.principal_angle

    # Store original values - digits job hooks
    original_run_training = _test_hooks.run_training
    original_load_settings = _test_hooks.load_settings
    original_make_job_context = _test_hooks.make_job_context

    # Store original values - calibration runner hooks
    original_build_dataset_from_spec = _test_hooks.build_dataset_from_spec
    original_measure_candidate_internal = _test_hooks.measure_candidate_internal
    original_emit_result_file = _test_hooks.emit_result_file
    original_runner_setup_logging = _test_hooks.runner_setup_logging
    original_file_open = _test_hooks.file_open

    # Store original values - calibration cache hooks
    original_now_ts = _test_hooks.now_ts

    # Store original values - engine hooks
    original_path_stat = _test_hooks.path_stat
    original_is_wrapped_state_dict = _test_hooks.is_wrapped_state_dict
    original_is_flat_state_dict = _test_hooks.is_flat_state_dict

    # Store original values - training hooks
    original_log_system_info = _test_hooks.log_system_info
    original_train_epoch = _test_hooks.train_epoch
    original_limit_thread_pools = _test_hooks.limit_thread_pools
    original_calibrate_input_pipeline = _test_hooks.calibrate_input_pipeline

    # Store original values - calibration runner logging hooks
    original_tempfile_mkdtemp = _test_hooks.tempfile_mkdtemp
    original_queue_handler_factory = _test_hooks.queue_handler_factory
    original_queue_listener_factory = _test_hooks.queue_listener_factory

    # Store original values - PIL histogram hook
    original_pil_histogram = _test_hooks.pil_histogram

    # Store original values - memory guard config hook
    original_get_memory_guard_config = _test_hooks.get_memory_guard_config

    # Store original values - training progress module hook
    original_get_training_progress_module = _test_hooks.get_training_progress_module

    yield

    # Restore original values - platform hooks
    config_test_hooks.get_env = original_platform_get_env

    # Restore original values - handwriting-ai hooks
    _test_hooks.test_runner = original_test_runner
    _test_hooks.redis_factory = original_redis_factory
    _test_hooks.rq_conn = original_rq_conn
    _test_hooks.rq_queue_factory = original_rq_queue_factory
    _test_hooks.artifact_store_factory = original_artifact_store_factory
    _test_hooks.run_worker = original_run_worker
    _test_hooks.guard_find_monorepo_root = original_guard_find
    _test_hooks.guard_load_orchestrator = original_guard_load
    _test_hooks.pil_image_open = original_pil_image_open
    _test_hooks.import_module = original_import_module
    _test_hooks.thread_factory = original_thread_factory
    _test_hooks.event_factory = original_event_factory
    _test_hooks.load_state_dict_file = original_load_state_dict_file
    _test_hooks.validate_state_dict = original_validate_state_dict
    _test_hooks.submit_predict_override = original_submit_predict_override
    _test_hooks.download_remote_override = original_download_remote_override
    _test_hooks.run_preprocess = original_run_preprocess
    _test_hooks.preprocess_signature = original_preprocess_signature

    # Restore original values - monitoring hooks
    _test_hooks.psutil_process = original_psutil_process
    _test_hooks.psutil_virtual_memory = original_psutil_virtual_memory
    _test_hooks.psutil_cpu_count = original_psutil_cpu_count
    _test_hooks.os_getpid = original_os_getpid
    _test_hooks.cgroup_mem_current = original_cgroup_mem_current
    _test_hooks.cgroup_mem_max = original_cgroup_mem_max
    _test_hooks.cgroup_mem_stat = original_cgroup_mem_stat

    # Restore original values - resource detection hooks
    _test_hooks.read_text_file = original_read_text_file
    _test_hooks.cgroup_cpu_max = original_cgroup_cpu_max
    _test_hooks.os_cpu_count = original_os_cpu_count
    _test_hooks.detect_resource_limits = original_detect_resource_limits

    # Restore original values - preprocess hooks
    _test_hooks.principal_angle_confidence = original_principal_angle_confidence

    # Restore original values - torch hooks
    _test_hooks.torch_set_interop_threads = original_torch_set_interop_threads
    _test_hooks.torch_has_set_num_interop_threads = original_torch_has_set_num_interop_threads
    _test_hooks.torch_has_get_num_interop_threads = original_torch_has_get_num_interop_threads
    _test_hooks.torch_get_num_interop_threads = original_torch_get_num_interop_threads
    _test_hooks.torch_cuda_is_available = original_torch_cuda_is_available
    _test_hooks.torch_cuda_current_device = original_torch_cuda_current_device
    _test_hooks.torch_cuda_memory_allocated = original_torch_cuda_memory_allocated
    _test_hooks.torch_cuda_memory_reserved = original_torch_cuda_memory_reserved
    _test_hooks.torch_cuda_max_memory_allocated = original_torch_cuda_max_memory_allocated

    # Restore original values - safety/monitoring hooks
    _test_hooks.get_memory_snapshot = original_get_memory_snapshot
    _test_hooks.check_memory_pressure = original_check_memory_pressure
    _test_hooks.is_cgroup_available = original_is_cgroup_available
    _test_hooks.on_batch_check = original_on_batch_check
    _test_hooks.get_logger = original_get_logger

    # Restore original values - time and os hooks
    _test_hooks.perf_counter = original_perf_counter
    _test_hooks.os_access = original_os_access
    _test_hooks.os_name = original_os_name

    # Restore original values - progress hooks
    _test_hooks.emit_batch = original_emit_batch

    # Restore original values - inference hooks
    _test_hooks.build_model = original_build_model

    # Restore original values - random hooks
    _test_hooks.random_random = original_random_random
    _test_hooks.random_randint = original_random_randint
    _test_hooks.random_uniform = original_random_uniform

    # Restore original values - calibration hooks
    _test_hooks.orchestrator_factory = original_orchestrator_factory
    _test_hooks.safe_loader = original_safe_loader
    _test_hooks.measure_training = original_measure_training
    _test_hooks.gc_collect = original_gc_collect
    _test_hooks.torch_cuda_empty_cache = original_torch_cuda_empty_cache
    _test_hooks.interop_configured_getter = original_interop_configured_getter
    _test_hooks.interop_configured_setter = original_interop_configured_setter
    # Reset internal state for interop configuration tracking
    _test_hooks._INTEROP_CONFIGURED = False

    # Restore original values - multiprocessing hooks
    _test_hooks.mp_active_children = original_mp_active_children
    _test_hooks.mp_get_all_start_methods = original_mp_get_all_start_methods
    _test_hooks.mp_get_context = original_mp_get_context

    # Restore original values - PIL preprocessing hooks
    _test_hooks.exif_transpose = original_exif_transpose
    _test_hooks.principal_angle = original_principal_angle

    # Restore original values - digits job hooks
    _test_hooks.run_training = original_run_training
    _test_hooks.load_settings = original_load_settings
    _test_hooks.make_job_context = original_make_job_context

    # Restore original values - calibration runner hooks
    _test_hooks.build_dataset_from_spec = original_build_dataset_from_spec
    _test_hooks.measure_candidate_internal = original_measure_candidate_internal
    _test_hooks.emit_result_file = original_emit_result_file
    _test_hooks.runner_setup_logging = original_runner_setup_logging
    _test_hooks.file_open = original_file_open

    # Restore original values - calibration cache hooks
    _test_hooks.now_ts = original_now_ts

    # Restore original values - engine hooks
    _test_hooks.path_stat = original_path_stat
    _test_hooks.is_wrapped_state_dict = original_is_wrapped_state_dict
    _test_hooks.is_flat_state_dict = original_is_flat_state_dict

    # Restore original values - training hooks
    _test_hooks.log_system_info = original_log_system_info
    _test_hooks.train_epoch = original_train_epoch
    _test_hooks.limit_thread_pools = original_limit_thread_pools
    _test_hooks.calibrate_input_pipeline = original_calibrate_input_pipeline

    # Restore original values - calibration runner logging hooks
    _test_hooks.tempfile_mkdtemp = original_tempfile_mkdtemp
    _test_hooks.queue_handler_factory = original_queue_handler_factory
    _test_hooks.queue_listener_factory = original_queue_listener_factory

    # Restore original values - PIL histogram hook
    _test_hooks.pil_histogram = original_pil_histogram

    # Restore original values - memory guard config hook
    _test_hooks.get_memory_guard_config = original_get_memory_guard_config

    # Restore original values - training progress module hook
    _test_hooks.get_training_progress_module = original_get_training_progress_module


# =============================================================================
# Default test environment
# =============================================================================


@pytest.fixture(autouse=True)
def _default_test_env() -> None:
    """Provide default test environment configuration via hooks."""
    env = make_fake_env(
        {
            "REDIS_URL": "redis://test-redis:6379/0",
            "APP__DATA_BANK_API_URL": "http://test-data-bank:8080",
            "APP__DATA_BANK_API_KEY": "test-api-key",
        }
    )
    config_test_hooks.get_env = env

    def _fake_redis(url: str) -> FakeRedis:
        r = FakeRedis()
        r.sadd("rq:workers", "worker-1")
        return r

    _test_hooks.redis_factory = _fake_redis

    def _fake_rq_conn(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    _test_hooks.rq_conn = _fake_rq_conn

    def _fake_rq_queue(name: str, connection: _RedisBytesClient) -> FakeQueue:
        _ = connection  # Unused in fake
        return FakeQueue(job_id="test-job-id")

    _test_hooks.rq_queue_factory = _fake_rq_queue

    # Set up fake artifact store
    _test_hooks.artifact_store_factory = make_fake_artifact_store()


# =============================================================================
# MNIST data fixture
# =============================================================================


def _write_mnist_raw(root: Path, n: int = 8) -> None:
    """Create minimal MNIST raw data files for testing."""
    raw = (root / "MNIST" / "raw").resolve()
    raw.mkdir(parents=True, exist_ok=True)

    img_path = raw / "train-images-idx3-ubyte.gz"
    rows = 28
    cols = 28
    total = int(n) * rows * cols
    header = (
        (2051).to_bytes(4, "big")
        + int(n).to_bytes(4, "big")
        + rows.to_bytes(4, "big")
        + cols.to_bytes(4, "big")
    )
    payload = bytes([0]) * total
    with gzip.open(img_path, "wb") as f:
        f.write(header)
        f.write(payload)

    lbl_path = raw / "train-labels-idx1-ubyte.gz"
    header_l = (2049).to_bytes(4, "big") + int(n).to_bytes(4, "big")
    labels = bytes([i % 10 for i in range(int(n))])
    with gzip.open(lbl_path, "wb") as f:
        f.write(header_l)
        f.write(labels)


class MnistRawWriter:
    """Callable class for creating MNIST raw test data."""

    def __call__(self, root: Path, n: int = 8) -> None:
        _write_mnist_raw(root, n)


def _make_mnist_raw_writer() -> MnistRawWriter:
    """Factory function for MnistRawWriter fixture."""
    return MnistRawWriter()


write_mnist_raw = pytest.fixture(_make_mnist_raw_writer)


# =============================================================================
# Fake Future for predict tests
# =============================================================================


class FakeFuture:
    """Fake Future for testing submit_predict hook.

    Wraps a callable that produces the result or raises an exception.
    """

    def __init__(
        self,
        result_fn: Callable[[float | None], PredictOutput] | None = None,
    ) -> None:
        self._fn = result_fn
        self._cancelled = False

    def result(self, timeout: float | None = None) -> PredictOutput:
        if self._fn is None:
            raise RuntimeError("No result function provided")
        return self._fn(timeout)

    def cancel(self) -> bool:
        self._cancelled = True
        return True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


# =============================================================================
# Legacy fixture for digits tests that need direct redis access
# =============================================================================


@pytest.fixture()
def digits_redis() -> FakeRedis:
    """Provide a typed Redis stub for digits jobs and capture published events.

    Sets the redis_factory hook to return this stub so digits job code
    gets the same instance for assertions.
    """
    stub = FakeRedis()

    def _redis_for_kv(url: str) -> FakeRedis:
        _ = url  # unused
        return stub

    _test_hooks.redis_factory = _redis_for_kv
    return stub


__all__ = [
    "FakeArtifactStore",
    "FakeFuture",
    "MnistRawWriter",
    "digits_redis",
    "make_fake_artifact_store",
    "write_mnist_raw",
]
