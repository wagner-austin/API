"""Default (production) implementations behind the Model-Trainer hooks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from pathlib import Path

import httpx
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import _JSONInputValue as JSONInputValue
from platform_ml.testing import (
    WandbModuleProtocol as WandbModuleLike,
)
from platform_workers.redis import (
    _RedisBytesClient,
)
from platform_workers.rq_harness import RQClientQueue, RQRetryLike, rq_queue, rq_retry

from model_trainer.api.schemas.tokenizers import (
    TokenizerTrainRequest,
    TokenizerTrainResponse,
)
from model_trainer.core._hook_protocols import (
    ArtifactStoreProto,
    DiskUsageProto,
    PathIterator,
    RandomLikeProto,
    ScandirIterator,
    ServiceContainerProto,
)
from model_trainer.core._hook_protocols_ml import (
    CorpusCacheCleanupServiceProto,
    CorpusFetcherProto,
    TokenizerCleanupServiceProto,
    TokenizerOrchestratorProto,
)
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.dataset import CorpusSplit, DatasetConfig
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.types import LMModelProto, TorchStateValue


def _default_rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
    """Production rq_queue - used as default hook."""
    return rq_queue(name, connection)


def _default_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
    """Production rq_retry - used as default hook."""
    return rq_retry(max_retries=max_retries, intervals=intervals)


def _default_load_settings() -> Settings:
    """Production load_settings - used as default hook."""
    from model_trainer.core.config.settings import load_settings as _load

    return _load()


def _default_artifact_store(
    base_url: str,
    api_key: str,
    *,
    timeout_seconds: float = 600.0,
) -> ArtifactStoreProto:
    """Production ArtifactStore - used as default hook.

    Validates its OWN credentials here rather than leaving that to callers.
    The check used to live in ``_upload_and_persist_pointer``, which meant a
    precondition belonging to this HTTP-backed store gated every store: a
    filesystem-backed implementation that needs no credentials was refused
    before the factory was ever reached. That cost a completed 20-epoch run,
    which trained for 49 minutes and then failed to save.

    Raises:
        AppError: With ``ARTIFACT_UPLOAD_FAILED`` when either credential is
            absent, since this store cannot reach the data bank without both.
    """
    from platform_core.data_bank_client import DataBankClient
    from platform_ml import ArtifactStore

    if base_url.strip() == "" or api_key.strip() == "":
        raise AppError(
            ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED,
            "data-bank-api configuration missing for artifact upload",
            model_trainer_status_for(ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED),
        )
    client = DataBankClient(base_url, api_key, timeout_seconds=timeout_seconds)
    return ArtifactStore(client)


def _default_service_container_from_settings(settings: Settings) -> ServiceContainerProto:
    """Production ServiceContainer.from_settings - used as default hook."""
    from model_trainer.core.services.container import ServiceContainer

    return ServiceContainer.from_settings(settings)


def _default_corpus_fetcher_factory(
    api_url: str, api_key: str, cache_dir: Path
) -> CorpusFetcherProto:
    """Production CorpusFetcher factory - used as default hook."""
    from model_trainer.core.services.data.corpus_fetcher import CorpusFetcher

    return CorpusFetcher(api_url, api_key, cache_dir)


def _default_load_tokenizer_for_training(settings: Settings, tokenizer_id: str) -> TokenizerHandle:
    """Production load_tokenizer_for_training - used as default hook."""
    from model_trainer.worker.job_utils import (
        load_tokenizer_for_training as _load_tok,
    )

    return _load_tok(settings, tokenizer_id)


def _default_httpx_client_factory(*, timeout_seconds: float = 30.0) -> httpx.Client:
    """Production httpx.Client factory - used as default hook."""
    return httpx.Client(timeout=timeout_seconds)


def _default_cuda_is_available() -> bool:
    """Production cuda_is_available - used as default hook."""
    import torch

    return torch.cuda.is_available()


def _default_cuda_device_name() -> str:
    """Production cuda_device_name - used as default hook.

    Callers gate on the run's device being "cuda" (which _setup_device has
    already proven available); repeating the check here would hide a caller
    that forgot the gate. Calling this initialises a CUDA context in the
    process, which is exactly why cpu-device runs must not reach it.
    """
    import torch

    return torch.cuda.get_device_name(0)


def _default_cuda_driver_version() -> str:
    """Production cuda_driver_version - used as default hook.

    Read from ``nvidia-smi`` rather than from torch. ``torch.version.cuda``
    is the CUDA runtime the wheel was BUILT against (12.4 here) and is not
    the driver; reporting it as one would put a wrong value in a field whose
    whole purpose is telling two otherwise-identical configurations apart.
    torch 2.6 exposes no public driver accessor -- everything NVML-side under
    ``torch.cuda`` is underscore-private.

    Callers gate on the run's device being "cuda", which means CUDA
    initialised, which means the driver answered. A failure here is therefore
    a real fault and propagates: a fingerprint that quietly records "unknown"
    for a run that HAD a driver would make two different configurations
    compare equal, which is the one outcome this field exists to prevent.

    Returns:
        The NVIDIA driver version, e.g. ``"591.86"``.

    Raises:
        CalledProcessError: When nvidia-smi exits non-zero.
        FileNotFoundError: When nvidia-smi is not present.
    """
    import subprocess as _sp

    out = _sp.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        stderr=_sp.DEVNULL,
    )
    return out.decode("utf-8").strip().splitlines()[0].strip()


def _default_env_git_commit() -> str | None:
    """Production env_git_commit - used as default hook.

    The deployed image carries no .git directory, so `git rev-parse` inside
    the container can never answer; the build bakes the commit into the
    GIT_COMMIT environment variable instead. An empty or unset variable is
    None: the manifest records that the commit was not stamped rather than
    an empty string that reads as a value. Reads through the platform config
    env hook, the one sanctioned environment accessor.
    """
    from platform_core.config import config_test_hooks

    value = config_test_hooks.get_env("GIT_COMMIT")
    return value if value else None


def _default_env_image_digest() -> str | None:
    """Production env_image_digest - used as default hook.

    The image cannot compute its own digest from inside itself: the digest
    covers the whole squashfs, including the file that would be doing the
    computing. The launcher knows it -- it is what the job's spec pins -- so
    it exports IMAGE_DIGEST and this reads it.

    An empty or unset variable is None, which the fingerprint records as
    unknown. That is the honest answer for a run out of a directory
    environment, where there is no image and therefore no digest, and it
    compares as a difference against every known digest rather than matching
    all of them.
    """
    from platform_core.config import config_test_hooks

    value = config_test_hooks.get_env("IMAGE_DIGEST")
    return value if value else None


def _default_model_dir(settings: Settings, run_id: str) -> Path:
    """Production model_dir - used as default hook."""
    from model_trainer.core.infra.paths import model_dir as _model_dir

    return _model_dir(settings, run_id)


def _default_split_corpus(cfg: DatasetConfig) -> CorpusSplit:
    """Production split_corpus - used as default hook.

    Args:
        cfg: Dataset configuration with corpus path and split ratios.

    Returns:
        The three disjoint partitions, as corpus lines.
    """
    from model_trainer.core.services.training.dataset_builder import (
        split_corpus as _split,
    )

    return _split(cfg)


def _default_freeze_embeddings(model: LMModelProto) -> None:
    """Production freeze_embeddings - used as default hook."""
    from model_trainer.core.services.training.trainer_grad_utils import (
        _freeze_embeddings as _freeze,
    )

    _freeze(model)


def _default_shutil_which(cmd: str) -> str | None:
    """Production shutil_which - used as default hook."""
    import shutil

    return shutil.which(cmd)


def _default_spm_require_cli() -> None:
    """Production spm_require_cli - used as default hook."""
    from platform_ml import sentencepiece as spm

    spm.require_module()


def _default_spm_train(files: list[str], *, model_prefix: str, vocab_size: int) -> None:
    """Production spm_train - used as default hook."""
    from model_trainer.core.services.tokenizer.spm_backend import (
        _spm_train as _real_spm_train,
    )

    _real_spm_train(files, model_prefix=model_prefix, vocab_size=vocab_size)


def _default_spm_encode_ids(model_path: str, text: str) -> list[int]:
    """Production spm_encode_ids - used as default hook."""
    from model_trainer.core.services.tokenizer.spm_backend import (
        _spm_encode_ids as _real_spm_encode_ids,
    )

    return _real_spm_encode_ids(model_path, text)


def _default_random_factory(seed: int) -> RandomLikeProto:
    """Production random.Random factory."""
    import random

    return random.Random(seed)


def _default_shutil_rmtree(path: Path | str) -> None:
    """Production shutil.rmtree."""
    import shutil

    shutil.rmtree(path)


def _default_os_utime(path: Path | str) -> None:
    """Production os.utime, stamping the path with the current time."""
    import os

    os.utime(path, None)


def _default_load_wandb_module() -> WandbModuleLike:
    """Production wandb module loader."""
    from platform_ml.wandb_publisher import _load_wandb_module as _load

    return _load()


def _default_load_gpt2_model(path: str) -> LMModelProto:
    """Production load_gpt2_model."""
    from model_trainer.core.services.model.backends.gpt2.hf_gpt2 import (
        load_gpt2_model as _load,
    )

    return _load(path)


def _default_sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Production _sample_token."""
    from model_trainer.core.services.model.backends.char_lstm.generate import (
        _sample_token as _sample,
    )

    return _sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)


def _default_spm_decode_ids(model_path: str, ids: list[int]) -> str:
    """Production spm_decode_ids."""
    from model_trainer.core.services.tokenizer.spm_backend import (
        _spm_decode_ids as _decode,
    )

    return _decode(model_path, ids)


def _default_os_scandir(path: str) -> ScandirIterator:
    """Production os.scandir - used as default hook."""
    import os

    return os.scandir(path)


def _default_shutil_disk_usage(path: str) -> DiskUsageProto:
    """Production shutil.disk_usage - used as default hook."""
    import shutil

    return shutil.disk_usage(path)


def _default_path_unlink(path: Path) -> None:
    """Production Path.unlink - used as default hook."""
    path.unlink()


def _default_time_sleep(seconds: float) -> None:
    """Production time.sleep - used as default hook."""
    import time

    time.sleep(seconds)


def _default_path_iterdir(path: Path) -> PathIterator:
    """Production Path.iterdir - used as default hook."""
    return path.iterdir()


def _default_corpus_cache_cleanup_service_factory(
    *, settings: Settings
) -> CorpusCacheCleanupServiceProto:
    """Production CorpusCacheCleanupService factory."""
    from model_trainer.core.services.data.corpus_cache_cleanup import (
        CorpusCacheCleanupService,
    )

    return CorpusCacheCleanupService(settings=settings)


def _default_tokenizer_cleanup_service_factory(
    *, settings: Settings
) -> TokenizerCleanupServiceProto:
    """Production TokenizerCleanupService factory."""
    from model_trainer.core.services.tokenizer.tokenizer_cleanup import (
        TokenizerCleanupService,
    )

    return TokenizerCleanupService(settings=settings)


def _default_dump_json_str(value: JSONInputValue, *, compact: bool = True) -> str:
    """Production dump_json_str - used as default hook."""
    from platform_core.json_utils import dump_json_str as _dump

    return _dump(value, compact=compact)


def _default_tokenizer_enqueue(
    orchestrator: TokenizerOrchestratorProto,
    req: TokenizerTrainRequest,
) -> TokenizerTrainResponse | None:
    """Enqueue tokenizer training through the orchestrator.

    Args:
        orchestrator: Orchestrator to enqueue through.
        req: Validated training request.

    Returns:
        The enqueue response, or None if the enqueue failed.
    """
    return orchestrator.enqueue_training(req)


def _default_load_prepared_gpt2_from_handle(
    artifact_path: str, tokenizer: TokenizerHandle | None
) -> PreparedLMModel:
    """Production load_prepared_gpt2_from_handle - used as default hook."""
    from model_trainer.core.services.model.backends.gpt2.io import (
        load_prepared_gpt2_from_handle as _load,
    )

    return _load(artifact_path, tokenizer)


def _default_torch_cuda_max_memory_allocated() -> int:
    """Production torch.cuda.max_memory_allocated - used as default hook.

    A thin adapter over torch. Whether CUDA is present is the caller's
    question, and `_default_gpu_max_memory_allocated` already asks it through
    the `cuda_is_available` hook before delegating here; repeating the check
    added a branch that no caller can reach and that no machine with a GPU can
    execute.

    Returns:
        Peak GPU memory allocated in bytes.
    """
    return torch.cuda.max_memory_allocated()


def _default_torch_cuda_reset_peak_memory_stats() -> None:
    """Production torch.cuda.reset_peak_memory_stats - used as default hook.

    A thin adapter over torch; `_default_gpu_reset_peak_memory_stats` owns the
    availability check.
    """
    torch.cuda.reset_peak_memory_stats()


def _default_time_monotonic() -> float:
    """Production time.monotonic - used as default hook."""
    import time

    return time.monotonic()


def _default_time_wall_clock() -> float:
    """Production time.time - used as default hook.

    Returns:
        Seconds since the Unix epoch.
    """
    import time

    return time.time()


def _default_datetime_utcnow_iso() -> str:
    """Production datetime.utcnow ISO format - used as default hook."""
    from datetime import datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")


def _default_count_model_parameters(model: LMModelProto) -> int:
    """Production count_model_parameters - used as default hook."""
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total


def _default_get_directory_size_bytes(path: Path) -> int:
    """Production get_directory_size_bytes - used as default hook."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _default_torch_cuda_get_rng_state_all() -> list[torch.Tensor]:
    """Production torch.cuda.get_rng_state_all - used as default hook.

    A thin adapter over torch; the checkpoint capture owns the
    availability check through the ``cuda_is_available`` hook.
    """
    return list(torch.cuda.get_rng_state_all())


def _default_torch_cuda_set_rng_state_all(states: list[torch.Tensor]) -> None:
    """Production torch.cuda.set_rng_state_all - used as default hook.

    Args:
        states: States previously returned by
            ``torch.cuda.get_rng_state_all``.
    """
    torch.cuda.set_rng_state_all(states)


def _default_random_getstate() -> tuple[TorchStateValue, ...]:
    """Production random.getstate - used as default hook."""
    random_mod = __import__("random")
    fn: Callable[[], tuple[TorchStateValue, ...]] = random_mod.getstate
    return fn()


def _default_random_setstate(state: tuple[TorchStateValue, ...]) -> None:
    """Production random.setstate - used as default hook.

    Args:
        state: State tuple previously returned by ``random.getstate``.
    """
    random_mod = __import__("random")
    fn: Callable[[tuple[TorchStateValue, ...]], None] = random_mod.setstate
    fn(state)


def _default_torch_device(device_str: str) -> torch.device:
    """Production torch.device - used as default hook."""
    return torch.device(device_str)
