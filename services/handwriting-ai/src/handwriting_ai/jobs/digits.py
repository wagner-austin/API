from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, TypedDict

from platform_core.config import _optional_env_str
from platform_core.digits_metrics_events import (
    DigitsMetricsEventV1,
    encode_digits_metrics_event,
    make_artifact_event,
    make_batch_metrics_event,
    make_best_metrics_event,
    make_completed_metrics_event,
    make_config_event,
    make_epoch_metrics_event,
    make_upload_event,
)
from platform_core.job_events import JobDomain, default_events_channel
from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.logging import get_logger
from platform_core.queues import DIGITS_QUEUE as _DIGITS_QUEUE
from platform_workers.job_context import JobContext
from platform_workers.redis import RedisStrProto
from platform_workers.rq_harness import get_current_job

from handwriting_ai import _test_hooks
from handwriting_ai.config import Settings
from handwriting_ai.training.dataset import load_mnist_dataset
from handwriting_ai.training.metrics import BatchMetrics
from handwriting_ai.training.mnist_train import (
    TrainConfig,
    TrainingResult,
    train_with_config,
)
from handwriting_ai.training.progress import BatchProgressEmitter, BestEmitter, EpochEmitter

_DIGITS_DOMAIN: JobDomain = "digits"
DEFAULT_EVENTS_CHANNEL: Final[str] = default_events_channel(_DIGITS_DOMAIN)


class DigitsTrainJobV1(TypedDict):
    type: Literal["digits.train.v1"]
    request_id: str
    user_id: int
    model_id: str
    epochs: int
    batch_size: int
    lr: float
    seed: int
    augment: bool
    notes: str | None


def _get_env(name: str, default: str) -> str:
    v = _optional_env_str(name)
    return v if v is not None else default


def _load_settings() -> Settings:
    from handwriting_ai import _test_hooks

    return _test_hooks.load_settings()


def _build_cfg(payload: DigitsTrainJobV1) -> TrainConfig:
    s = _load_settings()
    data_root = (s["app"]["data_root"] / "mnist").resolve()
    out_dir = (s["app"]["artifacts_root"] / "digits" / "models").resolve()
    return {
        "data_root": data_root,
        "out_dir": out_dir,
        "model_id": payload["model_id"],
        "epochs": payload["epochs"],
        "batch_size": payload["batch_size"],
        "lr": float(payload["lr"]),
        "weight_decay": 1e-2,
        "seed": payload["seed"],
        "device": "cpu",
        "optim": "adamw",
        "scheduler": "cosine",
        "step_size": 10,
        "gamma": 0.5,
        "min_lr": 1e-5,
        "patience": 0,
        "min_delta": 5e-4,
        "threads": 0,
        "augment": bool(payload["augment"]),
        "aug_rotate": 10.0,
        "aug_translate": 0.1,
        "noise_prob": 0.15,
        "noise_salt_vs_pepper": 0.5,
        "dots_prob": 0.20,
        "dots_count": 3,
        "dots_size_px": 2,
        "blur_sigma": 0.0,
        "morph": "none",
        "morph_kernel_px": 3,
        "progress_every_epochs": 1,
        "progress_every_batches": 10,
        "calibrate": True,
        "calibration_samples": 500,
        "force_calibration": False,
        "memory_guard": True,
    }


def _run_training(cfg: TrainConfig) -> TrainingResult:
    from handwriting_ai import _test_hooks

    return _test_hooks.run_training(cfg)


def _run_training_impl(cfg: TrainConfig) -> TrainingResult:
    """Real implementation of training."""
    train_ds = load_mnist_dataset(cfg["data_root"], train=True)
    test_ds = load_mnist_dataset(cfg["data_root"], train=False)
    return train_with_config(cfg, (train_ds, test_ds))


def _summarize_training_exception(exc: BaseException) -> str:
    if isinstance(exc, RuntimeError) and str(exc) == "memory_pressure_guard_triggered":
        mg = _test_hooks.get_memory_guard_config()
        thr = float(mg["threshold_percent"])
        return (
            f"Training aborted due to sustained memory pressure (>= {thr:.1f}%). "
            "Reduce batch size or DataLoader workers and retry."
        )
    if isinstance(exc, RuntimeError) and "artifact upload failed" in str(exc):
        return "Artifact upload failed: upstream API error. See worker logs for details."
    name = exc.__class__.__name__
    msg = str(exc)
    preview = msg[:300]
    return f"{name}: {preview}" if preview else name


def _decode_int_field(payload: dict[str, JSONValue], key: str) -> int:
    raw = payload.get(key)
    if isinstance(raw, bool):
        raise JSONTypeError(f"{key}: bool not allowed")
    if isinstance(raw, str):
        return int(raw)
    if isinstance(raw, int):
        return raw
    raise JSONTypeError(f"{key}: invalid int")


def _decode_float_field(payload: dict[str, JSONValue], key: str) -> float:
    raw = payload.get(key)
    if isinstance(raw, bool):
        raise JSONTypeError(f"{key}: bool not allowed")
    if isinstance(raw, (str, int, float)):
        return float(raw)
    raise JSONTypeError(f"{key}: invalid float")


def _decode_payload(payload: dict[str, JSONValue]) -> DigitsTrainJobV1:
    typ = payload.get("type")
    if typ != "digits.train.v1":
        raise JSONTypeError("invalid job type")
    return {
        "type": "digits.train.v1",
        "request_id": str(payload.get("request_id")),
        "user_id": _decode_int_field(payload, "user_id"),
        "model_id": str(payload.get("model_id")),
        "epochs": _decode_int_field(payload, "epochs"),
        "batch_size": _decode_int_field(payload, "batch_size"),
        "lr": _decode_float_field(payload, "lr"),
        "seed": _decode_int_field(payload, "seed"),
        "augment": bool(payload.get("augment", False)),
        "notes": (str(payload["notes"]) if isinstance(payload.get("notes"), str) else None),
    }


def _build_config_event(p: DigitsTrainJobV1, cfg: TrainConfig, queue: str) -> DigitsMetricsEventV1:
    _limits = _test_hooks.detect_resource_limits()
    _mem_mb = (
        int(_limits["memory_bytes"] // (1024 * 1024))
        if isinstance(_limits["memory_bytes"], int)
        else None
    )
    return make_config_event(
        job_id=p["request_id"],
        user_id=p["user_id"],
        model_id=p["model_id"],
        total_epochs=p["epochs"],
        queue=queue,
        cpu_cores=_limits["cpu_cores"],
        optimal_threads=_limits["optimal_threads"],
        memory_mb=_mem_mb,
        optimal_workers=_limits["optimal_workers"],
        max_batch_size=_limits.get("max_batch_size"),
        device=cfg["device"],
        batch_size=cfg["batch_size"],
        learning_rate=p["lr"],
        augment=cfg["augment"],
        aug_rotate=cfg["aug_rotate"],
        aug_translate=cfg["aug_translate"],
        noise_prob=cfg["noise_prob"],
        dots_prob=cfg["dots_prob"],
    )


class _JobEmitter(BatchProgressEmitter, EpochEmitter, BestEmitter):
    def __init__(self, job_ctx: JobContext, p: DigitsTrainJobV1):
        self._job_ctx = job_ctx
        self._p = p
        self.last_progress = 0

    def _calculate_progress(
        self, epoch: int, total_epochs: int, batch: int, total_batches: int
    ) -> int:
        if total_epochs <= 0:
            return 0
        epoch_progress = (epoch - 1) / total_epochs
        batch_progress = batch / (total_batches * total_epochs) if total_batches > 0 else 0
        progress = min(99, int((epoch_progress + batch_progress) * 100))
        self.last_progress = progress
        return progress

    def emit_batch(self, metrics: BatchMetrics) -> None:
        progress = self._calculate_progress(
            metrics["epoch"],
            metrics["total_epochs"],
            metrics["batch"],
            metrics["total_batches"],
        )
        payload: JSONValue = encode_digits_metrics_event(
            make_batch_metrics_event(
                job_id=self._p["request_id"],
                user_id=self._p["user_id"],
                model_id=self._p["model_id"],
                **metrics,
            )
        )
        self._job_ctx.publish_progress(progress, payload=payload)

    def emit_epoch(
        self, epoch: int, total_epochs: int, train_loss: float, val_acc: float, time_s: float
    ) -> None:
        progress = self._calculate_progress(epoch, total_epochs, 0, 0)
        payload: JSONValue = encode_digits_metrics_event(
            make_epoch_metrics_event(
                job_id=self._p["request_id"],
                user_id=self._p["user_id"],
                model_id=self._p["model_id"],
                epoch=epoch,
                total_epochs=total_epochs,
                train_loss=train_loss,
                val_acc=val_acc,
                time_s=time_s,
            )
        )
        self._job_ctx.publish_progress(progress, payload=payload)

    def emit_best(self, epoch: int, val_acc: float) -> None:
        payload: JSONValue = encode_digits_metrics_event(
            make_best_metrics_event(
                job_id=self._p["request_id"],
                user_id=self._p["user_id"],
                model_id=self._p["model_id"],
                epoch=epoch,
                val_acc=val_acc,
            )
        )
        self._job_ctx.publish_progress(self.last_progress, payload=payload)


def _decode_and_process_train_job(payload: dict[str, JSONValue]) -> None:
    p = _decode_payload(payload)
    current_job = get_current_job()
    queue_name = current_job.origin if current_job and current_job.origin else _DIGITS_QUEUE

    redis_url = _optional_env_str("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL not configured")
    redis_client: RedisStrProto = _test_hooks.redis_factory(redis_url)

    job_ctx: JobContext | None = None
    training_progress_module: _test_hooks.TrainingProgressModuleProtocol | None = None
    try:
        job_ctx = _test_hooks.make_job_context(
            redis=redis_client,
            domain=_DIGITS_DOMAIN,
            events_channel=default_events_channel(_DIGITS_DOMAIN),
            job_id=p["request_id"],
            user_id=p["user_id"],
            queue_name=queue_name,
        )
        if job_ctx is None:
            raise RuntimeError("JobContext could not be created")

        training_progress = _test_hooks.get_training_progress_module()

        if training_progress is None:
            raise RuntimeError("training progress module not available")
        training_progress_module = training_progress

        emitter = _JobEmitter(job_ctx, p)

        job_ctx.publish_started()

        cfg = _build_cfg(p)
        config_payload: JSONValue = encode_digits_metrics_event(
            _build_config_event(p, cfg, queue_name)
        )
        job_ctx.publish_progress(0, payload=config_payload)

        training_progress_module.set_batch_emitter(emitter)
        training_progress_module.set_epoch_emitter(emitter)
        training_progress_module.set_best_emitter(emitter)

        result = _run_training(cfg)

        model_dir = cfg["out_dir"] / result["model_id"]
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pt"

        import torch

        torch.save(result["state_dict"], model_path.as_posix())

        artifact_payload: JSONValue = encode_digits_metrics_event(
            make_artifact_event(
                job_id=p["request_id"],
                user_id=p["user_id"],
                model_id=result["model_id"],
                path=str(model_dir),
            )
        )
        job_ctx.publish_progress(95, payload=artifact_payload)

        _upload_and_finalize(job_ctx, p, result, model_dir)

    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        get_logger("handwriting_ai").error(
            "training_failed error=%s", _summarize_training_exception(exc)
        )
        if job_ctx:
            job_ctx.publish_failed("system", _summarize_training_exception(exc))
        raise
    finally:
        if training_progress_module is not None:
            training_progress_module.set_batch_emitter(None)
            training_progress_module.set_epoch_emitter(None)
            training_progress_module.set_best_emitter(None)
        redis_client.close()


def _upload_and_finalize(
    job_ctx: JobContext, p: DigitsTrainJobV1, result: TrainingResult, model_dir: Path
) -> None:
    from datetime import UTC, datetime

    from platform_core.json_utils import dump_json_str
    from platform_ml.manifest import ModelManifestV2

    from handwriting_ai.config import MNIST_N_CLASSES
    from handwriting_ai.preprocess import preprocess_signature

    api_url = _get_env("APP__DATA_BANK_API_URL", "")
    api_key = _get_env("APP__DATA_BANK_API_KEY", "")
    if not api_url or not api_key:
        raise RuntimeError("artifact upload failed: missing data bank API credentials")

    created_at = datetime.now(UTC).isoformat()
    meta = result["metadata"]
    v2_manifest: ModelManifestV2 = {
        "schema_version": "v2.0",
        "model_type": "resnet18",
        "model_id": result["model_id"],
        "created_at": created_at,
        "arch": "resnet18",
        "n_classes": MNIST_N_CLASSES,
        "val_acc": float(result["val_acc"]),
        "preprocess_hash": preprocess_signature(),
        "file_id": "",
        "file_size": 0,
        "file_sha256": "",
        "training": {
            "run_id": meta["run_id"],
            "epochs": meta["epochs"],
            "batch_size": meta["batch_size"],
            "learning_rate": meta["lr"],
            "seed": meta["seed"],
            "device": meta["device"],
            "optimizer": meta["optim"],
            "scheduler": meta["scheduler"],
            "augment": meta["augment"],
        },
    }

    store = _test_hooks.artifact_store_factory(api_url, api_key)
    resp = store.upload_artifact(
        model_dir, artifact_name=result["model_id"], request_id=p["request_id"]
    )

    v2_manifest.update(
        {"file_id": resp["file_id"], "file_size": resp["size"], "file_sha256": resp["sha256"]}
    )
    (model_dir / "manifest.json").write_text(
        dump_json_str(v2_manifest, compact=False), encoding="utf-8"
    )

    upload_payload: JSONValue = encode_digits_metrics_event(
        make_upload_event(
            job_id=p["request_id"],
            user_id=p["user_id"],
            model_id=result["model_id"],
            status=200,
            model_bytes=(model_dir / "model.pt").stat().st_size,
            manifest_bytes=(model_dir / "manifest.json").stat().st_size,
            file_id=resp["file_id"],
            file_sha256=resp["sha256"],
        )
    )
    job_ctx.publish_progress(98, payload=upload_payload)

    completed_payload: JSONValue = encode_digits_metrics_event(
        make_completed_metrics_event(
            job_id=p["request_id"],
            user_id=p["user_id"],
            model_id=result["model_id"],
            val_acc=result["val_acc"],
        )
    )
    job_ctx.publish_progress(99, payload=completed_payload)

    job_ctx.publish_completed(resp["file_id"], resp["size"])


def process_train_job(payload: Mapping[str, JSONValue]) -> None:
    _decode_and_process_train_job(dict(payload))
