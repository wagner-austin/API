"""Evaluation job processing."""

from __future__ import annotations

from pathlib import Path

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_core.trainer_keys import artifact_file_id_key, eval_key
from typing_extensions import TypedDict

from model_trainer.core.config.settings import load_settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.queue import EvalJobPayload
from model_trainer.core.infra.paths import model_eval_dir, models_dir
from model_trainer.core.services.container import ServiceContainer
from model_trainer.worker.job_utils import redis_client, setup_job_logging
from model_trainer.worker.manifest import (
    as_device,
    as_model_family,
    as_optimizer,
    as_precision,
    load_manifest_from_text,
)


class _EvalCacheModel(TypedDict, total=False):
    status: str
    split: str
    loss: float | None
    ppl: float | None
    artifact: str | None


def process_eval_job(payload: EvalJobPayload) -> None:
    """Process an evaluation job."""
    settings = load_settings()
    setup_job_logging(settings)

    log = get_logger(__name__)
    r = redis_client(settings)
    run_id = payload["run_id"]
    split = payload["split"]
    running: _EvalCacheModel = {"status": "running", "split": split}
    r.set(eval_key(run_id), dump_json_str(running))

    artifacts_root = settings["app"]["artifacts_root"]
    manifest_path = Path(artifacts_root) / "models" / run_id / "manifest.json"
    models_root = models_dir(settings)

    try:
        file_id = r.get(artifact_file_id_key(run_id))
        if not isinstance(file_id, str) or file_id.strip() == "":
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "artifact pointer not found for eval",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        from platform_ml import ArtifactStore

        api_url = settings["app"]["data_bank_api_url"]
        api_key = settings["app"]["data_bank_api_key"]
        store = ArtifactStore(api_url, api_key)
        expected_root = f"model-{run_id}"
        out_root = store.download_artifact(
            file_id.strip(), dest_dir=models_root, request_id=run_id, expected_root=expected_root
        )
        normalized = models_root / run_id
        if normalized.exists():
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                f"destination already exists: {normalized}",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )
        out_root.rename(normalized)

        if not manifest_path.exists():
            raise AppError(
                ModelTrainerErrorCode.MODEL_NOT_FOUND,
                f"manifest missing for run_id={run_id}",
                model_trainer_status_for(ModelTrainerErrorCode.MODEL_NOT_FOUND),
            )

        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = load_manifest_from_text(manifest_text)
        cfg: ModelTrainConfig = {
            "model_family": as_model_family(manifest["model_family"]),
            "model_size": manifest["model_size"],
            "max_seq_len": manifest["max_seq_len"],
            "num_epochs": manifest["epochs"],
            "batch_size": manifest["batch_size"],
            "learning_rate": manifest["learning_rate"],
            "tokenizer_id": manifest["tokenizer_id"],
            "corpus_path": manifest["corpus_path"],
            "holdout_fraction": manifest["holdout_fraction"],
            "seed": manifest["seed"],
            "pretrained_run_id": manifest["pretrained_run_id"],
            "freeze_embed": manifest["freeze_embed"],
            "gradient_clipping": manifest["gradient_clipping"],
            "optimizer": as_optimizer(manifest["optimizer"]),
            "device": as_device(manifest["device"]),
            "precision": as_precision(manifest["precision"]),
            "data_num_workers": 0,
            "data_pin_memory": False,
            "early_stopping_patience": manifest["early_stopping_patience"],
            "test_split_ratio": manifest["test_split_ratio"],
            "finetune_lr_cap": manifest["finetune_lr_cap"],
        }

        container = ServiceContainer.from_settings(settings)
        backend = container.model_registry.get(cfg["model_family"])
        if payload["path_override"] is not None:
            cfg["corpus_path"] = str(payload["path_override"]).strip()
        res = backend.evaluate(run_id=run_id, cfg=cfg, settings=settings)

        artifact_path = str(model_eval_dir(settings, run_id) / "metrics.json")
        out: _EvalCacheModel = {
            "status": "completed",
            "split": split,
            "loss": res["loss"],
            "ppl": res["perplexity"],
            "artifact": artifact_path,
        }
    except Exception as e:
        out_failed: _EvalCacheModel = {"status": "failed", "split": split}
        get_logger(__name__).exception("Eval failed run_id=%s error=%s", run_id, e)
        r.set(eval_key(run_id), dump_json_str(out_failed))
        raise
    else:
        r.set(eval_key(run_id), dump_json_str(out))
        log.info("Eval job completed run_id=%s split=%s", run_id, split)
