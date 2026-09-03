"""Evaluation job processing."""

from __future__ import annotations

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_core.trainer_keys import eval_key
from typing_extensions import TypedDict

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.dataset import as_corpus_format
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.queue import EvalJobPayload
from model_trainer.core.infra.paths import model_eval_dir
from model_trainer.worker.job_utils import (
    materialize_run_artifacts,
    redis_client,
    setup_job_logging,
)
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
    settings = _test_hooks.load_settings()
    setup_job_logging(settings)

    log = get_logger(__name__)
    r = redis_client(settings)
    run_id = payload["run_id"]
    split = payload["split"]
    running: _EvalCacheModel = {"status": "running", "split": split}
    r.set(eval_key(run_id), dump_json_str(running))

    try:
        # This used to download unconditionally and then fail with
        # "destination already exists" whenever the directory was present,
        # so evaluating a run that any other job had already fetched -- cloze,
        # score, generate, continued training -- was an error rather than a
        # cache hit. Sharing the helper makes an existing directory the fast
        # path it should always have been.
        normalized = materialize_run_artifacts(settings, r, run_id, purpose="eval")
        manifest_path = normalized / "manifest.json"

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
            "corpus_format": as_corpus_format(manifest["corpus_format"], "corpus_format"),
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
            # Evaluation reports perplexity over held-out text, and a masked
            # prefix would silently change what that number covers. Masking is
            # a training-time intervention only.
            "loss_mask_prefix_separator": None,
            "finetuning_strategy": "full",
            "hub_model_id": None,
            "lora": None,
            "cartridge": None,
            "quantization": None,
            "gguf_export": None,
        }

        container = _test_hooks.service_container_from_settings(settings)
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
