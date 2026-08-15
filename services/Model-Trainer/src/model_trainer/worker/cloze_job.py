"""Cloze evaluation job processing.

Runs alongside the perplexity evaluation job rather than as a mode of it. The
two answer different questions and produce different result shapes: perplexity
reports how unsurprising a held-out split is, while this reports how often the
model picks the true completion over distractors. Folding them together would
mean one cache model carrying both shapes and a branch in the eval path, so
they stay separate.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger
from platform_core.trainer_keys import artifact_file_id_key, cloze_key
from typing_extensions import TypedDict

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.cloze import ClozeItem, decode_cloze_item
from model_trainer.core.contracts.queue import ClozeJobPayload
from model_trainer.core.infra.paths import models_dir
from model_trainer.core.services.model.cloze import score_cloze_items
from model_trainer.worker.job_utils import redis_client, setup_job_logging
from model_trainer.worker.manifest import as_device, as_model_family, load_manifest_from_text


class _ClozeCacheModel(TypedDict, total=False):
    """Redis-cached shape of a cloze job's lifecycle and outcome."""

    status: str
    total: int | None
    correct: int | None
    accuracy: float | None
    chance: float | None


def parse_items(raw: str) -> list[ClozeItem]:
    """Parse newline-delimited JSON into validated cloze items.

    Blank lines are skipped because a trailing newline is normal in a JSONL
    file; every other line must decode, so a malformed item fails the job
    rather than silently shrinking the evaluation set.

    Args:
        raw: Contents of the items file.

    Returns:
        Items in file order.

    Raises:
        AppError: With ``CLOZE_ITEMS_EMPTY`` when no items were present.
        JSONTypeError: When a line is not an object or violates the item
            contract.
    """
    items: list[ClozeItem] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        items.append(decode_cloze_item(narrow_json_to_dict(load_json_str(stripped))))

    if len(items) == 0:
        raise AppError(
            ModelTrainerErrorCode.CLOZE_ITEMS_EMPTY,
            "items file contained no cloze items",
            model_trainer_status_for(ModelTrainerErrorCode.CLOZE_ITEMS_EMPTY),
        )
    return items


def process_cloze_job(payload: ClozeJobPayload) -> None:
    """Process a cloze evaluation job.

    Args:
        payload: Job payload naming the run, the item set, and the token
            budget.

    Raises:
        AppError: When the run's artifact pointer or manifest is missing, or
            the item set is empty or unscoreable.
    """
    settings = _test_hooks.load_settings()
    setup_job_logging(settings)

    log = get_logger(__name__)
    r = redis_client(settings)
    run_id = payload["run_id"]
    request_id = payload["request_id"]

    running: _ClozeCacheModel = {"status": "running"}
    r.set(cloze_key(run_id, request_id), dump_json_str(running))

    try:
        file_id = r.get(artifact_file_id_key(run_id))
        if not isinstance(file_id, str) or file_id.strip() == "":
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "artifact pointer not found for cloze evaluation",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        api_url = settings["app"]["data_bank_api_url"]
        api_key = settings["app"]["data_bank_api_key"]
        store = _test_hooks.artifact_store_factory(api_url, api_key)
        models_root = models_dir(settings)
        normalized = models_root / run_id

        if not normalized.exists():
            out_root = store.download_artifact(
                file_id.strip(),
                dest_dir=models_root,
                request_id=run_id,
                expected_root=f"model-{run_id}",
            )
            out_root.rename(normalized)

        manifest_path = normalized / "manifest.json"
        if not manifest_path.exists():
            raise AppError(
                ModelTrainerErrorCode.MODEL_NOT_FOUND,
                f"manifest missing for run_id={run_id}",
                model_trainer_status_for(ModelTrainerErrorCode.MODEL_NOT_FOUND),
            )

        manifest = load_manifest_from_text(manifest_path.read_text(encoding="utf-8"))

        cache_dir: Path = Path(settings["app"]["artifacts_root"]) / "cloze"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fetcher = _test_hooks.corpus_fetcher_factory(api_url, api_key, cache_dir)
        items_path = fetcher.fetch(payload["items_file_id"])
        items = parse_items(items_path.read_text(encoding="utf-8"))

        tokenizer_id = manifest["tokenizer_id"]
        tok_handle = (
            _test_hooks.load_tokenizer_for_training(settings, tokenizer_id)
            if tokenizer_id is not None
            else None
        )
        container = _test_hooks.service_container_from_settings(settings)
        backend = container.model_registry.get(as_model_family(manifest["model_family"]))
        prepared = backend.load(str(normalized), settings, tokenizer=tok_handle)

        result = score_cloze_items(
            items=items,
            model=prepared.model,
            encoder=prepared.tok_for_dataset,
            device=as_device(manifest["device"]),
            max_seq_len=payload["max_seq_len"],
        )

        out: _ClozeCacheModel = {
            "status": "completed",
            "total": result["total"],
            "correct": result["correct"],
            "accuracy": result["accuracy"],
            "chance": result["chance"],
        }
    except Exception as e:
        out_failed: _ClozeCacheModel = {"status": "failed"}
        log.exception("Cloze failed run_id=%s request_id=%s error=%s", run_id, request_id, e)
        r.set(cloze_key(run_id, request_id), dump_json_str(out_failed))
        raise
    else:
        r.set(cloze_key(run_id, request_id), dump_json_str(out))
        log.info(
            "Cloze job completed run_id=%s request_id=%s accuracy=%.4f chance=%.4f",
            run_id,
            request_id,
            result["accuracy"],
            result["chance"],
        )


__all__ = [
    "parse_items",
    "process_cloze_job",
]
