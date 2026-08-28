"""Scoring an untrained model on a cloze item set.

Every cloze result this service produces is only interpretable against an
unexposed-model floor: on real prose a model that has never seen the corpus
still beats chance, because the surrounding sentence constrains the answer. So
an arm's absolute accuracy means nothing without a baseline to subtract, and
until this job existed there was no way to measure one -- ``process_cloze_job``
resolves weights from a completed run's artifacts, and an untrained model has
none.

A baseline is deliberately not modelled as a run. It has no corpus, no
manifest, no training and no artifact, and giving it a run id would put a row
in the run ledger indistinguishable from something that actually trained. It
gets its own key namespace instead, identified by what it is: which model, on
which items.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.comparability import encode_run_fingerprint
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.logging import get_logger
from platform_core.trainer_keys import baseline_cloze_key

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.cloze import encode_cloze_item_outcome
from model_trainer.core.contracts.queue import BaselineClozeJobPayload
from model_trainer.core.run_fingerprint import capture_run_fingerprint, describe_run_fingerprint
from model_trainer.core.services.model.backends.hf_lm.io import load_prepared_hf_lm_from_hub
from model_trainer.core.services.model.cloze import score_cloze_items
from model_trainer.worker.cloze_job import ClozeCacheModel, parse_items
from model_trainer.worker.job_utils import redis_client, setup_job_logging


def process_baseline_cloze_job(payload: BaselineClozeJobPayload) -> None:
    """Score an untrained hub model on an item set and record the result.

    Args:
        payload: Job payload naming the model, the item set, the token budget
            and the device.

    Raises:
        AppError: When the item set is empty or an item cannot be scored.
    """
    settings = _test_hooks.load_settings()
    setup_job_logging(settings)

    # Before any CUDA work: CUBLAS_WORKSPACE_CONFIG is read once when the
    # cuBLAS handle is created, so a later call is accepted in silence and has
    # no effect. This floor is the number every arm accuracy is reported as
    # lift over, and it was measured unpinned.
    #
    # remove_split_k=True: a floor every arm is reported as lift over is the
    # one number that must not depend on which card answered the queue.
    determinism = _test_hooks.apply_determinism_hook(remove_split_k=True)

    log = get_logger(__name__)
    r = redis_client(settings)
    hub_model_id = payload["hub_model_id"]
    items_file_id = payload["items_file_id"]
    key = baseline_cloze_key(hub_model_id, items_file_id)

    running: ClozeCacheModel = {"status": "running"}
    r.set(key, dump_json_str(running))

    try:
        api_url = settings["app"]["data_bank_api_url"]
        api_key = settings["app"]["data_bank_api_key"]
        cache_dir: Path = Path(settings["app"]["artifacts_root"]) / "cloze"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fetcher = _test_hooks.corpus_fetcher_factory(api_url, api_key, cache_dir)
        items_path = fetcher.fetch(items_file_id)
        items = parse_items(items_path.read_text(encoding="utf-8"))

        prepared = load_prepared_hf_lm_from_hub(hub_model_id)

        fingerprint = capture_run_fingerprint(payload["device"], determinism)

        result = score_cloze_items(
            items=items,
            model=prepared.model,
            encoder=prepared.tok_for_dataset,
            device=payload["device"],
            max_seq_len=payload["max_seq_len"],
        )

        encoded_outcomes: list[JSONValue] = [
            encode_cloze_item_outcome(outcome) for outcome in result["outcomes"]
        ]
        out: ClozeCacheModel = {
            "status": "completed",
            "total": result["total"],
            "correct": result["correct"],
            "accuracy": result["accuracy"],
            "chance": result["chance"],
            "outcomes": encoded_outcomes,
            "fingerprint": encode_run_fingerprint(fingerprint),
        }
    except Exception as e:
        out_failed: ClozeCacheModel = {"status": "failed"}
        log.exception(
            "Baseline cloze failed hub_model_id=%s items_file_id=%s error=%s",
            hub_model_id,
            items_file_id,
            e,
        )
        r.set(key, dump_json_str(out_failed))
        raise
    else:
        r.set(key, dump_json_str(out))
        log.info(
            "Baseline cloze completed hub_model_id=%s items_file_id=%s "
            "accuracy=%.4f chance=%.4f %s",
            hub_model_id,
            items_file_id,
            result["accuracy"],
            result["chance"],
            describe_run_fingerprint(fingerprint),
        )


__all__ = ["process_baseline_cloze_job"]
