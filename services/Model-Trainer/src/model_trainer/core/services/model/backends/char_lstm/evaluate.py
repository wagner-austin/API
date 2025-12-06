from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Literal

import torch
from platform_core.json_utils import dump_json_str

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.dataset import DatasetBuilder, DatasetConfig
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.infra.paths import model_dir as _model_dir
from model_trainer.core.infra.paths import model_eval_dir
from model_trainer.core.services.training.dataset_builder import CausalLMDataset

from ..gpt2._dl import DataLoader
from .io import load_prepared_char_lstm_from_handle as _load_prepared_from_handle
from .io import token_ids


def _get_autocast_context(
    precision: Literal["fp32", "fp16", "bf16"], device_type: str
) -> AbstractContextManager[None]:
    """Get autocast context manager based on precision and device type.

    Args:
        precision: The precision to use.
        device_type: The device type ("cpu" or "cuda").

    Returns:
        A context manager for autocast, or nullcontext for fp32/cpu.
    """
    if precision == "fp32":
        return nullcontext()
    if device_type != "cuda":
        return nullcontext()
    torch_amp = __import__("torch.amp", fromlist=["autocast"])
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    ctx: AbstractContextManager[None] = torch_amp.autocast(device_type="cuda", dtype=dtype)
    return ctx


class EvalResult:
    loss: float
    perplexity: float

    def __init__(self: EvalResult, *, loss: float, perplexity: float) -> None:
        self.loss = loss
        self.perplexity = perplexity


def _load_tokenizer_for_dataset(tokenizer_path: str) -> TokenizerHandle:
    # Load a TokenizerHandle via char backend explicitly
    from model_trainer.core.services.tokenizer.char_backend import CharBackend

    return CharBackend().load(tokenizer_path)


def evaluate_char_lstm(
    *, run_id: str, cfg: ModelTrainConfig, settings: Settings, dataset_builder: DatasetBuilder
) -> EvalResult:
    artifacts_root = settings["app"]["artifacts_root"]
    tokenizer_path = str(
        Path(artifacts_root) / "tokenizers" / cfg["tokenizer_id"] / "tokenizer.json"
    )
    handle = _load_tokenizer_for_dataset(tokenizer_path)
    eos_id, pad_id, _ = token_ids(handle)

    ds_cfg = DatasetConfig(
        corpus_path=cfg["corpus_path"],
        holdout_fraction=cfg["holdout_fraction"],
        test_split_ratio=cfg["test_split_ratio"],
    )
    _, val_files, _ = dataset_builder.split(ds_cfg)
    prepared_for_tok = _load_prepared_from_handle(str(_model_dir(settings, run_id)), handle)
    dataset = CausalLMDataset(
        files=val_files,
        tokenizer=prepared_for_tok.tok_for_dataset,
        max_len=cfg["max_seq_len"],
        eos_id=eos_id,
        pad_id=pad_id,
    )
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["data_num_workers"],
        pin_memory=cfg["data_pin_memory"],
    )

    model_dir = str(_model_dir(settings, run_id))
    prepared = _load_prepared_from_handle(model_dir, handle)
    model = prepared.model
    model.eval()
    device: str = cfg["device"]
    model.to(device)

    # Use same precision as training for consistent metrics
    precision = cfg["precision"]
    autocast_ctx = _get_autocast_context(precision, device)

    total_loss = 0.0
    total_count = 0
    eval_dir = model_eval_dir(settings, run_id)
    eval_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(device)
            with autocast_ctx:
                outputs = model.forward(input_ids=inputs, labels=inputs)
            loss_t = outputs.loss
            batch_count: int = int(inputs.size(0))
            total_loss += float(loss_t.item()) * float(batch_count)
            total_count += batch_count
    avg_loss = total_loss / max(1, total_count)
    ppl = float(math.exp(avg_loss)) if avg_loss < 20 else float("inf")
    metrics = {"loss": avg_loss, "perplexity": ppl}
    (eval_dir / "metrics.json").write_text(dump_json_str(metrics), encoding="utf-8")
    return EvalResult(loss=avg_loss, perplexity=ppl)
