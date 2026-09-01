"""HuggingFace LM model evaluation.

Evaluates trained models on validation set.

Uses hooks from _test_hooks for dependency injection.
Production sets hooks to real implementations at startup.
Tests set hooks to fakes for isolation.
"""

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

from ._test_hooks import (
    CausalLMDatasetProto,
    CreateCausalDatasetFn,
    CreateDataLoaderFn,
    DataLoaderProto,
    EvalDirFn,
    Hooks,
    ModelDirFn,
    PreparedModelLoader,
    TokenizerLoader,
)


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
    """Evaluation result with loss and perplexity metrics."""

    loss: float
    perplexity: float

    def __init__(self, *, loss: float, perplexity: float) -> None:
        """Initialize evaluation result.

        Args:
            loss: Average loss value.
            perplexity: Perplexity metric.
        """
        self.loss = loss
        self.perplexity = perplexity


def evaluate_hf_lm(
    *,
    run_id: str,
    cfg: ModelTrainConfig,
    settings: Settings,
    dataset_builder: DatasetBuilder,
) -> EvalResult:
    """Evaluate a trained HuggingFace LM model on the validation set.

    Args:
        run_id: Identifier for the training run.
        cfg: Training configuration.
        settings: Application settings containing artifacts_root path.
        dataset_builder: Builder for creating train/val dataset splits.

    Returns:
        EvalResult containing average loss and perplexity.

    Raises:
        FileNotFoundError: If model files do not exist.
    """
    load_tok: TokenizerLoader = Hooks.load_tokenizer
    load_model: PreparedModelLoader = Hooks.load_prepared_model
    model_dir_fn: ModelDirFn = Hooks.get_model_dir
    eval_dir_fn: EvalDirFn = Hooks.get_eval_dir
    create_dataset: CreateCausalDatasetFn = Hooks.create_causal_dataset
    create_loader: CreateDataLoaderFn = Hooks.create_dataloader

    # Load tokenizer for dataset (optional for hf_lm - uses HF tokenizer from hub)
    tokenizer_id = cfg["tokenizer_id"]
    tokenizer_handle: TokenizerHandle | None
    if tokenizer_id is not None:
        artifacts_root = settings["app"]["artifacts_root"]
        tokenizer_dir = Path(artifacts_root) / "tokenizers" / tokenizer_id
        tokenizer_handle = load_tok(str(tokenizer_dir))
    else:
        # HF LM uses tokenizer from hub_model_id, not a custom tokenizer
        tokenizer_handle = None

    # Load prepared model
    model_path = str(model_dir_fn(settings, run_id))
    prepared = load_model(model_path, tokenizer_handle)

    # Get token IDs
    eos_id = prepared.eos_id
    pad_id = prepared.pad_id

    # Build validation dataset
    ds_cfg = DatasetConfig(
        corpus_path=cfg["corpus_path"],
        corpus_format=cfg["corpus_format"],
        holdout_fraction=cfg["holdout_fraction"],
        test_split_ratio=cfg["test_split_ratio"],
    )
    split = dataset_builder.split(ds_cfg)
    dataset: CausalLMDatasetProto = create_dataset(
        lines=split["validation"],
        tokenizer=prepared.tok_for_dataset,
        max_len=cfg["max_seq_len"],
        eos_id=eos_id,
        pad_id=pad_id,
    )

    # Create dataloader
    dataloader: DataLoaderProto = create_loader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["data_num_workers"],
        pin_memory=cfg["data_pin_memory"],
    )

    # Prepare model for evaluation
    model = prepared.model
    model.eval()
    device: str = cfg["device"]
    model.to(device)

    # Use same precision as training
    precision = cfg["precision"]
    autocast_ctx = _get_autocast_context(precision, device)

    # Evaluate
    total_loss = 0.0
    total_count = 0
    eval_dir = eval_dir_fn(settings, run_id)
    eval_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            with autocast_ctx:
                outputs = model.forward(input_ids=inputs, labels=labels)
            loss_t = outputs.loss
            batch_count: int = int(inputs.size(0))
            total_loss += float(loss_t.item()) * float(batch_count)
            total_count += batch_count

    avg_loss = total_loss / max(1, total_count)
    ppl = float(math.exp(avg_loss)) if avg_loss < 20 else float("inf")

    # Save metrics
    metrics = {"loss": avg_loss, "perplexity": ppl}
    (eval_dir / "metrics.json").write_text(dump_json_str(metrics), encoding="utf-8")

    return EvalResult(loss=avg_loss, perplexity=ppl)


__all__ = [
    "EvalResult",
    "evaluate_hf_lm",
]
