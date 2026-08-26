from __future__ import annotations

from typing import Literal

from platform_core.comparability import RunFingerprint
from typing_extensions import TypedDict


class EvalCache(TypedDict):
    status: Literal["queued", "running", "completed", "failed"]
    split: str
    loss: float | None
    ppl: float | None
    artifact: str | None


class TrainingManifestVersions(TypedDict):
    torch: str
    transformers: str
    tokenizers: str
    datasets: str


class TrainingManifestSystem(TypedDict):
    """Where the run happened, for the axes nothing compares on.

    The card is NOT here. It used to be, as ``gpu_name``, read from the same
    ``cuda_device_name`` hook the run fingerprint reads -- one value computed
    twice, stored under two names, in two shapes, and never read back by
    anything. Meanwhile the fingerprint, which IS read, could not see the
    training path at all. The card now lives in
    :attr:`TrainingManifest.fingerprint` with the driver and the image beside
    it, which is where a comparison looks for it.
    """

    cpu_count: int
    platform: str
    platform_release: str
    machine: str


class TrainingManifestTiming(TypedDict):
    """Timing information for training run.

    Attributes:
        training_duration_sec: Total training time in seconds.
        started_at: ISO 8601 timestamp when training began.
        completed_at: ISO 8601 timestamp when training finished.
    """

    training_duration_sec: float
    started_at: str
    completed_at: str


class TrainingManifestPerformance(TypedDict):
    """Performance metrics from training run.

    Attributes:
        peak_gpu_memory_mb: Maximum GPU memory used in megabytes (None if CPU).
        avg_samples_per_sec: Average throughput during training.
        total_tokens_processed: Total tokens seen during training.
    """

    peak_gpu_memory_mb: float | None
    avg_samples_per_sec: float
    total_tokens_processed: int


class TrainingManifestModelInfo(TypedDict):
    """Model metadata captured after training.

    Attributes:
        param_count: Total trainable parameters in the model.
        model_size_mb: Size of saved model on disk in megabytes.
        vocab_size: Tokenizer vocabulary size.
    """

    param_count: int
    model_size_mb: float
    vocab_size: int


class GgufExportManifest(TypedDict):
    """Manifest section for GGUF export results.

    Attributes:
        output_type: The output precision format used for GGUF export.
        output_filename: Name of the generated GGUF file.
        output_size_bytes: Size of the GGUF file in bytes.
    """

    output_type: str
    output_filename: str
    output_size_bytes: int


class TrainingManifest(TypedDict):
    """Training manifest with all configuration and results.

    For hf_lm models, tokenizer_id may be None because the HF tokenizer
    is loaded from hub_model_id.
    """

    run_id: str
    model_family: str
    model_size: str
    epochs: int
    batch_size: int
    max_seq_len: int
    steps: int
    loss: float
    learning_rate: float
    tokenizer_id: str | None  # None for hf_lm (uses HF tokenizer from hub_model_id)
    corpus_path: str
    holdout_fraction: float
    optimizer: str
    freeze_embed: bool
    gradient_clipping: float
    seed: int
    pretrained_run_id: str | None
    versions: TrainingManifestVersions
    system: TrainingManifestSystem
    # The configuration this run's numbers were produced under: image digest,
    # card, driver and determinism posture. The SAME type the scoring path
    # records, so `compare_configurations` reads a training run and a scoring
    # run without knowing which is which.
    #
    # It replaces two fields that each answered part of this and disagreed
    # about the shape: `system.gpu_name` (the card, from the same hook, never
    # read by anything) and a top-level `determinism`. Neither carried the
    # image digest or the driver, so the training path could not answer the
    # question the scoring path was already answering -- and the image digest
    # is now the axis that matters most, because these runs happen inside
    # abl.sif.
    #
    # None in manifests written before the field existed, which the decoder
    # reads as "not recorded" -- the treatment git_commit already gets, and
    # for the same reason: refusing to decode an old manifest would break
    # LOADING a trained model, and loading is not comparing. A run that
    # writes one always writes it whole; there is no half-populated
    # fingerprint, because an axis silently absent compares equal to another
    # run missing the same axis.
    fingerprint: RunFingerprint | None
    git_commit: str | None
    device: str
    precision: str
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float
    loss_mask_prefix_separator: str | None
    test_loss: float | None
    test_perplexity: float | None
    best_val_loss: float | None
    early_stopped: bool
    # None for a run that trained start to finish in one execution; the
    # epoch index a resumed execution continued from otherwise.
    resumed_from_epoch: int | None
    # New fields for feature roadmap
    timing: TrainingManifestTiming
    performance: TrainingManifestPerformance
    model_info: TrainingManifestModelInfo
    gguf_export: GgufExportManifest | None


class TrainingManifestConfig(TypedDict):
    """Configuration section of training manifest.

    For hf_lm models, tokenizer_id may be None because the HF tokenizer
    is loaded from hub_model_id.
    """

    model_family: str
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    tokenizer_id: str | None  # None for hf_lm (uses HF tokenizer from hub_model_id)
    corpus_path: str
    holdout_fraction: float
    seed: int
    pretrained_run_id: str | None
    freeze_embed: bool
    gradient_clipping: float
    optimizer: str
    device: str
    precision: str
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float
    loss_mask_prefix_separator: str | None


class TrainingManifestFull(TypedDict):
    """Full training manifest with embedded config block.

    For hf_lm models, tokenizer_id may be None because the HF tokenizer
    is loaded from hub_model_id.
    """

    run_id: str
    model_family: str
    model_size: str
    epochs: int
    batch_size: int
    max_seq_len: int
    steps: int
    loss: float
    learning_rate: float
    tokenizer_id: str | None  # None for hf_lm (uses HF tokenizer from hub_model_id)
    corpus_path: str
    holdout_fraction: float
    optimizer: str
    freeze_embed: bool
    gradient_clipping: float
    seed: int
    pretrained_run_id: str | None
    versions: TrainingManifestVersions
    system: TrainingManifestSystem
    fingerprint: RunFingerprint | None
    git_commit: str | None
    config: TrainingManifestConfig
    device: str
    precision: str
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float
    loss_mask_prefix_separator: str | None
    test_loss: float | None
    test_perplexity: float | None
    best_val_loss: float | None
    early_stopped: bool
    # None for a run that trained start to finish in one execution; the
    # epoch index a resumed execution continued from otherwise.
    resumed_from_epoch: int | None
    # New fields for feature roadmap
    timing: TrainingManifestTiming
    performance: TrainingManifestPerformance
    model_info: TrainingManifestModelInfo
    gguf_export: GgufExportManifest | None
