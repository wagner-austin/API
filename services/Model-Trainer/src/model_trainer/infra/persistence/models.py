from __future__ import annotations

from typing import Literal

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
    cpu_count: int
    platform: str
    platform_release: str
    machine: str


class TrainingManifest(TypedDict):
    """Training manifest with all configuration and results."""

    run_id: str
    model_family: str
    model_size: str
    epochs: int
    batch_size: int
    max_seq_len: int
    steps: int
    loss: float
    learning_rate: float
    tokenizer_id: str
    corpus_path: str
    holdout_fraction: float
    optimizer: str
    freeze_embed: bool
    gradient_clipping: float
    seed: int
    pretrained_run_id: str | None
    versions: TrainingManifestVersions
    system: TrainingManifestSystem
    git_commit: str | None
    device: str
    precision: str
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float
    test_loss: float | None
    test_perplexity: float | None
    best_val_loss: float | None
    early_stopped: bool


class TrainingManifestConfig(TypedDict):
    """Configuration section of training manifest."""

    model_family: str
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    tokenizer_id: str
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


class TrainingManifestFull(TypedDict):
    """Full training manifest with embedded config block."""

    run_id: str
    model_family: str
    model_size: str
    epochs: int
    batch_size: int
    max_seq_len: int
    steps: int
    loss: float
    learning_rate: float
    tokenizer_id: str
    corpus_path: str
    holdout_fraction: float
    optimizer: str
    freeze_embed: bool
    gradient_clipping: float
    seed: int
    pretrained_run_id: str | None
    versions: TrainingManifestVersions
    system: TrainingManifestSystem
    git_commit: str | None
    config: TrainingManifestConfig
    device: str
    precision: str
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float
    test_loss: float | None
    test_perplexity: float | None
    best_val_loss: float | None
    early_stopped: bool
