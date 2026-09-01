"""Turning a training manifest into the bytes that go on disk.

THE MANIFEST HAD A DECODER AND NO ENCODER. That asymmetry was harmless only
while every field happened to be JSON-native, and it stopped being harmless
the moment one was not: ``DeterminismRecord.settings`` is a tuple of
``(name, value)`` PAIRS in memory -- sorted at construction so two postures
compare equal -- and only :func:`encode_determinism_record` turns it into an
object. Dumping the manifest straight to JSON wrote

    "settings": [["cudnn_deterministic", "true"], ...]

which the decoder refuses with "Field 'settings' must be an object, got
list". The manifest was unreadable by the code that had just written it.

That defect shipped in 36a51b50 and no test saw it, because the field was
optional and every test left it None -- so the value that would have broken
was never serialised. Making the posture REQUIRED is what surfaced it, which
is the case for "required" in one sentence.

Split from ``manifest`` when that module passed the 600-line ceiling. The
decoder stays there; this is the other direction, and the two are meant to be
read as a pair -- a field added to one without the other is the bug above.
"""

from __future__ import annotations

from platform_core.comparability import encode_run_fingerprint
from platform_core.json_utils import JSONObject

from model_trainer.infra.persistence.models import (
    GgufExportManifest,
    TrainingManifestConfig,
    TrainingManifestFull,
    TrainingManifestModelInfo,
    TrainingManifestPerformance,
    TrainingManifestSystem,
    TrainingManifestTiming,
    TrainingManifestVersions,
)


def _encode_versions(versions: TrainingManifestVersions) -> JSONObject:
    """Encode the library versions block.

    Args:
        versions: The block to encode.

    Returns:
        A JSON object carrying every version string.
    """
    return {
        "torch": versions["torch"],
        "transformers": versions["transformers"],
        "tokenizers": versions["tokenizers"],
        "datasets": versions["datasets"],
    }


def _encode_system(system: TrainingManifestSystem) -> JSONObject:
    """Encode the host block.

    The card is deliberately absent: it belongs to the fingerprint, and a
    copy here is the fork this consolidation removed.

    Args:
        system: The block to encode.

    Returns:
        A JSON object carrying the host facts.
    """
    return {
        "cpu_count": system["cpu_count"],
        "platform": system["platform"],
        "platform_release": system["platform_release"],
        "machine": system["machine"],
    }


def _encode_config(config: TrainingManifestConfig) -> JSONObject:
    """Encode the embedded configuration block.

    Args:
        config: The block to encode.

    Returns:
        A JSON object carrying every configured value.
    """
    return {
        "model_family": config["model_family"],
        "model_size": config["model_size"],
        "max_seq_len": config["max_seq_len"],
        "num_epochs": config["num_epochs"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "tokenizer_id": config["tokenizer_id"],
        "corpus_path": config["corpus_path"],
        "corpus_format": config["corpus_format"],
        "holdout_fraction": config["holdout_fraction"],
        "seed": config["seed"],
        "pretrained_run_id": config["pretrained_run_id"],
        "freeze_embed": config["freeze_embed"],
        "gradient_clipping": config["gradient_clipping"],
        "optimizer": config["optimizer"],
        "device": config["device"],
        "precision": config["precision"],
        "early_stopping_patience": config["early_stopping_patience"],
        "test_split_ratio": config["test_split_ratio"],
        "finetune_lr_cap": config["finetune_lr_cap"],
        "loss_mask_prefix_separator": config["loss_mask_prefix_separator"],
    }


def _encode_timing(timing: TrainingManifestTiming) -> JSONObject:
    """Encode the timing block.

    Args:
        timing: The block to encode.

    Returns:
        A JSON object carrying the duration and both timestamps.
    """
    return {
        "training_duration_sec": timing["training_duration_sec"],
        "started_at": timing["started_at"],
        "completed_at": timing["completed_at"],
    }


def _encode_performance(performance: TrainingManifestPerformance) -> JSONObject:
    """Encode the performance block.

    Args:
        performance: The block to encode.

    Returns:
        A JSON object carrying the throughput figures.
    """
    return {
        "peak_gpu_memory_mb": performance["peak_gpu_memory_mb"],
        "avg_samples_per_sec": performance["avg_samples_per_sec"],
        "total_tokens_processed": performance["total_tokens_processed"],
    }


def _encode_model_info(model_info: TrainingManifestModelInfo) -> JSONObject:
    """Encode the model-metadata block.

    Args:
        model_info: The block to encode.

    Returns:
        A JSON object carrying the parameter count, size and vocabulary.
    """
    return {
        "param_count": model_info["param_count"],
        "model_size_mb": model_info["model_size_mb"],
        "vocab_size": model_info["vocab_size"],
    }


def _encode_gguf_export(gguf_export: GgufExportManifest | None) -> JSONObject | None:
    """Encode the GGUF export block, which most runs do not have.

    Args:
        gguf_export: The block to encode, or None when the run exported none.

    Returns:
        A JSON object, or None.
    """
    if gguf_export is None:
        return None
    return {
        "output_type": gguf_export["output_type"],
        "output_filename": gguf_export["output_filename"],
        "output_size_bytes": gguf_export["output_size_bytes"],
    }


def encode_training_manifest_full(full: TrainingManifestFull) -> JSONObject:
    """Encode a full manifest for the file on disk.

    Written out field by field rather than as ``{**full}``. A spread passes
    any future non-JSON-native member straight through and fails only when
    something reads the file back -- which is exactly how the determinism
    posture reached disk in a shape its own decoder rejects. Naming every
    field means adding one forces a decision here.

    Args:
        full: The manifest to encode.

    Returns:
        A JSON object. The fingerprint goes through the SAME encoder the
        scoring path's records use, so the two on-disk spellings of a
        configuration cannot drift.
    """
    fingerprint = full["fingerprint"]
    return {
        "run_id": full["run_id"],
        "model_family": full["model_family"],
        "model_size": full["model_size"],
        "epochs": full["epochs"],
        "batch_size": full["batch_size"],
        "max_seq_len": full["max_seq_len"],
        "steps": full["steps"],
        "loss": full["loss"],
        "learning_rate": full["learning_rate"],
        "tokenizer_id": full["tokenizer_id"],
        "corpus_path": full["corpus_path"],
        "corpus_format": full["corpus_format"],
        "holdout_fraction": full["holdout_fraction"],
        "optimizer": full["optimizer"],
        "freeze_embed": full["freeze_embed"],
        "gradient_clipping": full["gradient_clipping"],
        "seed": full["seed"],
        "pretrained_run_id": full["pretrained_run_id"],
        "versions": _encode_versions(full["versions"]),
        "system": _encode_system(full["system"]),
        "fingerprint": None if fingerprint is None else encode_run_fingerprint(fingerprint),
        "git_commit": full["git_commit"],
        "config": _encode_config(full["config"]),
        "device": full["device"],
        "precision": full["precision"],
        "early_stopping_patience": full["early_stopping_patience"],
        "test_split_ratio": full["test_split_ratio"],
        "finetune_lr_cap": full["finetune_lr_cap"],
        "loss_mask_prefix_separator": full["loss_mask_prefix_separator"],
        "test_loss": full["test_loss"],
        "test_perplexity": full["test_perplexity"],
        "best_val_loss": full["best_val_loss"],
        "early_stopped": full["early_stopped"],
        "resumed_from_epoch": full["resumed_from_epoch"],
        "timing": _encode_timing(full["timing"]),
        "performance": _encode_performance(full["performance"]),
        "model_info": _encode_model_info(full["model_info"]),
        "gguf_export": _encode_gguf_export(full["gguf_export"]),
    }


__all__ = ["encode_training_manifest_full"]
