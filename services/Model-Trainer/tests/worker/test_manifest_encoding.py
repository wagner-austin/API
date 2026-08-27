"""The manifest a run writes must be readable by the code that wrote it.

That was not true between 36a51b50 and this module. The manifest had a
decoder and no encoder, so it was dumped straight to JSON -- and the moment a
field stopped being JSON-native, the file on disk stopped being one the
decoder accepts. ``DeterminismRecord.settings`` is a tuple of
``(name, value)`` pairs in memory, which dumps as a list of lists, and the
decoder requires an object.

Nothing caught it because the field was optional and every test left it
None, so the value that would have broken was never serialised. These tests
are the round trip that would have.
"""

from __future__ import annotations

from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import TRUE, UNPINNED_STACK, determinism_record
from platform_core.json_utils import dump_json_str, narrow_json_to_dict
from platform_core.json_utils import load_json_str as load_json
from platform_core.testing import sample_run_fingerprint

from model_trainer.infra.persistence.models import (
    GgufExportManifest,
    TrainingManifestConfig,
    TrainingManifestFull,
)
from model_trainer.worker.manifest import load_manifest_from_text
from model_trainer.worker.manifest_encoding import encode_training_manifest_full

_PINNED = determinism_record("torch", {"cudnn_deterministic": TRUE})

_FINGERPRINT = sample_run_fingerprint(
    image_digest="sha256:" + "cd" * 32,
    gpu_model="NVIDIA A100 80GB PCIe",
    driver_version="580.82.07",
    determinism=_PINNED,
)

_CONFIG: TrainingManifestConfig = {
    "model_family": "gpt2",
    "model_size": "small",
    "max_seq_len": 8,
    "num_epochs": 1,
    "batch_size": 1,
    "learning_rate": 0.001,
    "tokenizer_id": "tok",
    "corpus_path": "/corpus",
    "holdout_fraction": 0.1,
    "seed": 0,
    "pretrained_run_id": None,
    "freeze_embed": False,
    "gradient_clipping": 1.0,
    "optimizer": "adamw",
    "device": "cuda",
    "precision": "fp32",
    "early_stopping_patience": 5,
    "test_split_ratio": 0.15,
    "finetune_lr_cap": 5e-5,
    "loss_mask_prefix_separator": None,
}


def _full(
    *,
    fingerprint: RunFingerprint | None,
    gguf_export: GgufExportManifest | None,
) -> TrainingManifestFull:
    """Build a complete manifest for encoding.

    Args:
        fingerprint: The configuration to record, or None for a run that
            records none.
        gguf_export: The export block, or None for the usual case.

    Returns:
        The manifest.
    """
    return {
        "run_id": "r",
        "model_family": "gpt2",
        "model_size": "small",
        "epochs": 1,
        "batch_size": 1,
        "max_seq_len": 8,
        "steps": 0,
        "loss": 0.0,
        "learning_rate": 0.001,
        "tokenizer_id": "tok",
        "corpus_path": "/corpus",
        "holdout_fraction": 0.1,
        "optimizer": "adamw",
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "seed": 0,
        "pretrained_run_id": None,
        "versions": {
            "torch": "2.6.0+cu124",
            "transformers": "4.46.3",
            "tokenizers": "0.20.3",
            "datasets": "3.6.0",
        },
        "system": {
            "cpu_count": 8,
            "platform": "Linux",
            "platform_release": "5.15.0",
            "machine": "x86_64",
        },
        "fingerprint": fingerprint,
        "git_commit": "g",
        "config": _CONFIG,
        "device": "cuda",
        "precision": "fp32",
        "early_stopping_patience": 5,
        "test_split_ratio": 0.15,
        "finetune_lr_cap": 5e-5,
        "loss_mask_prefix_separator": None,
        "test_loss": None,
        "test_perplexity": None,
        "best_val_loss": None,
        "early_stopped": False,
        "resumed_from_epoch": None,
        "timing": {
            "training_duration_sec": 10.5,
            "started_at": "2026-08-25T10:00:00",
            "completed_at": "2026-08-25T10:00:10",
        },
        "performance": {
            "peak_gpu_memory_mb": 1024.0,
            "avg_samples_per_sec": 100.0,
            "total_tokens_processed": 1024,
        },
        "model_info": {"param_count": 1000, "model_size_mb": 5.0, "vocab_size": 256},
        "gguf_export": gguf_export,
    }


class TestTheWrittenManifestIsReadable:
    """The regression that motivated the encoder existing at all."""

    def test_a_pinned_posture_survives_the_round_trip(self) -> None:
        encoded = encode_training_manifest_full(_full(fingerprint=_FINGERPRINT, gguf_export=None))

        decoded = load_manifest_from_text(dump_json_str(encoded))
        fingerprint = decoded["fingerprint"]
        if fingerprint is None:
            raise AssertionError("the encoded manifest declared a fingerprint")
        assert fingerprint == _FINGERPRINT

    def test_the_settings_reach_disk_as_an_object_not_a_list(self) -> None:
        """The exact shape the decoder requires and the raw dump did not write.

        Asserted on the JSON rather than on the decoded record, because the
        decode is what used to fail -- checking only the round trip would
        pass on any encoding the decoder happens to accept, and this pins
        the one it must produce.
        """
        encoded = encode_training_manifest_full(_full(fingerprint=_FINGERPRINT, gguf_export=None))

        reparsed = narrow_json_to_dict(load_json(dump_json_str(encoded)))
        fingerprint = narrow_json_to_dict(reparsed["fingerprint"])
        determinism = narrow_json_to_dict(fingerprint["determinism"])
        assert determinism["settings"] == {"cudnn_deterministic": "true"}

    def test_an_unpinned_posture_also_survives(self) -> None:
        """A record with no settings must not encode as an empty list."""
        unpinned = sample_run_fingerprint(
            image_digest="",
            gpu_model="",
            driver_version="",
            determinism=determinism_record(UNPINNED_STACK, {}),
        )

        encoded = encode_training_manifest_full(_full(fingerprint=unpinned, gguf_export=None))

        assert load_manifest_from_text(dump_json_str(encoded))["fingerprint"] == unpinned

    def test_a_manifest_recording_no_configuration_encodes_null(self) -> None:
        encoded = encode_training_manifest_full(_full(fingerprint=None, gguf_export=None))

        assert encoded["fingerprint"] is None
        assert load_manifest_from_text(dump_json_str(encoded))["fingerprint"] is None


class TestTheOtherBlocksSurviveToo:
    def test_a_gguf_export_is_carried_through(self) -> None:
        """Most runs export none, so this branch is the one a round trip misses."""
        export: GgufExportManifest = {
            "output_type": "q4_k_m",
            "output_filename": "model-q4_k_m.gguf",
            "output_size_bytes": 4096,
        }

        encoded = encode_training_manifest_full(_full(fingerprint=_FINGERPRINT, gguf_export=export))

        assert narrow_json_to_dict(encoded["gguf_export"]) == {
            "output_type": "q4_k_m",
            "output_filename": "model-q4_k_m.gguf",
            "output_size_bytes": 4096,
        }

    def test_no_gguf_export_encodes_null_rather_than_an_empty_object(self) -> None:
        encoded = encode_training_manifest_full(_full(fingerprint=None, gguf_export=None))

        assert encoded["gguf_export"] is None

    def test_the_system_block_carries_no_card(self) -> None:
        """The card belongs to the fingerprint; a copy here is the old fork."""
        encoded = encode_training_manifest_full(_full(fingerprint=_FINGERPRINT, gguf_export=None))

        assert sorted(narrow_json_to_dict(encoded["system"])) == [
            "cpu_count",
            "machine",
            "platform",
            "platform_release",
        ]

    def test_every_nested_block_carries_exactly_its_own_fields(self) -> None:
        """A field dropped from the encoder fails here, not on a later read.

        Asserted as exact key sets rather than presence, so adding a field to
        a manifest type without adding it to the encoder is caught by this
        test rather than by whoever reads the file next.
        """
        encoded = encode_training_manifest_full(_full(fingerprint=_FINGERPRINT, gguf_export=None))

        assert sorted(narrow_json_to_dict(encoded["versions"])) == [
            "datasets",
            "tokenizers",
            "torch",
            "transformers",
        ]
        assert sorted(narrow_json_to_dict(encoded["timing"])) == [
            "completed_at",
            "started_at",
            "training_duration_sec",
        ]
        assert sorted(narrow_json_to_dict(encoded["performance"])) == [
            "avg_samples_per_sec",
            "peak_gpu_memory_mb",
            "total_tokens_processed",
        ]
        assert sorted(narrow_json_to_dict(encoded["model_info"])) == [
            "model_size_mb",
            "param_count",
            "vocab_size",
        ]
        assert sorted(narrow_json_to_dict(encoded["config"])) == sorted(_CONFIG)
