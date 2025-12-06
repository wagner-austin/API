from __future__ import annotations

import pytest

from platform_discord.handwriting.embeds import build_training_embed
from platform_discord.handwriting.types import BatchProgress, TrainingConfig, TrainingMetrics


def _base_config() -> TrainingConfig:
    return {
        "model_id": "mnist_resnet18_v1",
        "total_epochs": 10,
        "queue": "digits",
        "batch_size": 32,
        "learning_rate": 0.01,
        "device": "cpu",
        "cpu_cores": 4,
        "memory_mb": 1024,
        "optimal_threads": 2,
        "optimal_workers": 1,
        "augment": True,
        "aug_rotate": 5.0,
        "aug_translate": 0.1,
        "noise_prob": 0.05,
        "dots_prob": 0.02,
    }


@pytest.fixture(autouse=True)
def _patch_discord_embed_module(monkeypatch: pytest.MonkeyPatch) -> None:
    # Replace platform_discord.embed_helpers._load_discord_module with a minimal fake
    import platform_discord.embed_helpers as eh

    class _FakeColor:
        def __init__(self, value: int) -> None:
            self.value = int(value)

    class _Field:
        def __init__(self, name: str, value: str, inline: bool) -> None:
            self.name = name
            self.value = value
            self.inline = inline

    class _Footer:
        def __init__(self) -> None:
            self.text: str | None = None

    class _FakeEmbed:
        def __init__(
            self,
            *,
            title: str | None = None,
            description: str | None = None,
            color: _FakeColor | None = None,
        ) -> None:
            self.title = title
            self.description = description
            self.color = color
            self.footer = _Footer()
            self.fields: list[_Field] = []

        def add_field(self, *, name: str, value: str, inline: bool = True) -> _FakeEmbed:
            self.fields.append(_Field(name, value, inline))
            return self

        def set_footer(self, *, text: str) -> _FakeEmbed:
            self.footer.text = text
            return self

    class _FakeDiscordModule:
        Embed = _FakeEmbed
        Color = _FakeColor

    def _fake_loader() -> type[_FakeDiscordModule]:
        return _FakeDiscordModule

    monkeypatch.setattr(eh, "_load_discord_module", _fake_loader, raising=True)


def test_build_embed_starting_and_progress() -> None:
    cfg = _base_config()
    e1 = build_training_embed(request_id="r1", config=cfg, status="starting")
    # Minimal smoke checks: footer and job info are present via embed dict
    d1 = e1.to_dict()
    assert "footer" in d1 and "fields" in d1

    prog: BatchProgress = {
        "epoch": 3,
        "total_epochs": cfg["total_epochs"],
        "batch": 5,
        "total_batches": 20,
        "batch_loss": 0.4,
        "batch_acc": 0.8,
        "avg_loss": 0.5,
        "samples_per_sec": 123.4,
        "main_rss_mb": 256,
        "workers_rss_mb": 128,
        "worker_count": 2,
        "cgroup_usage_mb": 800,
        "cgroup_limit_mb": 2048,
        "cgroup_pct": 39.0,
        "anon_mb": 100,
        "file_mb": 50,
    }
    e2 = build_training_embed(request_id="r2", config=cfg, status="training", progress=prog)
    d2 = e2.to_dict()
    # Expect multiple fields including Progress and Memory
    names = [f["name"] for f in d2.get("fields", [])]
    assert "Progress" in names and "Memory" in names


def test_build_embed_completed_and_failed_variants() -> None:
    cfg = _base_config()
    metrics: TrainingMetrics = {
        "final_avg_loss": 0.1,
        "final_train_loss": 0.09,
        "total_time_s": 65.0,
        "avg_samples_per_sec": 111.1,
        "best_epoch": 8,
        "peak_memory_mb": 1400,
    }
    e3 = build_training_embed(
        request_id="r3",
        config=cfg,
        status="completed",
        final_val_acc=0.93,
        final_metrics=metrics,
        run_id="run-1",
    )
    d3 = e3.to_dict()
    names3 = [f["name"] for f in d3.get("fields", [])]
    assert "Training Summary" in names3 and "Final Performance" in names3 and "Run ID" in names3

    e4 = build_training_embed(
        request_id="r4",
        config=cfg,
        status="failed",
        error_kind="user",
        error_message="bad config",
    )
    d4 = e4.to_dict()
    names4 = [f["name"] for f in d4.get("fields", [])]
    assert "Configuration Issue" in names4 and "Next Steps" in names4

    e5 = build_training_embed(
        request_id="r5",
        config=cfg,
        status="failed",
        error_kind="system",
        error_message="memory pressure detected",
    )
    d5 = e5.to_dict()
    names5 = [f["name"] for f in d5.get("fields", [])]
    assert "System Error" in names5 and "Next Steps" in names5


def test_add_completion_summary_with_none_metrics() -> None:
    """Cover early return when final_metrics is None."""
    cfg = _base_config()
    # Pass no final_metrics (default is None) to hit the early return branch
    e = build_training_embed(
        request_id="r6",
        config=cfg,
        status="completed",
        final_val_acc=0.95,
        final_metrics=None,
        run_id="run-2",
    )
    d = e.to_dict()
    names = [f["name"] for f in d.get("fields", [])]
    # Training Summary should NOT be present when metrics is None
    assert "Training Summary" not in names
    assert "Final Performance" in names


def test_add_failure_section_generic_error() -> None:
    """Cover fallback next_steps for generic errors."""
    cfg = _base_config()
    e = build_training_embed(
        request_id="r7",
        config=cfg,
        status="failed",
        error_kind="system",
        error_message="unknown error occurred",  # no memory/upload keywords
    )
    d = e.to_dict()
    names = [f["name"] for f in d.get("fields", [])]
    assert "System Error" in names and "Next Steps" in names
    # Check that fallback message is present
    fields_dict = {f["name"]: f["value"] for f in d.get("fields", [])}
    assert "try again" in fields_dict.get("Next Steps", "").lower()


def test_add_failure_section_artifact_upload_error() -> None:
    """Cover the 'upload' or 'artifact' branch in _add_failure_section."""
    cfg = _base_config()
    e = build_training_embed(
        request_id="r8",
        config=cfg,
        status="failed",
        error_kind="system",
        error_message="artifact upload failed due to network issue",
    )
    d = e.to_dict()
    fields_dict = {f["name"]: f["value"] for f in d.get("fields", [])}
    assert "Artifact upload failed" in fields_dict.get("Next Steps", "")


def test_augmentations_section_all_zero_values() -> None:
    """Cover augmentation values are 0 while augment=True."""
    cfg: TrainingConfig = {
        "model_id": "test",
        "total_epochs": 5,
        "queue": "digits",
        "batch_size": 16,
        "learning_rate": 0.01,
        "device": "cpu",
        "cpu_cores": 2,
        "memory_mb": 512,
        "optimal_threads": 1,
        "optimal_workers": 0,
        "augment": True,  # augment enabled but all values zero
        "aug_rotate": 0.0,
        "aug_translate": 0.0,
        "noise_prob": 0.0,
        "dots_prob": 0.0,
    }
    e = build_training_embed(request_id="r9", config=cfg, status="starting")
    d = e.to_dict()
    # With augment=True but all values zero, aug_lines will be empty
    # so Augmentations field is NOT added
    fields_dict = {f["name"]: f["value"] for f in d.get("fields", [])}
    assert "Augmentations" not in fields_dict


def test_augmentations_section_disabled() -> None:
    """Cover augment=False branch."""
    cfg: TrainingConfig = {
        "model_id": "test",
        "total_epochs": 5,
        "queue": "digits",
        "batch_size": 16,
        "learning_rate": 0.01,
        "device": "cpu",
        "cpu_cores": 2,
        "memory_mb": 512,
        "optimal_threads": 1,
        "optimal_workers": 0,
        "augment": False,
        "aug_rotate": None,
        "aug_translate": None,
        "noise_prob": None,
        "dots_prob": None,
    }
    e = build_training_embed(request_id="r10", config=cfg, status="starting")
    d = e.to_dict()
    fields_dict = {f["name"]: f["value"] for f in d.get("fields", [])}
    assert fields_dict.get("Augmentations") == "*None*"


def test_final_performance_without_val_acc() -> None:
    """Cover final_val_acc is None branch."""
    cfg = _base_config()
    metrics: TrainingMetrics = {
        "final_avg_loss": 0.1,
        "final_train_loss": 0.09,
        "total_time_s": 30.0,  # less than 60s for branch coverage
        "avg_samples_per_sec": 100.0,
        "best_epoch": 2,
        "peak_memory_mb": 500,
    }
    e = build_training_embed(
        request_id="r11",
        config=cfg,
        status="completed",
        final_val_acc=None,
        final_metrics=metrics,
        run_id=None,  # Also cover missing run_id
    )
    d = e.to_dict()
    names = [f["name"] for f in d.get("fields", [])]
    assert "Final Performance" not in names
    assert "Run ID" not in names


def test_training_summary_partial_metrics() -> None:
    """Cover branches where individual metrics are 0 (skip)."""
    cfg = _base_config()
    metrics: TrainingMetrics = {
        "final_avg_loss": 0.0,  # Skip this line
        "final_train_loss": 0.0,  # Skip
        "total_time_s": 0.0,  # Skip
        "avg_samples_per_sec": 0.0,  # Skip
        "best_epoch": 0,  # Skip
        "peak_memory_mb": 0,  # Skip
    }
    e = build_training_embed(
        request_id="r12",
        config=cfg,
        status="completed",
        final_val_acc=0.95,
        final_metrics=metrics,
        run_id="run",
    )
    d = e.to_dict()
    names = [f["name"] for f in d.get("fields", [])]
    # Training Summary should NOT be present when all metrics are 0
    assert "Training Summary" not in names


def test_canceled_status() -> None:
    """Cover the 'canceled' status color and error handling."""
    cfg = _base_config()
    e = build_training_embed(
        request_id="r13",
        config=cfg,
        status="canceled",
        error_kind="system",
        error_message="user canceled training",
    )
    d = e.to_dict()
    assert d["title"] == "Training Canceled"


def test_unknown_status_color_fallback() -> None:
    """Cover fallback color for unknown status."""
    cfg = _base_config()
    e = build_training_embed(request_id="r14", config=cfg, status="unknown_status")
    d = e.to_dict()
    assert d["title"] == "Training Unknown_Status"
