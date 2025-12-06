from __future__ import annotations

import pytest

from platform_discord.trainer.embeds import build_training_embed
from platform_discord.trainer.types import FinalMetrics, Progress, TrainingConfig


def _base_config() -> TrainingConfig:
    return {
        "type": "trainer.metrics.config.v1",
        "job_id": "test-job",
        "user_id": 1,
        "model_family": "gpt2",
        "model_size": "small",
        "total_epochs": 10,
        "queue": "training",
        "batch_size": 8,
        "learning_rate": 0.0005,
        "cpu_cores": 4,
        "memory_mb": 2048,
        "optimal_threads": 2,
        "optimal_workers": 1,
    }


def _base_progress() -> Progress:
    return {
        "type": "trainer.metrics.progress.v1",
        "job_id": "test-job",
        "user_id": 1,
        "epoch": 1,
        "total_epochs": 10,
        "step": 100,
        "train_loss": 1.5,
        "train_ppl": 4.48,
        "grad_norm": 0.25,
        "samples_per_sec": 150.0,
    }


def _base_final() -> FinalMetrics:
    return {
        "type": "trainer.metrics.completed.v1",
        "job_id": "test-job",
        "user_id": 1,
        "test_loss": 0.8,
        "test_ppl": 2.23,
        "artifact_path": "/x",
    }


@pytest.fixture(autouse=True)
def _patch_discord_embed_module(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_trainer_build_starting_training_completed_failed() -> None:
    cfg = _base_config()
    e1 = build_training_embed(request_id="r", config=cfg, status="starting")
    d1 = e1.to_dict()
    assert "footer" in d1 and "fields" in d1

    prog = _base_progress()
    e2 = build_training_embed(request_id="r", config=cfg, status="training", progress=prog)
    d2 = e2.to_dict()
    names2 = [f["name"] for f in d2.get("fields", [])]
    assert "Progress" in names2

    final = _base_final()
    e3 = build_training_embed(request_id="r", config=cfg, status="completed", final=final)
    d3 = e3.to_dict()
    names3 = [f["name"] for f in d3.get("fields", [])]
    assert "Results" in names3

    e4 = build_training_embed(
        request_id="r",
        config=cfg,
        status="failed",
        error_kind="system",
        error_message="boom",
    )
    d4 = e4.to_dict()
    names4 = [f["name"] for f in d4.get("fields", [])]
    assert "System Error" in names4


def test_trainer_completed_with_no_final_adds_no_results() -> None:
    cfg = _base_config()
    e = build_training_embed(request_id="r2", config=cfg, status="completed", final=None)
    d = e.to_dict()
    names = [f["name"] for f in d.get("fields", [])]
    assert "Results" not in names


def test_trainer_failed_user_kind_adds_next_steps() -> None:
    cfg = _base_config()
    e = build_training_embed(
        request_id="r4",
        config=cfg,
        status="canceled",
        error_kind="user",
        error_message="invalid config",
    )
    d = e.to_dict()
    names = [f["name"] for f in d.get("fields", [])]
    assert "Configuration Issue" in names and "Next Steps" in names


def test_trainer_progress_with_validation_metrics() -> None:
    cfg = _base_config()
    prog: Progress = {
        "type": "trainer.metrics.progress.v1",
        "job_id": "test-job",
        "user_id": 1,
        "epoch": 5,
        "total_epochs": 10,
        "step": 500,
        "train_loss": 0.9,
        "train_ppl": 2.46,
        "grad_norm": 0.15,
        "samples_per_sec": 180.0,
        "val_loss": 1.1,
        "val_ppl": 3.0,
    }
    e = build_training_embed(request_id="r5", config=cfg, status="training", progress=prog)
    d = e.to_dict()
    progress_field = next(f for f in d.get("fields", []) if f["name"] == "Progress")
    assert "Val Loss" in progress_field["value"]
    assert "Val PPL" in progress_field["value"]
