from __future__ import annotations

import pytest

from platform_discord.trainer.runtime import (
    RequestAction,
    TrainerRuntime,
    new_runtime,
    on_completed,
    on_config,
    on_failed,
    on_progress,
)
from platform_discord.trainer.types import FinalMetrics, Progress, TrainingConfig


def _rt() -> TrainerRuntime:
    return new_runtime()


def _make_config(
    job_id: str = "r",
    user_id: int = 1,
    model_family: str = "gpt2",
    model_size: str = "small",
    total_epochs: int = 2,
    queue: str = "training",
) -> TrainingConfig:
    return {
        "type": "trainer.metrics.config.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_family": model_family,
        "model_size": model_size,
        "total_epochs": total_epochs,
        "queue": queue,
    }


def _make_progress(
    job_id: str = "r",
    user_id: int = 1,
    epoch: int = 1,
    total_epochs: int = 2,
    step: int = 10,
    train_loss: float = 1.0,
    train_ppl: float = 2.72,
    grad_norm: float = 0.25,
    samples_per_sec: float = 100.0,
) -> Progress:
    return {
        "type": "trainer.metrics.progress.v1",
        "job_id": job_id,
        "user_id": user_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "train_loss": train_loss,
        "train_ppl": train_ppl,
        "grad_norm": grad_norm,
        "samples_per_sec": samples_per_sec,
    }


def _make_final(
    job_id: str = "r",
    user_id: int = 1,
    test_loss: float = 0.5,
    test_ppl: float = 1.2,
    artifact_path: str = "/x",
) -> FinalMetrics:
    return {
        "type": "trainer.metrics.completed.v1",
        "job_id": job_id,
        "user_id": user_id,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "artifact_path": artifact_path,
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


def test_trainer_runtime_flow() -> None:
    rt = _rt()
    a1: RequestAction = on_config(rt, _make_config())
    assert a1["request_id"] == "r"

    a2 = on_progress(rt, _make_progress())
    assert a2["request_id"] == "r"

    a3 = on_completed(rt, _make_final())
    assert a3["request_id"] == "r"


def test_runtime_failed_without_prior_config() -> None:
    rt = _rt()
    a = on_failed(
        rt,
        user_id=2,
        request_id="r2",
        error_kind="system",
        message="boom",
        status="failed",
    )
    assert a["request_id"] == "r2"


def test_trainer_progress_without_prior_config_triggers_fallback() -> None:
    rt = _rt()
    prog = _make_progress(job_id="rx", user_id=3, total_epochs=3, step=5, train_loss=2.0)
    a = on_progress(rt, prog)
    assert a["request_id"] == "rx"


def test_trainer_completed_without_prior_config_triggers_fallback() -> None:
    rt = _rt()
    a = on_completed(
        rt, _make_final(job_id="ry", user_id=4, test_loss=0.9, test_ppl=1.8, artifact_path="/y")
    )
    assert a["request_id"] == "ry"


def test_trainer_failed_with_prior_config() -> None:
    rt = _rt()
    on_config(rt, _make_config(job_id="rz", user_id=5, model_size="s", total_epochs=1, queue="q"))
    a = on_failed(
        rt,
        user_id=5,
        request_id="rz",
        error_kind="system",
        message="boom",
        status="failed",
    )
    assert a["request_id"] == "rz"
