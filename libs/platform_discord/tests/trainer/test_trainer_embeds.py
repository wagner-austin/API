from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_discord.testing import fake_load_discord_module, hooks
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


def _base_final(*, held_out: bool = True) -> FinalMetrics:
    """Build a completion event.

    Args:
        held_out: Whether the figures were measured on data the model did not
            train on.

    Returns:
        The event.
    """
    return {
        "type": "trainer.metrics.completed.v1",
        "job_id": "test-job",
        "user_id": 1,
        "final_loss": 0.8,
        "final_ppl": 2.23,
        "held_out": held_out,
        "artifact_path": "/x",
    }


@pytest.fixture(autouse=True)
def _use_fake_discord() -> Generator[None, None, None]:
    """Set up fake discord module via hooks."""
    hooks.load_discord_module = fake_load_discord_module
    yield


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


def _results_value(final: FinalMetrics) -> str:
    """Render a completed embed and return its Results field.

    Args:
        final: Completion event to render.

    Returns:
        The text a person would read in Discord.
    """
    embed = build_training_embed(
        request_id="r", config=_base_config(), status="completed", final=final
    )
    fields = embed.to_dict().get("fields", [])
    return next(f["value"] for f in fields if f["name"] == "Results")


class TestTheResultsLabelMatchesTheNumber:
    """The embed said "Test Loss" for runs that held nothing out.

    `train_job_lifecycle` substituted training loss whenever `test_loss` was
    None, into an event field named `test_loss`, and this embed printed it
    under a label a person reads as generalisation. Both halves are now
    asserted here, because the publisher-side fix is invisible from Discord
    and this is the surface where it was wrong.
    """

    def test_held_out_figures_are_labelled_test(self) -> None:
        value = _results_value(_base_final(held_out=True))
        assert "**Test Loss:** `0.8000`" in value
        assert "**Test PPL:** `2.23`" in value
        assert "Train Loss" not in value

    def test_training_figures_are_labelled_train(self) -> None:
        value = _results_value(_base_final(held_out=False))
        assert "**Train Loss:** `0.8000`" in value
        assert "**Train PPL:** `2.23`" in value
        assert "Test Loss" not in value

    def test_training_figures_say_why_there_is_no_test_number(self) -> None:
        """A relabel alone reads as a different metric, not a missing one."""
        value = _results_value(_base_final(held_out=False))
        assert "_No held-out split: these are training figures._" in value

    def test_held_out_figures_carry_no_such_caveat(self) -> None:
        assert "held-out split" not in _results_value(_base_final(held_out=True))


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
