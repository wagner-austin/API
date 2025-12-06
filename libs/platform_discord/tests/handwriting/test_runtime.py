from __future__ import annotations

import pytest

from platform_discord.handwriting.runtime import (
    DigitsRuntime,
    RequestAction,
    new_runtime,
    on_artifact,
    on_batch,
    on_best,
    on_completed,
    on_failed,
    on_progress,
    on_prune,
    on_started,
    on_upload,
)


def _base_runtime() -> DigitsRuntime:
    return new_runtime()


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


def test_runtime_started_progress_completed_flow() -> None:
    rt = _base_runtime()
    act1: RequestAction = on_started(
        rt,
        user_id=1,
        request_id="r1",
        model_id="mnist",
        total_epochs=5,
        queue="digits",
        cpu_cores=2,
    )
    assert act1["request_id"] == "r1" and act1["user_id"] == 1

    act2 = on_progress(rt, user_id=1, request_id="r1", epoch=1, total_epochs=5, val_acc=0.5)
    assert act2["request_id"] == "r1"

    act3 = on_batch(
        rt,
        user_id=1,
        request_id="r1",
        model_id="mnist",
        epoch=1,
        total_epochs=5,
        batch=1,
        total_batches=10,
        batch_loss=0.4,
        batch_acc=0.8,
        avg_loss=0.45,
        samples_per_sec=120.0,
        main_rss_mb=200,
        workers_rss_mb=100,
        worker_count=2,
        cgroup_usage_mb=800,
        cgroup_limit_mb=2048,
        cgroup_pct=39.0,
        anon_mb=100,
        file_mb=50,
    )
    assert act3["request_id"] == "r1"

    on_best(rt, user_id=1, request_id="r1", epoch=1, val_acc=0.6)
    on_artifact(rt, user_id=1, request_id="r1", path="/tmp/model.pt")
    on_upload(rt, user_id=1, request_id="r1", status=200, model_bytes=10, manifest_bytes=1)
    on_prune(rt, user_id=1, request_id="r1", deleted_count=0)

    act4 = on_completed(rt, user_id=1, request_id="r1", model_id="mnist", run_id="rid", val_acc=0.9)
    assert act4 is not None and act4["request_id"] == "r1"


def test_runtime_batch_without_prior_config_and_failure() -> None:
    rt = _base_runtime()
    # on_batch for unknown request should synthesize a default config
    act = on_batch(
        rt,
        user_id=2,
        request_id="r2",
        model_id="m",
        epoch=1,
        total_epochs=2,
        batch=1,
        total_batches=2,
        batch_loss=1.0,
        batch_acc=0.1,
        avg_loss=1.1,
        samples_per_sec=10.0,
        main_rss_mb=50,
        workers_rss_mb=10,
        worker_count=1,
        cgroup_usage_mb=60,
        cgroup_limit_mb=100,
        cgroup_pct=60.0,
        anon_mb=10,
        file_mb=5,
    )
    assert act["request_id"] == "r2"

    # on_failed should build a failed embed and cleanup state
    fail = on_failed(
        rt,
        user_id=2,
        request_id="r2",
        model_id="m",
        error_kind="system",
        message="upload failed",
        queue="digits",
        status="failed",
    )
    assert fail["request_id"] == "r2"


def test_on_progress_branches_for_time_formatting() -> None:
    """Cover time_s formatting branches."""
    rt = _base_runtime()
    # First call to set up the config
    on_started(
        rt,
        user_id=1,
        request_id="r3",
        model_id="test",
        total_epochs=5,
        queue="digits",
    )

    # Test with train_loss but no val_acc
    act1 = on_progress(
        rt,
        user_id=1,
        request_id="r3",
        epoch=2,
        total_epochs=5,
        val_acc=None,
        train_loss=0.5,
        time_s=None,
    )
    assert act1["request_id"] == "r3"

    # Test with time_s < 60 seconds (seconds only)
    act2 = on_progress(
        rt,
        user_id=1,
        request_id="r3",
        epoch=3,
        total_epochs=5,
        val_acc=0.8,
        train_loss=0.3,
        time_s=30.0,
    )
    assert act2["request_id"] == "r3"

    # Test with time_s > 60 seconds (minutes branch)
    act3 = on_progress(
        rt,
        user_id=1,
        request_id="r3",
        epoch=4,
        total_epochs=5,
        val_acc=0.85,
        train_loss=0.2,
        time_s=125.0,
    )
    assert act3["request_id"] == "r3"

    # Test with no train_loss and no val_acc (skip field add)
    act4 = on_progress(
        rt,
        user_id=1,
        request_id="r3",
        epoch=5,
        total_epochs=5,
        val_acc=None,
        train_loss=None,
        time_s=45.0,
    )
    assert act4["request_id"] == "r3"


def test_on_completed_without_prior_config() -> None:
    """Cover on_completed when config is None."""
    rt = _base_runtime()
    # Call on_completed without any prior on_started call
    result = on_completed(
        rt,
        user_id=3,
        request_id="r4",
        model_id="model-without-config",
        run_id="run-123",
        val_acc=0.92,
    )
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "r4"
    assert result["user_id"] == 3


def test_on_failed_without_prior_config() -> None:
    """Cover on_failed when config is None."""
    rt = _base_runtime()
    # Call on_failed without any prior on_started call
    result = on_failed(
        rt,
        user_id=4,
        request_id="r5",
        model_id="model-without-config",
        error_kind="system",
        message="some error",
        queue="test-queue",
        status="failed",
    )
    assert result["request_id"] == "r5"
    assert result["user_id"] == 4


def test_on_progress_val_acc_better_epoch() -> None:
    """Cover val_acc tracking when epoch improves."""
    rt = _base_runtime()
    on_started(
        rt,
        user_id=1,
        request_id="r6",
        model_id="test",
        total_epochs=10,
        queue="digits",
    )

    # First progress with val_acc
    on_progress(
        rt,
        user_id=1,
        request_id="r6",
        epoch=2,
        total_epochs=10,
        val_acc=0.7,
    )

    # Second progress with higher epoch and val_acc - updates best_epoch
    on_progress(
        rt,
        user_id=1,
        request_id="r6",
        epoch=5,
        total_epochs=10,
        val_acc=0.85,
    )

    # Check metrics were tracked
    assert rt["_metrics"]["r6"]["best_epoch"] == 5
