"""Digits training event subscriber and Discord notifications (strict, typed, DRY)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from platform_core.digits_metrics_events import (
    DEFAULT_DIGITS_EVENTS_CHANNEL,
    DigitsArtifactV1,
    DigitsBatchMetricsV1,
    DigitsBestMetricsV1,
    DigitsCompletedMetricsV1,
    DigitsConfigV1,
    DigitsEpochMetricsV1,
    DigitsEventV1,
    DigitsPruneV1,
    DigitsUploadV1,
    JobFailedV1,
    is_digits_artifact,
    is_digits_batch,
    is_digits_best,
    is_digits_completed_metrics,
    is_digits_config,
    is_digits_epoch,
    is_digits_job_failed,
    is_digits_prune,
    is_digits_upload,
    try_decode_digits_event,
)
from platform_core.json_utils import InvalidJsonError
from platform_discord.bot_subscriber import BotEventSubscriber
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
from platform_discord.handwriting.types import TrainingConfig, TrainingMetrics
from platform_discord.protocols import BotProto
from platform_discord.subscriber import MessageSource

_EVENT_TASK_NAME: Final[str] = "digits-event-subscriber"


def _decode_digits_safe(payload: str) -> DigitsEventV1 | None:
    """Decode digits event, returning None on decode failure."""
    try:
        return try_decode_digits_event(payload)
    except (InvalidJsonError, ValueError) as exc:
        from platform_core.logging import get_logger

        get_logger(__name__).debug("Failed to decode digits event: %s", exc)
        return None


class DigitsEventSubscriber(BotEventSubscriber[DigitsEventV1]):
    """Event subscriber for digits training events with Discord DM notifications.

    Inherits lifecycle management, DM notifications, and message caching from
    BotEventSubscriber. Implements event routing and runtime state management.
    """

    __slots__ = ("_runtime",)

    def __init__(
        self,
        bot: BotProto,
        *,
        redis_url: str,
        events_channel: str | None = None,
        source_factory: Callable[[str], MessageSource] | None = None,
    ) -> None:
        super().__init__(
            bot,
            redis_url=redis_url,
            events_channel=events_channel or DEFAULT_DIGITS_EVENTS_CHANNEL,
            task_name=_EVENT_TASK_NAME,
            decode=_decode_digits_safe,
            source_factory=source_factory,
        )
        self._runtime: DigitsRuntime = new_runtime()

    @property
    def configs(self) -> dict[str, TrainingConfig]:
        return self._runtime["_configs"]

    @property
    def metrics(self) -> dict[str, TrainingMetrics]:
        return self._runtime["_metrics"]

    @property
    def _configs(self) -> dict[str, TrainingConfig]:
        return self._runtime["_configs"]

    @property
    def _metrics_map(self) -> dict[str, TrainingMetrics]:
        return self._runtime["_metrics"]

    @property
    def _rt(self) -> DigitsRuntime:
        return self._runtime

    async def _handle_event(self, event: DigitsEventV1) -> None:
        if is_digits_config(event):
            await self._handle_config_event(event)
            return
        if is_digits_batch(event):
            await self._handle_batch_event(event)
            return
        if is_digits_epoch(event):
            await self._handle_epoch_event(event)
            return
        if is_digits_best(event):
            self._handle_best_event(event)
            return
        if is_digits_artifact(event):
            self._handle_artifact_event(event)
            return
        if is_digits_upload(event):
            self._handle_upload_event(event)
            return
        if is_digits_prune(event):
            self._handle_prune_event(event)
            return
        if is_digits_completed_metrics(event):
            await self._handle_completed_event(event)
            return
        if is_digits_job_failed(event):
            await self._handle_failed_event(event)

    async def _handle_config_event(self, event: DigitsConfigV1) -> None:
        cpu_cores_val = event.get("cpu_cores")
        optimal_threads_val = event.get("optimal_threads")
        memory_mb_val = event.get("memory_mb")
        optimal_workers_val = event.get("optimal_workers")
        max_batch_size_val = event.get("max_batch_size")
        device_val = event.get("device")
        batch_size_val = event.get("batch_size")
        learning_rate_val = event.get("learning_rate")
        augment_val = event.get("augment")
        aug_rotate_val = event.get("aug_rotate")
        aug_translate_val = event.get("aug_translate")
        noise_prob_val = event.get("noise_prob")
        dots_prob_val = event.get("dots_prob")
        act = on_started(
            self._runtime,
            user_id=event["user_id"],
            request_id=event["job_id"],
            model_id=event["model_id"],
            total_epochs=event["total_epochs"],
            queue=event["queue"],
            cpu_cores=(cpu_cores_val if isinstance(cpu_cores_val, int) else None),
            optimal_threads=(optimal_threads_val if isinstance(optimal_threads_val, int) else None),
            memory_mb=(memory_mb_val if isinstance(memory_mb_val, int) else None),
            optimal_workers=(optimal_workers_val if isinstance(optimal_workers_val, int) else None),
            max_batch_size=(max_batch_size_val if isinstance(max_batch_size_val, int) else None),
            device=(device_val if isinstance(device_val, str) else None),
            batch_size=(batch_size_val if isinstance(batch_size_val, int) else None),
            learning_rate=(
                float(learning_rate_val) if isinstance(learning_rate_val, int | float) else None
            ),
            augment=(augment_val if isinstance(augment_val, bool) else None),
            aug_rotate=(float(aug_rotate_val) if isinstance(aug_rotate_val, int | float) else None),
            aug_translate=(
                float(aug_translate_val) if isinstance(aug_translate_val, int | float) else None
            ),
            noise_prob=(float(noise_prob_val) if isinstance(noise_prob_val, int | float) else None),
            dots_prob=(float(dots_prob_val) if isinstance(dots_prob_val, int | float) else None),
        )
        await self._maybe_notify(act)

    async def _handle_batch_event(self, event: DigitsBatchMetricsV1) -> None:
        act = on_batch(
            self._runtime,
            user_id=event["user_id"],
            request_id=event["job_id"],
            model_id=event["model_id"],
            epoch=event["epoch"],
            total_epochs=event["total_epochs"],
            batch=event["batch"],
            total_batches=event["total_batches"],
            batch_loss=event["batch_loss"],
            batch_acc=event["batch_acc"],
            avg_loss=event["avg_loss"],
            samples_per_sec=event["samples_per_sec"],
            main_rss_mb=event["main_rss_mb"],
            workers_rss_mb=event["workers_rss_mb"],
            worker_count=event["worker_count"],
            cgroup_usage_mb=event["cgroup_usage_mb"],
            cgroup_limit_mb=event["cgroup_limit_mb"],
            cgroup_pct=event["cgroup_pct"],
            anon_mb=event["anon_mb"],
            file_mb=event["file_mb"],
        )
        await self._maybe_notify(act)

    async def _handle_epoch_event(self, event: DigitsEpochMetricsV1) -> None:
        act = on_progress(
            self._runtime,
            user_id=event["user_id"],
            request_id=event["job_id"],
            epoch=event["epoch"],
            total_epochs=event["total_epochs"],
            val_acc=event.get("val_acc"),
            train_loss=event.get("train_loss"),
            time_s=event.get("time_s"),
        )
        await self._maybe_notify(act)

    def _handle_best_event(self, event: DigitsBestMetricsV1) -> None:
        on_best(
            self._runtime,
            user_id=event["user_id"],
            request_id=event["job_id"],
            epoch=event["epoch"],
            val_acc=event["val_acc"],
        )

    def _handle_artifact_event(self, event: DigitsArtifactV1) -> None:
        on_artifact(
            self._runtime,
            user_id=event["user_id"],
            request_id=event["job_id"],
            path=event["path"],
        )

    def _handle_upload_event(self, event: DigitsUploadV1) -> None:
        on_upload(
            self._runtime,
            user_id=event["user_id"],
            request_id=event["job_id"],
            status=event["status"],
            model_bytes=event["model_bytes"],
            manifest_bytes=event["manifest_bytes"],
        )

    def _handle_prune_event(self, event: DigitsPruneV1) -> None:
        on_prune(
            self._runtime,
            user_id=event["user_id"],
            request_id=event["job_id"],
            deleted_count=event["deleted_count"],
        )

    async def _handle_completed_event(self, event: DigitsCompletedMetricsV1) -> None:
        act_opt = on_completed(
            self._runtime,
            user_id=event["user_id"],
            request_id=event["job_id"],
            model_id=event["model_id"],
            run_id=None,
            val_acc=event["val_acc"],
        )
        if act_opt is not None:
            await self._maybe_notify(act_opt)

    async def _handle_failed_event(self, event: JobFailedV1) -> None:
        act = on_failed(
            self._runtime,
            user_id=event["user_id"],
            request_id=event["job_id"],
            model_id="unknown",
            error_kind=event["error_kind"],
            message=event["message"],
            queue="unknown",
            status="failed",
        )
        await self._maybe_notify(act)

    async def _maybe_notify(self, act: RequestAction) -> None:
        embed = act["embed"]
        if embed is None:
            return
        await self.notify(act["user_id"], act["request_id"], embed)

    # Expose direct helpers for tests to invoke strict handlers
    async def _on_config(self, event: DigitsConfigV1) -> None:
        await self._handle_config_event(event)

    async def _on_batch(self, event: DigitsBatchMetricsV1) -> None:
        await self._handle_batch_event(event)

    async def _on_epoch(self, event: DigitsEpochMetricsV1) -> None:
        await self._handle_epoch_event(event)

    def _on_best(self, event: DigitsBestMetricsV1) -> None:
        self._handle_best_event(event)

    def _on_artifact(self, event: DigitsArtifactV1) -> None:
        self._handle_artifact_event(event)

    def _on_upload(self, event: DigitsUploadV1) -> None:
        self._handle_upload_event(event)

    def _on_prune(self, event: DigitsPruneV1) -> None:
        self._handle_prune_event(event)

    async def _on_completed(self, event: DigitsCompletedMetricsV1) -> None:
        await self._handle_completed_event(event)

    async def _on_failed(self, event: JobFailedV1) -> None:
        await self._handle_failed_event(event)


__all__ = ["DigitsEventSubscriber"]
