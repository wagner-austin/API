from __future__ import annotations

import logging

import pytest
from monorepo_guards._types import UnknownJson
from platform_discord.protocols import InteractionProto
from tests.support.discord_fakes import FakeBot, FakeDigitService, RecordingInteraction
from tests.support.settings import build_settings

from clubbot.cogs.digits import DigitsCog
from clubbot.config import DiscordbotSettings


class FakeEnqueuer:
    def __init__(self) -> None:
        self.calls: list[dict[str, UnknownJson]] = []

    def enqueue_train(
        self,
        *,
        request_id: str,
        user_id: int,
        model_id: str,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        augment: bool,
        notes: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "model_id": model_id,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "augment": augment,
                "notes": notes,
            }
        )
        return "job-xyz"


def make_cfg(public: bool = False) -> DiscordbotSettings:
    return build_settings(
        qr_default_border=2,
        qr_public_responses=True,
        digits_public_responses=public,
        digits_rate_limit=2,
        digits_rate_window_seconds=60,
        digits_max_image_mb=2,
    )


@pytest.mark.asyncio
async def test_train_without_enqueuer_replies_not_configured() -> None:
    bot = FakeBot()
    service = FakeDigitService()
    cfg = make_cfg(public=False)
    cog = DigitsCog(bot, cfg, service, enqueuer=None)
    inter = RecordingInteraction()
    await cog._train_impl(inter, inter.user)
    msg = str(inter.sent[-1]["content"]) if inter.sent else ""
    assert "Training is not configured" in msg


@pytest.mark.asyncio
async def test_train_enqueues_and_acknowledges() -> None:
    bot = FakeBot()
    service = FakeDigitService()
    cfg = make_cfg(public=False)
    enq = FakeEnqueuer()
    cog = DigitsCog(bot, cfg, service, enqueuer=enq)
    inter = RecordingInteraction()
    await cog._train_impl(inter, inter.user)
    assert enq.calls and isinstance(enq.calls[-1], dict)
    last = inter.sent[-1]
    assert last["embed"] is not None and last["ephemeral"] is True


@pytest.mark.asyncio
async def test_train_early_ack_return_path() -> None:
    class _Cog(DigitsCog):
        async def _safe_defer(self, interaction: InteractionProto, *, ephemeral: bool) -> bool:
            _ = interaction
            _ = ephemeral
            return False

    bot = FakeBot()
    service = FakeDigitService()
    cfg = make_cfg(public=False)
    cog = _Cog(bot, cfg, service, enqueuer=None)
    inter = RecordingInteraction()
    await cog._train_impl(inter, inter.user)
    assert inter.sent == []


@pytest.mark.asyncio
async def test_train_user_id_none_triggers_user_error() -> None:
    bot = FakeBot()
    service = FakeDigitService()
    cfg = make_cfg(public=False)
    enq = FakeEnqueuer()
    cog = DigitsCog(bot, cfg, service, enqueuer=enq)
    inter = RecordingInteraction()
    await cog._train_impl(inter, None)
    msg = str(inter.sent[-1]["content"]) if inter.sent else ""
    assert "Could not determine your user id" in msg


@pytest.mark.asyncio
async def test_train_handles_enqueue_exception() -> None:
    bot = FakeBot()
    service = FakeDigitService()
    cfg = make_cfg(public=False)

    class _BadEnq(FakeEnqueuer):
        def enqueue_train(
            self,
            *,
            request_id: str,
            user_id: int,
            model_id: str,
            epochs: int,
            batch_size: int,
            lr: float,
            seed: int,
            augment: bool,
            notes: str | None = None,
        ) -> str:
            raise RuntimeError("boom")

    cog = DigitsCog(bot, cfg, service, enqueuer=_BadEnq())
    inter = RecordingInteraction()
    await cog._train_impl(inter, inter.user)
    assert inter.sent and "An error occurred" in str(inter.sent[-1]["content"])


logger = logging.getLogger(__name__)
