"""Tests for cog command callbacks invoking their implementations.

These tests verify that Discord command callbacks correctly call through
to their implementation methods using proper hook-based testing.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Generic, Protocol, TypeVar

import discord
import pytest
from platform_discord.protocols import InteractionProto, UserProto, _DiscordInteraction
from tests.support.discord_fakes import FakeAttachment, FakeBot, RecordingInteraction
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.cogs.digits import DigitsCog, _HasId
from clubbot.cogs.invite import InviteCog
from clubbot.cogs.qr import QRCog
from clubbot.cogs.trainer import TrainerCog
from clubbot.cogs.transcript import TranscriptCog
from clubbot.config import DiscordbotSettings
from clubbot.services.qr.client import QRService
from clubbot.services.transcript.client import TranscriptService


class _HasIdProto(Protocol):
    """Protocol for objects with optional id property."""

    @property
    def id(self) -> int | None: ...


class _InviteCB(Protocol):
    def __call__(
        self, self_obj: InviteCog, interaction: _DiscordInteraction
    ) -> Awaitable[None]: ...


class _QRCB(Protocol):
    def __call__(
        self, self_obj: QRCog, interaction: _DiscordInteraction, url: str
    ) -> Awaitable[None]: ...


class _DigitsReadCB(Protocol):
    def __call__(
        self, self_obj: DigitsCog, interaction: _DiscordInteraction, image: discord.Attachment
    ) -> Awaitable[None]: ...


class _DigitsTrainCB(Protocol):
    def __call__(
        self, self_obj: DigitsCog, interaction: _DiscordInteraction
    ) -> Awaitable[None]: ...


class _TrainerCB(Protocol):
    def __call__(
        self,
        self_obj: TrainerCog,
        interaction: _DiscordInteraction,
        model_family: str,
        model_size: str,
        max_seq_len: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        corpus_path: str,
        tokenizer_id: str,
    ) -> Awaitable[None]: ...


class _TranscriptCB(Protocol):
    def __call__(
        self, self_obj: TranscriptCog, interaction: _DiscordInteraction, url: str
    ) -> Awaitable[None]: ...


_CB = TypeVar("_CB")


class _HasCallback(Protocol, Generic[_CB]):
    callback: _CB


class _TrackingInviteCog(InviteCog):
    """InviteCog subclass that tracks impl calls."""

    def __init__(self, bot: FakeBot, config: DiscordbotSettings) -> None:
        super().__init__(bot, config)
        self.impl_called = 0

    async def _invite_impl(self, interaction: InteractionProto) -> None:
        self.impl_called += 1


class _TrackingQRCog(QRCog):
    """QRCog subclass that tracks impl calls."""

    def __init__(self, bot: FakeBot, config: DiscordbotSettings, svc: QRService) -> None:
        super().__init__(bot, config, svc)
        self.impl_called = 0

    async def _qrcode_impl(self, interaction: InteractionProto, url: str) -> None:
        _ = url
        self.impl_called += 1


class _TrackingDigitsCog(DigitsCog):
    """DigitsCog subclass that tracks impl calls."""

    def __init__(
        self,
        bot: FakeBot,
        config: DiscordbotSettings,
        svc: _test_hooks.DigitsEnqueuerLike | None,
    ) -> None:
        from clubbot.services.digits.app import DigitService

        _ = svc  # Unused in this test subclass
        service = DigitService(config)
        super().__init__(bot, config, service, enqueuer=None, autostart_subscriber=False)
        self.read_called = 0
        self.train_called = 0

    async def _read_impl(
        self,
        interaction: InteractionProto,
        user: _HasId | None,
        image: discord.Attachment,
    ) -> None:
        _ = (user, image)
        self.read_called += 1

    async def _train_impl(
        self,
        interaction: InteractionProto,
        user: discord.User | discord.Member | UserProto | None,
    ) -> None:
        _ = user
        self.train_called += 1


class _TrackingTrainerCog(TrainerCog):
    """TrainerCog subclass that tracks impl calls."""

    def __init__(self, bot: FakeBot, config: DiscordbotSettings) -> None:
        super().__init__(bot=bot, config=config)
        self.impl_called = 0

    async def _train_model_impl(
        self,
        interaction: InteractionProto,
        *,
        model_family: str,
        model_size: str,
        max_seq_len: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        corpus_path: str,
        tokenizer_id: str,
    ) -> None:
        _ = (
            model_family,
            model_size,
            max_seq_len,
            num_epochs,
            batch_size,
            learning_rate,
            corpus_path,
            tokenizer_id,
        )
        self.impl_called += 1


class _TrackingTranscriptCog(TranscriptCog):
    """TranscriptCog subclass that tracks impl calls."""

    def __init__(
        self,
        bot: FakeBot,
        config: DiscordbotSettings,
        svc: TranscriptService,
    ) -> None:
        super().__init__(bot=bot, config=config, transcript_service=svc)
        self.impl_called = 0

    async def _transcript_impl(
        self,
        wrapped: InteractionProto,
        user_obj: _HasIdProto | None,
        guild_obj: _HasIdProto | None,
        url: str,
    ) -> None:
        _ = (wrapped, user_obj, guild_obj, url)
        self.impl_called += 1


def _wrap_fake(
    interaction: _test_hooks.DiscordInteractionLike,
) -> _test_hooks.InteractionProtoLike:
    """Fake wrap_interaction that returns a RecordingInteraction."""
    _ = interaction
    result: _test_hooks.InteractionProtoLike = RecordingInteraction()
    return result


@pytest.mark.asyncio
async def test_invite_wrapper_callback_invokes_impl() -> None:
    cfg = build_settings()
    bot = FakeBot()
    cog = _TrackingInviteCog(bot, cfg)

    # Override wrap_interaction hook
    _test_hooks.wrap_interaction = _wrap_fake

    name_inv = "invite"
    inv_attr: _HasCallback[_InviteCB] = getattr(cog, name_inv)
    cb: _InviteCB = inv_attr.callback
    await cb(cog, RecordingInteraction())

    assert cog.impl_called == 1


@pytest.mark.asyncio
async def test_qr_wrapper_callback_invokes_impl() -> None:
    cfg = build_settings(qr_api_url="http://api")
    bot2 = FakeBot()
    cog = _TrackingQRCog(bot2, cfg, QRService(cfg))

    # Override wrap_interaction hook
    _test_hooks.wrap_interaction = _wrap_fake

    name_qr = "qrcode"
    qr_attr: _HasCallback[_QRCB] = getattr(cog, name_qr)
    cb: _QRCB = qr_attr.callback
    await cb(cog, RecordingInteraction(), "https://x")

    assert cog.impl_called == 1


@pytest.mark.asyncio
async def test_digits_read_and_train_wrappers_invoke_impls() -> None:
    cfg = build_settings(handwriting_api_url="http://h")
    bot3 = FakeBot()
    cog = _TrackingDigitsCog(bot3, cfg, None)

    # Override wrap_interaction hook
    _test_hooks.wrap_interaction = _wrap_fake

    name_read = "read"
    name_train = "train"
    read_attr: _HasCallback[_DigitsReadCB] = getattr(cog, name_read)
    train_attr: _HasCallback[_DigitsTrainCB] = getattr(cog, name_train)
    read_cb: _DigitsReadCB = read_attr.callback
    train_cb: _DigitsTrainCB = train_attr.callback

    att = FakeAttachment(filename="x.png", content_type="image/png", size=1, data=b"x")
    await read_cb(cog, RecordingInteraction(), att)
    await train_cb(cog, RecordingInteraction())

    assert cog.read_called == 1
    assert cog.train_called == 1


@pytest.mark.asyncio
async def test_trainer_and_transcript_wrappers_invoke_impls() -> None:
    cfg = build_settings(model_trainer_api_url="http://t")
    tcfg = build_settings(transcript_api_url="http://x")
    tsvc = TranscriptService(tcfg)
    fb = FakeBot()
    trainer = _TrackingTrainerCog(bot=fb, config=cfg)
    transcript = _TrackingTranscriptCog(bot=fb, config=tcfg, svc=tsvc)

    # Override wrap_interaction hook
    _test_hooks.wrap_interaction = _wrap_fake

    name_tm = "train_model"
    name_tx = "transcript"
    tm_attr: _HasCallback[_TrainerCB] = getattr(trainer, name_tm)
    tx_attr: _HasCallback[_TranscriptCB] = getattr(transcript, name_tx)
    t_cb: _TrainerCB = tm_attr.callback
    x_cb: _TranscriptCB = tx_attr.callback

    await t_cb(
        trainer,
        RecordingInteraction(),
        model_family="gpt2",
        model_size="small",
        max_seq_len=64,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        corpus_path="/d",
        tokenizer_id="tok",
    )
    await x_cb(transcript, RecordingInteraction(), "https://x")

    assert trainer.impl_called == 1
    assert transcript.impl_called == 1
