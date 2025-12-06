from __future__ import annotations

from collections.abc import Awaitable
from typing import Generic, Protocol, TypeVar

import discord
import pytest
from platform_discord.protocols import InteractionProto, UserProto, _DiscordInteraction
from tests.support.discord_fakes import FakeAttachment, RecordingInteraction
from tests.support.settings import build_settings

from clubbot.cogs.digits import DigitsCog
from clubbot.cogs.invite import InviteCog
from clubbot.cogs.qr import QRCog
from clubbot.cogs.trainer import TrainerCog
from clubbot.cogs.transcript import TranscriptCog
from clubbot.services.qr.client import QRService
from clubbot.services.transcript.client import TranscriptService


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


@pytest.mark.asyncio
async def test_invite_wrapper_callback_invokes_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_settings()
    from tests.support.discord_fakes import FakeBot

    bot = FakeBot()
    cog = InviteCog(bot, cfg)
    called: dict[str, int] = {"n": 0}

    async def impl(self_obj: InviteCog, _i: InteractionProto) -> None:
        called["n"] += 1

    def _wrap(_i: _DiscordInteraction) -> RecordingInteraction:
        return RecordingInteraction()

    monkeypatch.setattr("platform_discord.protocols.wrap_interaction", _wrap, raising=True)
    monkeypatch.setattr(InviteCog, "_invite_impl", impl, raising=True)
    name_inv = "invite"
    inv_attr: _HasCallback[_InviteCB] = getattr(cog, name_inv)
    cb: _InviteCB = inv_attr.callback
    await cb(cog, RecordingInteraction())
    assert called["n"] == 1


@pytest.mark.asyncio
async def test_qr_wrapper_callback_invokes_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_settings(qr_api_url="http://api")
    from tests.support.discord_fakes import FakeBot

    bot2 = FakeBot()
    cog = QRCog(bot2, cfg, QRService(cfg))
    called: dict[str, int] = {"n": 0}

    async def impl(self_obj: QRCog, _i: InteractionProto, _url: str) -> None:
        called["n"] += 1

    def _wrap2(_i: _DiscordInteraction) -> RecordingInteraction:
        return RecordingInteraction()

    monkeypatch.setattr("platform_discord.protocols.wrap_interaction", _wrap2, raising=True)
    monkeypatch.setattr(QRCog, "_qrcode_impl", impl, raising=True)
    name_qr = "qrcode"
    qr_attr: _HasCallback[_QRCB] = getattr(cog, name_qr)
    cb: _QRCB = qr_attr.callback
    await cb(cog, RecordingInteraction(), "https://x")
    assert called["n"] == 1


@pytest.mark.asyncio
async def test_digits_read_and_train_wrappers_invoke_impls(monkeypatch: pytest.MonkeyPatch) -> None:
    from clubbot.services.digits.app import DigitService

    cfg = build_settings(handwriting_api_url="http://h")
    from tests.support.discord_fakes import FakeBot

    bot3 = FakeBot()
    svc = DigitService(cfg)
    cog = DigitsCog(bot3, cfg, svc, enqueuer=None, autostart_subscriber=False)
    calls: dict[str, int] = {"read": 0, "train": 0}

    async def impl_read(
        self_obj: DigitsCog, _i: InteractionProto, _user: UserProto, _img: discord.Attachment
    ) -> None:
        calls["read"] += 1

    async def impl_train(self_obj: DigitsCog, _i: InteractionProto, _user: UserProto) -> None:
        calls["train"] += 1

    def _wrap3(_i: _DiscordInteraction) -> RecordingInteraction:
        return RecordingInteraction()

    monkeypatch.setattr("platform_discord.protocols.wrap_interaction", _wrap3, raising=True)
    monkeypatch.setattr(DigitsCog, "_read_impl", impl_read, raising=True)
    monkeypatch.setattr(DigitsCog, "_train_impl", impl_train, raising=True)

    name_read = "read"
    name_train = "train"
    read_attr: _HasCallback[_DigitsReadCB] = getattr(cog, name_read)
    train_attr: _HasCallback[_DigitsTrainCB] = getattr(cog, name_train)
    read_cb: _DigitsReadCB = read_attr.callback
    train_cb: _DigitsTrainCB = train_attr.callback

    att = FakeAttachment(filename="x.png", content_type="image/png", size=1, data=b"x")
    await read_cb(cog, RecordingInteraction(), att)
    await train_cb(cog, RecordingInteraction())
    assert calls == {"read": 1, "train": 1}


@pytest.mark.asyncio
async def test_trainer_and_transcript_wrappers_invoke_impls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_settings(model_trainer_api_url="http://t")
    tcfg = build_settings(transcript_api_url="http://x")
    tsvc = TranscriptService(tcfg)
    from tests.support.discord_fakes import FakeBot

    fb = FakeBot()
    trainer = TrainerCog(bot=fb, config=cfg)
    transcript = TranscriptCog(bot=fb, config=tcfg, transcript_service=tsvc)
    counts: dict[str, int] = {"train": 0, "transcript": 0}

    async def impl_train(
        self_obj: TrainerCog,
        _i: InteractionProto,
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
        counts["train"] += 1

    async def impl_transcript(
        self_obj: TranscriptCog,
        _i: InteractionProto,
        _user: UserProto,
        _guild: UserProto | None,
        _url: str,
    ) -> None:
        counts["transcript"] += 1

    def _wrap4(_i: _DiscordInteraction) -> RecordingInteraction:
        return RecordingInteraction()

    monkeypatch.setattr("platform_discord.protocols.wrap_interaction", _wrap4, raising=True)
    monkeypatch.setattr(TrainerCog, "_train_model_impl", impl_train, raising=True)
    monkeypatch.setattr(TranscriptCog, "_transcript_impl", impl_transcript, raising=True)

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
    assert counts == {"train": 1, "transcript": 1}
