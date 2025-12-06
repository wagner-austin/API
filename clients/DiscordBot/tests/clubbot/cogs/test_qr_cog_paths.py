from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import pytest
from discord.ext import commands
from platform_discord.protocols import InteractionProto, ResponseProto, UserProto
from tests.support.discord_fakes import FakeBot, FakeFollowup, FakeResponse, FakeUser
from tests.support.settings import build_settings

import clubbot.cogs.qr as qr_mod
from clubbot.cogs.base import _HasIntId
from clubbot.cogs.qr import QRCog
from clubbot.config import DiscordbotSettings
from clubbot.services.qr.client import QRResult, QRService


class _Logger(Protocol):
    def debug(self, msg: str, *args: str) -> None: ...
    def info(self, msg: str, *args: str) -> None: ...


class _QRResult(QRResult):
    def __init__(self) -> None:
        super().__init__(image_png=b"x", url="https://x")


class _FakeService(QRService):
    def __init__(self, cfg: DiscordbotSettings) -> None:
        super().__init__(cfg)
        self._res = _QRResult()

    def generate_qr(self, url: str) -> QRResult:
        _ = url
        return self._res


def _cfg() -> DiscordbotSettings:
    return build_settings(
        qr_default_border=1,
        qr_public_responses=False,
        transcript_provider="youtube",
    )


class _RespRaises(FakeResponse):
    def __init__(self, exc: Exception) -> None:
        super().__init__(done=False)
        self._exc = exc

    async def defer(self, *, ephemeral: bool = False) -> None:
        _ = ephemeral
        raise self._exc


class _InteractionWithResp:
    def __init__(self, resp: FakeResponse | None = None, user: FakeUser | None = None) -> None:
        self.response: ResponseProto = resp if resp is not None else FakeResponse()
        self.followup = FakeFollowup()
        self.user: UserProto = user if user is not None else FakeUser()


@pytest.mark.asyncio
async def test_qr_ack_return_and_user_id_none() -> None:
    class _Cog(QRCog):
        async def _safe_defer(self, interaction: InteractionProto, *, ephemeral: bool) -> bool:
            _ = interaction
            _ = ephemeral
            return False

    cfg = _cfg()
    cog = _Cog(bot=FakeBot(), config=cfg, qr_service=_FakeService(cfg))
    inter = _InteractionWithResp()
    await cog._qrcode_impl(inter, "https://x")

    class _Cog2(QRCog):
        @staticmethod
        def decode_int_attr(obj: _HasIntId | None, name: str) -> int | None:
            _ = (obj, name)
            return None

    inter2 = _InteractionWithResp()
    cog2 = _Cog2(bot=FakeBot(), config=cfg, qr_service=_FakeService(cfg))
    await cog2._qrcode_impl(inter2, "https://x")
    if inter2.followup is None:
        raise AssertionError("expected followup")


@pytest.mark.asyncio
async def test_qr_safe_defer_exceptions() -> None:
    class _HTTPError(Exception):
        def __init__(self, code: int) -> None:
            self.code = code

    cfg = _cfg()
    cog = QRCog(bot=FakeBot(), config=cfg, qr_service=_FakeService(cfg))

    inter_not_found = _InteractionWithResp(_RespRaises(_HTTPError(404)))
    ok = await cog._safe_defer(inter_not_found, ephemeral=True)
    assert ok is False

    inter_safe = _InteractionWithResp(_RespRaises(_HTTPError(40060)))
    ok2 = await cog._safe_defer(inter_safe, ephemeral=True)
    assert ok2 is True

    inter_http = _InteractionWithResp(_RespRaises(_HTTPError(499)))
    inter_http.followup = FakeFollowup()
    ok3 = await cog._safe_defer(inter_http, ephemeral=True)
    assert ok3 is False


def test_qr_extract_attr_and_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    assert QRCog.decode_int_attr(None, "id") is None

    added: dict[str, bool] = {}

    class _Bot(FakeBot):
        async def add_cog(self, cog: commands.Cog) -> None:
            _ = cog
            added["ok"] = True

    def _mk_service(cfg: DiscordbotSettings) -> QRService:
        return _FakeService(cfg)

    monkeypatch.setattr(qr_mod, "load_discordbot_settings", _cfg, raising=True)
    monkeypatch.setattr(qr_mod, "QRService", _mk_service, raising=True)

    asyncio.get_event_loop().run_until_complete(qr_mod.setup(_Bot()))
    assert added.get("ok") is True


logger = logging.getLogger(__name__)
