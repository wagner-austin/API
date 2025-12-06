from __future__ import annotations

from tests.support.discord_fakes import FakeBot, FakeUser
from tests.support.settings import build_settings

from clubbot.cogs.trainer import TrainerCog


def test_decode_int_attr_unknown_name_returns_none() -> None:
    cfg = build_settings(model_trainer_api_url="http://t")
    _ = TrainerCog(bot=FakeBot(), config=cfg)

    assert TrainerCog.decode_int_attr(FakeUser(), "nope") is None
