from __future__ import annotations

import pytest
from tests.support.discord_fakes import FakeBot
from tests.support.settings import build_settings

from clubbot.cogs.trainer import TrainerCog


def test_trainer_ensure_subscriber_started_invokes_start(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_settings(
        model_trainer_api_url="https://example",
        model_trainer_api_key=None,
        model_trainer_api_timeout_seconds=10,
        redis_url="redis://example",
    )

    class _Sub:
        def __init__(self) -> None:
            self.n = 0

        def start(self) -> None:
            self.n += 1

    # Build without autostart
    cog = TrainerCog(bot=FakeBot(), config=cfg, autostart_subscriber=False)
    # Inject a fake subscriber and ensure the helper starts it
    fake = _Sub()
    object.__setattr__(cog, "_subscriber", fake)
    cog.ensure_subscriber_started()
    assert fake.n == 1


def test_trainer_autostart_with_none_bot_noop() -> None:
    cfg = build_settings(
        model_trainer_api_url="https://example",
        model_trainer_api_key=None,
        model_trainer_api_timeout_seconds=10,
        redis_url="redis://example",
    )
    _ = TrainerCog(bot=None, config=cfg)


def test_trainer_ensure_subscriber_started_noop_on_none() -> None:
    cfg = build_settings(
        model_trainer_api_url="https://example",
        model_trainer_api_key=None,
        model_trainer_api_timeout_seconds=10,
        redis_url=None,
    )
    cog = TrainerCog(bot=FakeBot(), config=cfg, autostart_subscriber=False)
    cog.ensure_subscriber_started()
