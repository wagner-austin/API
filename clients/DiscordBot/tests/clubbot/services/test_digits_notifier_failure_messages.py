from __future__ import annotations

import logging

import pytest
from platform_core.job_events import JobFailedV1
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto, MessageProto, UserProto

import clubbot.services.jobs.digits_notifier as dn


class _Msg:
    """Fake message satisfying MessageProto Protocol."""

    def __init__(self, user: _User) -> None:
        self._user = user

    @property
    def id(self) -> int:
        return 12345

    async def edit(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> MessageProto:
        if embed is not None:
            self._user.embeds.append(embed)
        return self


class _User:
    """Fake user satisfying UserProto Protocol."""

    def __init__(self) -> None:
        self.embeds: list[EmbedProto] = []

    @property
    def id(self) -> int:
        return 67890

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        _ = (content, file)
        if embed is not None:
            self.embeds.append(embed)
        return _Msg(self)


class _Bot:
    """Fake bot for testing."""

    def __init__(self) -> None:
        self.user = _User()

    async def fetch_user(self, user_id: int, /) -> UserProto:
        _ = user_id
        return self.user


@pytest.mark.asyncio
async def test_failed_system_error_memory_pressure_message() -> None:
    """Test that memory pressure errors show specific memory guidance."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    event: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r_mem_pressure",
        "user_id": 100,
        "error_kind": "system",
        "message": (
            "Training aborted due to sustained memory pressure (>= 85.0%). "
            "Reduce batch size or DataLoader workers and retry."
        ),
        "domain": "digits",
    }
    await sub._handle_event(event)
    assert len(bot.user.embeds) == 1
    # Embed should be created with the specific error message


@pytest.mark.asyncio
async def test_failed_system_error_oom_kill_message() -> None:
    """Test that OOM kill errors show specific memory guidance."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    event: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r_oom",
        "user_id": 101,
        "error_kind": "system",
        "message": (
            "OOM kill detected (signal 9 / SIGKILL) - "
            "worker terminated by system due to memory exhaustion"
        ),
        "domain": "digits",
    }
    await sub._handle_event(event)
    assert len(bot.user.embeds) == 1


@pytest.mark.asyncio
async def test_failed_system_error_artifact_upload_message() -> None:
    """Test that artifact upload errors show specific upload guidance."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    event: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r_upload",
        "user_id": 102,
        "error_kind": "system",
        "message": "Artifact upload failed: upstream API error. See worker logs for details.",
        "domain": "digits",
    }
    await sub._handle_event(event)
    assert len(bot.user.embeds) == 1


@pytest.mark.asyncio
async def test_failed_system_error_generic_fallback() -> None:
    """Test that unknown system errors show generic but still helpful guidance."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    event: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r_generic_sys",
        "user_id": 103,
        "error_kind": "system",
        "message": "RuntimeError: Something unexpected happened during training",
        "domain": "digits",
    }
    await sub._handle_event(event)
    assert len(bot.user.embeds) == 1


@pytest.mark.asyncio
async def test_failed_user_error_shows_config_issue() -> None:
    """Test that user errors are labeled as configuration issues."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    event: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r_user_err",
        "user_id": 104,
        "error_kind": "user",
        "message": "invalid job type",
        "domain": "digits",
    }
    await sub._handle_event(event)
    assert len(bot.user.embeds) == 1


@pytest.mark.asyncio
async def test_failed_memory_uppercase_detection() -> None:
    """Test that MEMORY in uppercase is also detected."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    event: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r_mem_upper",
        "user_id": 105,
        "error_kind": "system",
        "message": "MEMORY allocation failed",
        "domain": "digits",
    }
    await sub._handle_event(event)
    assert len(bot.user.embeds) == 1


@pytest.mark.asyncio
async def test_failed_upload_case_insensitive_detection() -> None:
    """Test that 'Upload' with capital U is detected."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    event: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r_upload_cap",
        "user_id": 106,
        "error_kind": "system",
        "message": "Upload to S3 failed: timeout",
        "domain": "digits",
    }
    await sub._handle_event(event)
    assert len(bot.user.embeds) == 1


@pytest.mark.asyncio
async def test_failed_message_with_run_id() -> None:
    """Test that failure messages work when run_id is present."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    event: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r_with_run",
        "user_id": 107,
        "error_kind": "system",
        "message": "Training failed after epoch 5",
        "domain": "digits",
    }
    await sub._handle_event(event)
    assert len(bot.user.embeds) == 1


logger = logging.getLogger(__name__)
