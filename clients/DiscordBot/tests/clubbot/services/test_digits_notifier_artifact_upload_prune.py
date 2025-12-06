from __future__ import annotations

import logging

import pytest
from platform_core.digits_metrics_events import (
    DigitsArtifactV1,
    DigitsPruneV1,
    DigitsUploadV1,
)
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto, MessageProto

import clubbot.services.jobs.digits_notifier as dn


class _User:
    def __init__(self) -> None:
        self.embeds: list[EmbedProto | None] = []

    @property
    def id(self) -> int:
        return 12345

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        _ = (content, file)
        self.embeds.append(embed)

        class _Msg:
            def __init__(self, u: _User) -> None:
                self._u = u

            @property
            def id(self) -> int:
                return 1

            async def edit(
                self, *, content: str | None = None, embed: EmbedProto | None = None
            ) -> MessageProto:
                _ = content
                self._u.embeds.append(embed)
                return self

        return _Msg(self)


class _Bot:
    def __init__(self) -> None:
        self.user = _User()

    async def fetch_user(self, user_id: int, /) -> _User:
        _ = user_id
        return self.user


@pytest.mark.asyncio
async def test_handle_artifact_event_noop() -> None:
    """Test that artifact events are handled without crashing (no-op)."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    artifact: DigitsArtifactV1 = {
        "type": "digits.metrics.artifact.v1",
        "job_id": "r_art",
        "user_id": 200,
        "model_id": "mnist",
        "path": "/artifacts/digits/models/mnist_resnet18_v1",
    }
    await sub._handle_event(artifact)
    # Artifact handler is a no-op, should not send messages
    assert len(bot.user.embeds) == 0


@pytest.mark.asyncio
async def test_handle_upload_event_noop() -> None:
    """Test that upload events are handled without crashing (no-op)."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    upload: DigitsUploadV1 = {
        "type": "digits.metrics.upload.v1",
        "job_id": "r_upload",
        "user_id": 201,
        "model_id": "mnist",
        "status": 200,
        "model_bytes": 45678901,
        "manifest_bytes": 1234,
        "file_id": "fid",
        "file_sha256": "sha",
    }
    await sub._handle_event(upload)
    # Upload handler is a no-op, should not send messages
    assert len(bot.user.embeds) == 0


@pytest.mark.asyncio
async def test_handle_upload_event_with_error_status() -> None:
    """Test that upload events with error status are handled (no-op currently)."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    upload: DigitsUploadV1 = {
        "type": "digits.metrics.upload.v1",
        "job_id": "r_upload_err",
        "user_id": 202,
        "model_id": "mnist",
        "status": 500,  # Error status
        "model_bytes": 0,
        "manifest_bytes": 0,
        "file_id": "fid",
        "file_sha256": "sha",
    }
    await sub._handle_event(upload)
    # Upload handler is a no-op even for errors, should not send messages
    assert len(bot.user.embeds) == 0


@pytest.mark.asyncio
async def test_handle_prune_event_noop() -> None:
    """Test that prune events are handled without crashing (no-op)."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    prune: DigitsPruneV1 = {
        "type": "digits.metrics.prune.v1",
        "job_id": "r_prune",
        "user_id": 203,
        "model_id": "mnist",
        "deleted_count": 3,
    }
    await sub._handle_event(prune)
    # Prune handler is a no-op, should not send messages
    assert len(bot.user.embeds) == 0


@pytest.mark.asyncio
async def test_handle_prune_event_zero_deleted() -> None:
    """Test that prune events with zero deletions are handled (no-op)."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    prune: DigitsPruneV1 = {
        "type": "digits.metrics.prune.v1",
        "job_id": "r_prune_zero",
        "user_id": 204,
        "model_id": "mnist",
        "deleted_count": 0,
    }
    await sub._handle_event(prune)
    # Prune handler is a no-op, should not send messages
    assert len(bot.user.embeds) == 0


def test_on_artifact_direct_call() -> None:
    """Test direct call to _on_artifact method."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    artifact: DigitsArtifactV1 = {
        "type": "digits.metrics.artifact.v1",
        "user_id": 205,
        "job_id": "r_art_direct",
        "model_id": "m",
        "path": "/path/to/artifact",
    }
    sub._on_artifact(artifact)
    # Should not crash and not send messages
    assert len(bot.user.embeds) == 0


def test_on_upload_direct_call() -> None:
    """Test direct call to _on_upload method."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    upload: DigitsUploadV1 = {
        "type": "digits.metrics.upload.v1",
        "user_id": 206,
        "job_id": "r_upload_direct",
        "model_id": "m",
        "status": 200,
        "model_bytes": 1000,
        "manifest_bytes": 100,
        "file_id": "fid",
        "file_sha256": "sha",
    }
    sub._on_upload(upload)
    # Should not crash and not send messages
    assert len(bot.user.embeds) == 0


def test_on_prune_direct_call() -> None:
    """Test direct call to _on_prune method."""
    bot = _Bot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    prune: DigitsPruneV1 = {
        "type": "digits.metrics.prune.v1",
        "user_id": 207,
        "job_id": "r_prune_direct",
        "model_id": "m",
        "deleted_count": 5,
    }
    sub._on_prune(prune)
    # Should not crash and not send messages
    assert len(bot.user.embeds) == 0


logger = logging.getLogger(__name__)
