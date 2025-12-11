from __future__ import annotations

import pytest

from platform_core.config.discordbot import (
    load_discordbot_settings,
    require_discord_token,
)
from platform_core.testing import make_fake_env


def test_load_discordbot_settings_defaults() -> None:
    make_fake_env()
    settings = load_discordbot_settings()

    assert settings["discord"]["token"] == ""
    assert settings["discord"]["guild_id"] is None
    assert settings["discord"]["guild_ids"] == []
    assert settings["discord"]["log_level"] == "INFO"
    assert settings["discord"]["commands_sync_global"] is False

    assert settings["qr"]["api_url"] == ""
    assert settings["qr"]["rate_limit"] == 1
    assert settings["qr"]["rate_window_seconds"] == 1
    assert settings["qr"]["default_error_correction"] == "M"
    assert settings["qr"]["default_box_size"] == 10
    assert settings["qr"]["default_border"] == 1
    assert settings["qr"]["default_fill_color"] == "#000000"
    assert settings["qr"]["default_back_color"] == "#FFFFFF"
    assert settings["qr"]["public_responses"] is True

    assert settings["transcript"]["api_url"] == "http://localhost:8000"
    assert settings["transcript"]["provider"] == "api"
    assert settings["transcript"]["public_responses"] is False
    assert settings["transcript"]["rate_limit"] == 2
    assert settings["transcript"]["rate_window_seconds"] == 60
    assert settings["transcript"]["preferred_langs"] == "en,en-US,en-GB"
    assert settings["transcript"]["max_message_chars"] == 1800
    assert settings["transcript"]["max_video_seconds"] == 5400
    assert settings["transcript"]["max_file_mb"] == 25
    assert settings["transcript"]["stt_rtf"] == 0.5
    assert settings["transcript"]["dl_mib_per_sec"] == 4.0
    assert settings["transcript"]["stt_api_timeout_seconds"] == 900
    assert settings["transcript"]["stt_api_max_retries"] == 2
    assert settings["transcript"]["max_attachment_mb"] == 25
    assert settings["transcript"]["estimated_text_kb_per_min"] == 1.0
    assert settings["transcript"]["enable_chunking"] is True
    assert settings["transcript"]["chunk_threshold_mb"] == 20.0
    assert settings["transcript"]["target_chunk_mb"] == 20.0
    assert settings["transcript"]["max_chunk_duration_seconds"] == 600.0
    assert settings["transcript"]["max_concurrent_chunks"] == 3
    assert settings["transcript"]["silence_threshold_db"] == -40.0
    assert settings["transcript"]["silence_duration_seconds"] == 0.5
    assert settings["transcript"]["cookies_text"] is None
    assert settings["transcript"]["cookies_path"] is None
    assert settings["transcript"]["youtube_api_key"] is None
    assert settings["transcript"]["openai_api_key"] is None

    assert settings["redis"]["redis_url"] is None
    assert settings["redis"]["job_queue_brpop_timeout_seconds"] == 0
    assert settings["redis"]["rq_transcript_job_timeout_sec"] == 600
    assert settings["redis"]["rq_transcript_result_ttl_sec"] == 86400
    assert settings["redis"]["rq_transcript_failure_ttl_sec"] == 604800
    assert settings["redis"]["rq_transcript_retry_max"] == 2
    assert settings["redis"]["rq_transcript_retry_intervals_sec"] == (60, 300)

    assert settings["handwriting"]["api_url"] is None
    assert settings["handwriting"]["api_key"] is None
    assert settings["handwriting"]["api_timeout_seconds"] == 5
    assert settings["handwriting"]["api_max_retries"] == 1

    assert settings["digits"]["public_responses"] is False
    assert settings["digits"]["rate_limit"] == 2
    assert settings["digits"]["rate_window_seconds"] == 60
    assert settings["digits"]["max_image_mb"] == 2

    assert settings["model_trainer"]["api_url"] is None
    assert settings["model_trainer"]["api_key"] is None
    assert settings["model_trainer"]["api_timeout_seconds"] == 10
    assert settings["model_trainer"]["api_max_retries"] == 1


def test_load_discordbot_settings_single_guild_id() -> None:
    env = make_fake_env()
    env.set("DISCORD_GUILD_ID", "123456789")

    settings = load_discordbot_settings()

    assert settings["discord"]["guild_id"] == "123456789"
    assert settings["discord"]["guild_ids"] == [123456789]


def test_load_discordbot_settings_multiple_guild_ids() -> None:
    env = make_fake_env()
    env.set("DISCORD_GUILD_IDS", "111,222,333")

    settings = load_discordbot_settings()

    assert settings["discord"]["guild_ids"] == [111, 222, 333]


def test_load_discordbot_settings_guild_ids_with_spaces() -> None:
    env = make_fake_env()
    env.set("DISCORD_GUILD_IDS", "111 222 333")

    settings = load_discordbot_settings()

    assert settings["discord"]["guild_ids"] == [111, 222, 333]


def test_load_discordbot_settings_guild_ids_filters_non_digits() -> None:
    env = make_fake_env()
    env.set("DISCORD_GUILD_IDS", "111,invalid,222,not-a-number,333")

    settings = load_discordbot_settings()

    assert settings["discord"]["guild_ids"] == [111, 222, 333]


def test_load_discordbot_settings_retry_intervals_custom() -> None:
    env = make_fake_env()
    env.set("RQ_TRANSCRIPT_RETRY_INTERVALS_SEC", "30,120")

    settings = load_discordbot_settings()

    assert settings["redis"]["rq_transcript_retry_intervals_sec"] == (30, 120)


def test_load_discordbot_settings_retry_intervals_with_spaces() -> None:
    env = make_fake_env()
    env.set("RQ_TRANSCRIPT_RETRY_INTERVALS_SEC", "30 120")

    settings = load_discordbot_settings()

    assert settings["redis"]["rq_transcript_retry_intervals_sec"] == (30, 120)


def test_load_discordbot_settings_retry_intervals_invalid_defaults() -> None:
    env = make_fake_env()
    env.set("RQ_TRANSCRIPT_RETRY_INTERVALS_SEC", "invalid")

    settings = load_discordbot_settings()

    assert settings["redis"]["rq_transcript_retry_intervals_sec"] == (60, 300)


def test_load_discordbot_settings_retry_intervals_insufficient_parts() -> None:
    env = make_fake_env()
    env.set("RQ_TRANSCRIPT_RETRY_INTERVALS_SEC", "30")

    settings = load_discordbot_settings()

    assert settings["redis"]["rq_transcript_retry_intervals_sec"] == (60, 300)


def test_load_discordbot_settings_openai_key_primary() -> None:
    env = make_fake_env()
    env.set("OPENAI_API_KEY", "primary-key")

    settings = load_discordbot_settings()

    assert settings["transcript"]["openai_api_key"] == "primary-key"


def test_load_discordbot_settings_openai_key_alt() -> None:
    env = make_fake_env()
    env.set("OPEN_AI_API_KEY", "alt-key")

    settings = load_discordbot_settings()

    assert settings["transcript"]["openai_api_key"] == "alt-key"


def test_load_discordbot_settings_openai_key_primary_takes_precedence() -> None:
    env = make_fake_env()
    env.set("OPENAI_API_KEY", "primary-key")
    env.set("OPEN_AI_API_KEY", "alt-key")

    settings = load_discordbot_settings()

    assert settings["transcript"]["openai_api_key"] == "primary-key"


def test_load_discordbot_settings_all_overrides() -> None:
    env = make_fake_env()
    env.set("DISCORD_TOKEN", "test-token")
    env.set("LOG_LEVEL", "debug")
    env.set("COMMANDS_SYNC_GLOBAL", "true")
    env.set("QR_API_URL", "http://qr-api")
    env.set("TRANSCRIPT_API_URL", "http://transcript-api")
    env.set("REDIS_URL", "redis://localhost:6379/1")
    env.set("HANDWRITING_API_URL", "http://handwriting-api")
    env.set("HANDWRITING_API_KEY", "handwriting-key")
    env.set("MODEL_TRAINER_API_URL", "http://trainer-api")
    env.set("MODEL_TRAINER_API_KEY", "trainer-key")

    settings = load_discordbot_settings()

    assert settings["discord"]["token"] == "test-token"
    assert settings["discord"]["log_level"] == "DEBUG"
    assert settings["discord"]["commands_sync_global"] is True
    assert settings["qr"]["api_url"] == "http://qr-api"
    assert settings["transcript"]["api_url"] == "http://transcript-api"
    assert settings["redis"]["redis_url"] == "redis://localhost:6379/1"
    assert settings["handwriting"]["api_url"] == "http://handwriting-api"
    assert settings["handwriting"]["api_key"] == "handwriting-key"
    assert settings["model_trainer"]["api_url"] == "http://trainer-api"
    assert settings["model_trainer"]["api_key"] == "trainer-key"


def test_require_discord_token_raises_when_empty() -> None:
    make_fake_env()
    settings = load_discordbot_settings()

    with pytest.raises(RuntimeError, match="DISCORD_TOKEN is required"):
        require_discord_token(settings)


def test_require_discord_token_succeeds_when_present() -> None:
    env = make_fake_env()
    env.set("DISCORD_TOKEN", "test-token")
    settings = load_discordbot_settings()

    require_discord_token(settings)
