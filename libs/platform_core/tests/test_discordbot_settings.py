from __future__ import annotations

import pytest

from platform_core.config.discordbot import (
    load_discordbot_settings,
    require_discord_token,
)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all discordbot-related environment variables."""
    keys = [
        "DISCORD_TOKEN",
        "DISCORD_GUILD_ID",
        "DISCORD_GUILD_IDS",
        "LOG_LEVEL",
        "COMMANDS_SYNC_GLOBAL",
        "QR_API_URL",
        "QRCODE_RATE_LIMIT",
        "QRCODE_RATE_WINDOW_SECONDS",
        "QR_DEFAULT_ERROR_CORRECTION",
        "QR_DEFAULT_BOX_SIZE",
        "QR_DEFAULT_BORDER",
        "QR_DEFAULT_FILL_COLOR",
        "QR_DEFAULT_BACK_COLOR",
        "QR_PUBLIC_RESPONSES",
        "TRANSCRIPT_API_URL",
        "TRANSCRIPT_PROVIDER",
        "TRANSCRIPT_PUBLIC_RESPONSES",
        "TRANSCRIPT_RATE_LIMIT",
        "TRANSCRIPT_RATE_WINDOW_SECONDS",
        "TRANSCRIPT_PREFERRED_LANGS",
        "TRANSCRIPT_MAX_MESSAGE_CHARS",
        "TRANSCRIPT_MAX_FILE_MB",
        "TRANSCRIPT_STT_RTF",
        "TRANSCRIPT_DL_MIB_PER_SEC",
        "TRANSCRIPT_STT_API_TIMEOUT_SECONDS",
        "TRANSCRIPT_STT_API_MAX_RETRIES",
        "TRANSCRIPT_MAX_ATTACHMENT_MB",
        "TRANSCRIPT_ESTIMATED_TEXT_KB_PER_MIN",
        "TRANSCRIPT_ENABLE_CHUNKING",
        "TRANSCRIPT_CHUNK_THRESHOLD_MB",
        "TRANSCRIPT_TARGET_CHUNK_MB",
        "TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS",
        "TRANSCRIPT_MAX_CONCURRENT_CHUNKS",
        "TRANSCRIPT_SILENCE_THRESHOLD_DB",
        "TRANSCRIPT_SILENCE_DURATION_SECONDS",
        "TRANSCRIPT_COOKIES_TEXT",
        "TRANSCRIPT_COOKIES_PATH",
        "YOUTUBE_API_KEY",
        "OPENAI_API_KEY",
        "OPEN_AI_API_KEY",
        "REDIS_URL",
        "JOB_QUEUE_BRPOP_TIMEOUT_SECONDS",
        "RQ_TRANSCRIPT_JOB_TIMEOUT_SEC",
        "RQ_TRANSCRIPT_RESULT_TTL_SEC",
        "RQ_TRANSCRIPT_FAILURE_TTL_SEC",
        "RQ_TRANSCRIPT_RETRY_MAX",
        "RQ_TRANSCRIPT_RETRY_INTERVALS_SEC",
        "HANDWRITING_API_URL",
        "HANDWRITING_API_KEY",
        "HANDWRITING_API_TIMEOUT_SECONDS",
        "HANDWRITING_API_MAX_RETRIES",
        "DIGITS_PUBLIC_RESPONSES",
        "DIGITS_RATE_LIMIT",
        "DIGITS_RATE_WINDOW_SECONDS",
        "DIGITS_MAX_IMAGE_MB",
        "MODEL_TRAINER_API_URL",
        "MODEL_TRAINER_API_KEY",
        "MODEL_TRAINER_API_TIMEOUT_SECONDS",
        "MODEL_TRAINER_API_MAX_RETRIES",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_load_discordbot_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)

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


def test_load_discordbot_settings_single_guild_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("DISCORD_GUILD_ID", "123456789")

    settings = load_discordbot_settings()

    assert settings["discord"]["guild_id"] == "123456789"
    assert settings["discord"]["guild_ids"] == [123456789]


def test_load_discordbot_settings_multiple_guild_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("DISCORD_GUILD_IDS", "111,222,333")

    settings = load_discordbot_settings()

    assert settings["discord"]["guild_ids"] == [111, 222, 333]


def test_load_discordbot_settings_guild_ids_with_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("DISCORD_GUILD_IDS", "111 222 333")

    settings = load_discordbot_settings()

    assert settings["discord"]["guild_ids"] == [111, 222, 333]


def test_load_discordbot_settings_guild_ids_filters_non_digits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("DISCORD_GUILD_IDS", "111,invalid,222,not-a-number,333")

    settings = load_discordbot_settings()

    assert settings["discord"]["guild_ids"] == [111, 222, 333]


def test_load_discordbot_settings_retry_intervals_custom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RQ_TRANSCRIPT_RETRY_INTERVALS_SEC", "30,120")

    settings = load_discordbot_settings()

    assert settings["redis"]["rq_transcript_retry_intervals_sec"] == (30, 120)


def test_load_discordbot_settings_retry_intervals_with_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RQ_TRANSCRIPT_RETRY_INTERVALS_SEC", "30 120")

    settings = load_discordbot_settings()

    assert settings["redis"]["rq_transcript_retry_intervals_sec"] == (30, 120)


def test_load_discordbot_settings_retry_intervals_invalid_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RQ_TRANSCRIPT_RETRY_INTERVALS_SEC", "invalid")

    settings = load_discordbot_settings()

    assert settings["redis"]["rq_transcript_retry_intervals_sec"] == (60, 300)


def test_load_discordbot_settings_retry_intervals_insufficient_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RQ_TRANSCRIPT_RETRY_INTERVALS_SEC", "30")

    settings = load_discordbot_settings()

    assert settings["redis"]["rq_transcript_retry_intervals_sec"] == (60, 300)


def test_load_discordbot_settings_openai_key_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "primary-key")

    settings = load_discordbot_settings()

    assert settings["transcript"]["openai_api_key"] == "primary-key"


def test_load_discordbot_settings_openai_key_alt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPEN_AI_API_KEY", "alt-key")

    settings = load_discordbot_settings()

    assert settings["transcript"]["openai_api_key"] == "alt-key"


def test_load_discordbot_settings_openai_key_primary_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "primary-key")
    monkeypatch.setenv("OPEN_AI_API_KEY", "alt-key")

    settings = load_discordbot_settings()

    assert settings["transcript"]["openai_api_key"] == "primary-key"


def test_load_discordbot_settings_all_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)

    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("COMMANDS_SYNC_GLOBAL", "true")
    monkeypatch.setenv("QR_API_URL", "http://qr-api")
    monkeypatch.setenv("TRANSCRIPT_API_URL", "http://transcript-api")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("HANDWRITING_API_URL", "http://handwriting-api")
    monkeypatch.setenv("HANDWRITING_API_KEY", "handwriting-key")
    monkeypatch.setenv("MODEL_TRAINER_API_URL", "http://trainer-api")
    monkeypatch.setenv("MODEL_TRAINER_API_KEY", "trainer-key")

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


def test_require_discord_token_raises_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    settings = load_discordbot_settings()

    with pytest.raises(RuntimeError, match="DISCORD_TOKEN is required"):
        require_discord_token(settings)


def test_require_discord_token_succeeds_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    settings = load_discordbot_settings()

    require_discord_token(settings)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
