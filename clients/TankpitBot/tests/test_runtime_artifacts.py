"""Tests for canonical runtime artifact path builders."""

from __future__ import annotations

from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from tankpit_bot.runtime_artifacts import (
    build_bot_run_artifacts,
    build_probe_run_artifacts,
    build_sniff_run_artifacts,
    decode_bot_run_artifacts,
    decode_probe_run_artifacts,
    decode_sniff_run_artifacts,
    encode_bot_run_artifacts,
    encode_probe_run_artifacts,
    encode_sniff_run_artifacts,
    resolve_bot_instance,
)


def test_build_bot_run_artifacts() -> None:
    """Sole-bot artifacts use stable latest paths plus timestamped archives."""
    artifacts = build_bot_run_artifacts("20260331-230405", "")

    assert artifacts["log_dir"] == "runs\\bot"
    assert artifacts["latest_log_path"] == "runs\\bot\\latest.log"
    assert artifacts["archive_log_path"] == "runs\\bot\\bot-20260331-230405.log"
    assert artifacts["latest_events_path"] == "runs\\bot\\latest.events.jsonl"
    assert artifacts["archive_events_path"] == "runs\\bot\\bot-20260331-230405.events.jsonl"


def test_instance_namespaces_every_bot_artifact() -> None:
    """A named instance nests the whole bundle under its own directory.

    The two-bots-one-map lift (2026-08-06): parallel processes must
    never overwrite each other's latest.* files or captures.
    """
    artifacts = build_bot_run_artifacts("20260331-230405", "alpha")

    assert artifacts["log_dir"] == "runs\\bot\\alpha"
    assert artifacts["latest_log_path"] == "runs\\bot\\alpha\\latest.log"
    assert artifacts["archive_log_path"] == "runs\\bot\\alpha\\bot-20260331-230405.log"
    assert artifacts["latest_capture_path"] == "runs\\bot\\alpha\\latest.capture_session.json"


def test_resolve_bot_instance_env_contract() -> None:
    """Unset/empty is the sole-bot namespace; bad names are loud errors."""
    import pytest

    from tankpit_bot import _test_hooks
    from tests.conftest import FakeEnv

    original_get_env = _test_hooks.get_env
    try:
        _test_hooks.get_env = FakeEnv({})
        assert resolve_bot_instance() == ""
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_INSTANCE": "alpha-2"})
        assert resolve_bot_instance() == "alpha-2"
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_INSTANCE": "../escape"})
        with pytest.raises(ValueError, match="not a valid instance name"):
            resolve_bot_instance()
        _test_hooks.get_env = FakeEnv({"TANKPIT_BOT_INSTANCE": "UPPER"})
        with pytest.raises(ValueError, match="not a valid instance name"):
            resolve_bot_instance()
    finally:
        _test_hooks.get_env = original_get_env


def test_build_sniff_run_artifacts() -> None:
    """Sniffer run artifacts include latest and archived capture outputs."""
    artifacts = build_sniff_run_artifacts("20260331-230405")

    assert artifacts["log_dir"] == "runs\\sniff"
    assert artifacts["latest_log_path"] == "runs\\sniff\\latest.log"
    assert artifacts["archive_log_path"] == "runs\\sniff\\sniff-20260331-230405.log"
    assert artifacts["latest_capture_path"] == "runs\\sniff\\latest.capture_session.json"
    assert artifacts["archive_capture_path"] == (
        "runs\\sniff\\sniff-20260331-230405.capture_session.json"
    )
    assert artifacts["latest_summary_path"] == "runs\\sniff\\latest.session_summary.json"
    assert artifacts["archive_summary_path"] == (
        "runs\\sniff\\sniff-20260331-230405.session_summary.json"
    )


def test_encode_decode_bot_run_artifacts_round_trip() -> None:
    """Bot run artifacts round-trip through JSON encoding."""
    artifacts = build_bot_run_artifacts("20260331-230405", "")

    encoded = encode_bot_run_artifacts(artifacts)
    decoded = decode_bot_run_artifacts(narrow_json_to_dict(load_json_str(dump_json_str(encoded))))

    assert decoded == artifacts


def test_encode_decode_sniff_run_artifacts_round_trip() -> None:
    """Sniffer run artifacts round-trip through JSON encoding."""
    artifacts = build_sniff_run_artifacts("20260331-230405")

    encoded = encode_sniff_run_artifacts(artifacts)
    decoded = decode_sniff_run_artifacts(narrow_json_to_dict(load_json_str(dump_json_str(encoded))))

    assert decoded == artifacts


def test_build_probe_run_artifacts() -> None:
    """Probe run artifacts include the probe name in latest+archive filenames."""
    artifacts = build_probe_run_artifacts("fuel", "20260331-230405")

    assert artifacts["log_dir"] == "runs\\probe"
    assert artifacts["probe_name"] == "fuel"
    assert artifacts["latest_log_path"] == "runs\\probe\\latest.fuel.log"
    assert artifacts["archive_log_path"] == "runs\\probe\\fuel-20260331-230405.log"
    assert artifacts["latest_events_path"] == "runs\\probe\\latest.fuel.events.jsonl"
    assert artifacts["archive_events_path"] == ("runs\\probe\\fuel-20260331-230405.events.jsonl")


def test_build_probe_run_artifacts_rejects_empty_name() -> None:
    """Probe name must be non-empty -- prevents nameless filenames in runs/probe/."""
    import pytest

    with pytest.raises(ValueError, match="probe_name must be non-empty"):
        build_probe_run_artifacts("", "20260331-230405")


def test_encode_decode_probe_run_artifacts_round_trip() -> None:
    """Probe run artifacts round-trip through JSON encoding."""
    artifacts = build_probe_run_artifacts("equipment", "20260331-230405")

    encoded = encode_probe_run_artifacts(artifacts)
    decoded = decode_probe_run_artifacts(narrow_json_to_dict(load_json_str(dump_json_str(encoded))))

    assert decoded == artifacts
