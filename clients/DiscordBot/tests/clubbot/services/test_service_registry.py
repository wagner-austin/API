from __future__ import annotations

from platform_core.digits_metrics_events import DigitsConfigV1 as _DigitsConfig
from platform_core.job_events import make_completed_event
from platform_core.json_utils import dump_json_str
from platform_core.trainer_metrics_events import TrainerConfigV1

from clubbot.services.registry import SERVICE_REGISTRY


def test_service_registry_contains_expected_entries() -> None:
    assert set(SERVICE_REGISTRY.keys()) >= {"digits", "trainer", "transcript"}


def test_registry_decode_functions_work_minimally() -> None:
    # digits
    d_config: _DigitsConfig = {
        "type": "digits.metrics.config.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 1,
        "queue": "q",
    }
    dec_d = SERVICE_REGISTRY["digits"]["decode_event"]
    if dec_d(dump_json_str(d_config)) is None:
        raise AssertionError("expected digits decode result")

    # trainer - using new metrics config event
    t_config: TrainerConfigV1 = {
        "type": "trainer.metrics.config.v1",
        "job_id": "r",
        "user_id": 1,
        "model_family": "gpt2",
        "model_size": "small",
        "total_epochs": 1,
        "queue": "training",
    }
    dec_t = SERVICE_REGISTRY["trainer"]["decode_event"]
    if dec_t(dump_json_str(t_config)) is None:
        raise AssertionError("expected trainer decode result")

    # transcript
    c = make_completed_event(
        domain="transcript",
        job_id="r",
        user_id=1,
        result_id="https://x",
        result_bytes=10,
    )
    dec_tr = SERVICE_REGISTRY["transcript"]["decode_event"]
    if dec_tr(dump_json_str(c)) is None:
        raise AssertionError("expected transcript decode result")


def test_transcript_decode_filters_other_domains() -> None:
    other = make_completed_event(
        domain="turkic",
        job_id="r2",
        user_id=1,
        result_id="file",
        result_bytes=1,
    )
    dec_tr = SERVICE_REGISTRY["transcript"]["decode_event"]
    assert dec_tr(dump_json_str(other)) is None


def test_digits_decode_returns_none_on_invalid_json() -> None:
    """Test that digits decode returns None for invalid JSON."""
    dec_d = SERVICE_REGISTRY["digits"]["decode_event"]
    assert dec_d("not-valid-json") is None


def test_digits_decode_returns_none_on_value_error() -> None:
    """Test that digits decode returns None for events with missing required fields."""
    # Valid JSON, correct type but missing required fields triggers ValueError
    dec_d = SERVICE_REGISTRY["digits"]["decode_event"]
    bad_event = dump_json_str(
        {
            "type": "digits.metrics.config.v1",
            "job_id": "r",
            "user_id": 1,
            # missing model_id, total_epochs, queue - will raise ValueError
        }
    )
    assert dec_d(bad_event) is None
