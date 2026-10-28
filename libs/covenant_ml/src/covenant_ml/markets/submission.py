"""Shared durable demo submission and offline evidence verification."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote

from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_str,
)

from .demo_trial_records import validate_acknowledgement
from .execution_journal import execution_journal
from .execution_records import OrderIntent, encode_intent
from .execution_storage import persist_evidence
from .kalshi_transport import DemoTransport
from .submission_evidence import (
    decode_submission_capture,
    decode_submission_timing,
    encode_submission_capture,
    encode_submission_timing,
)
from .validation import parse_timestamp, read_document


def submit_reserved_order(
    transport: DemoTransport,
    clock: Callable[[], str],
    root: Path,
    ledger: Path,
    intent: OrderIntent,
    preflight_completed_ms: int,
) -> None:
    """Capture liquidity, reserve once, persist evidence and send one demo order.

    Args:
        transport: Demo-only signed HTTP boundary.
        clock: Aware local clock.
        root: Existing exclusive evidence directory for this order.
        ledger: Existing aggregate risk journal.
        intent: Validated one-contract order intent.
        preflight_completed_ms: Completion of verified account evidence.

    Raises:
        ValueError: Invalid quote, reservation, acknowledgement or timing.
        OSError: Evidence cannot be durably stored.
        HTTPError: Ambiguous submission failure; reservation remains unresolved.
    """
    ticker = quote(intent["ticker"], safe="")
    market = require_dict(transport.get("/markets/" + ticker, {}), "market")
    market_received = parse_timestamp(clock())
    book = transport.get("/markets/" + ticker + "/orderbook", {})
    book_received = parse_timestamp(clock())
    observed = parse_timestamp(clock())
    capture = decode_submission_capture(
        {
            "intent": encode_intent(intent),
            "market": market,
            "book": book,
            "preflight_completed_ms": preflight_completed_ms,
            "market_received_ms": market_received,
            "book_received_ms": book_received,
            "decision_ms": observed,
            "started_ms": parse_timestamp(clock()),
        }
    )
    execution_journal().transact(
        ledger,
        lambda _: {"action": "reserve", "observed_ms": observed, "data": encode_intent(intent)},
    )
    capture_digest = persist_evidence(
        root / "submission-capture.json", encode_submission_capture(capture)
    )
    acknowledgement = transport.submit(
        intent,
        valid_from_ms=capture["started_ms"],
        valid_until_ms=min(
            preflight_completed_ms + 5001,
            parse_timestamp(require_str(market, "close_time")),
        ),
    )
    returned = parse_timestamp(clock())
    acknowledgement_digest = persist_evidence(root / "acknowledgement.json", acknowledgement)
    timing = decode_submission_timing(
        {
            "client_order_id": intent["client_order_id"],
            "started_ms": capture["started_ms"],
            "returned_ms": returned,
            "capture_sha256": capture_digest,
            "acknowledgement_sha256": acknowledgement_digest,
        }
    )
    persist_evidence(root / "submission-timing.json", encode_submission_timing(timing))
    validate_acknowledgement(acknowledgement, intent, returned)


def read_submission_evidence(root: Path, intent: OrderIntent) -> JSONObject:
    """Verify saved submission contents against the caller's bound intent.

    Args:
        root: Directory holding native capture, acknowledgement and timing.
        intent: Independently loaded and validated order intent.

    Returns:
        Verified evidence and local workflow duration in milliseconds.

    Raises:
        ValueError: DEMO-EVIDENCE-004 for identity or digest disagreement.
        OSError: Required evidence is unavailable.
    """
    captured_bytes = (root / "submission-capture.json").read_bytes()
    capture = decode_submission_capture(
        narrow_json_to_dict(load_json_str(captured_bytes.decode("utf-8")))
    )
    timing = decode_submission_timing(read_document(root / "submission-timing.json"))
    payload = (root / "acknowledgement.json").read_bytes()
    if (
        capture["intent"] != intent
        or timing["client_order_id"] != intent["client_order_id"]
        or timing["started_ms"] != capture["started_ms"]
        or timing["capture_sha256"] != sha256(captured_bytes).hexdigest()
        or timing["acknowledgement_sha256"] != sha256(payload).hexdigest()
    ):
        raise ValueError("[DEMO-EVIDENCE-004] capture, timing and acknowledgement disagree")
    acknowledgement = narrow_json_to_dict(load_json_str(payload.decode("utf-8")))
    validate_acknowledgement(acknowledgement, intent, timing["returned_ms"])
    return {
        "mode": "demo_submission_evidence",
        "live_ready": False,
        "capture": encode_submission_capture(capture),
        "timing": encode_submission_timing(timing),
        "acknowledgement": acknowledgement,
        "local_workflow_duration_ms": timing["returned_ms"] - timing["started_ms"],
    }
