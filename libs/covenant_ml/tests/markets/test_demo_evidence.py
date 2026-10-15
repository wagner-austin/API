"""Real socket and filesystem failures exercise durable submission evidence."""

from __future__ import annotations

from contextlib import closing
from hashlib import sha256
from pathlib import Path

import pytest
from httpx import RemoteProtocolError
from platform_core.http_client import build_client
from platform_core.json_utils import dump_json_str, require_dict

from covenant_ml.markets.demo_trial import build_demo_trial
from covenant_ml.markets.execution_journal import read_execution_journal
from covenant_ml.markets.submission_evidence import (
    decode_submission_capture,
    decode_submission_timing,
    encode_submission_capture,
    encode_submission_timing,
)
from covenant_ml.markets.validation import read_document

from ._demo_trial_fixtures import serve_submission
from ._kalshi_fixtures import account_clock, serve_account, signing_key
from ._server import ApiServer, api_server

__all__ = ["api_server"]


def test_success_preserves_rules_depth_and_ack_identity(
    api_server: ApiServer, tmp_path: Path
) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"

    def clock() -> str:
        if api_server.bodies:
            return "1970-01-01T00:02:00.200Z"
        if (root / "submission-capture.json").exists():
            return "1970-01-01T00:02:00.100Z"
        return account_clock()

    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        market = require_dict(api_server.documents["/trade-api/v2/markets/MARKET-1"][0], "market")
        market["rules_primary"] = "Captured contract rules."
        market["rules_secondary"] = "Captured exceptions."
        native_book = api_server.documents["/trade-api/v2/markets/MARKET-1/orderbook"][0]
        report = service.submit(root)
        capture = decode_submission_capture(read_document(root / "submission-capture.json"))
        timing = decode_submission_timing(read_document(root / "submission-timing.json"))
        assert encode_submission_capture(capture) == {
            "intent": read_document(root / "trial.json")["intent"],
            "market": market,
            "book": native_book,
            "preflight_completed_ms": 120000,
            "market_received_ms": 120000,
            "book_received_ms": 120000,
            "decision_ms": 120000,
            "started_ms": 120000,
        }
        assert encode_submission_timing(timing) == {
            "client_order_id": capture["intent"]["client_order_id"],
            "started_ms": 120000,
            "returned_ms": 120200,
            "capture_sha256": sha256((root / "submission-capture.json").read_bytes()).hexdigest(),
            "acknowledgement_sha256": sha256(
                (root / "acknowledgement.json").read_bytes()
            ).hexdigest(),
        }
        assert report["confirmed_orders"] == 1
        assert len(api_server.bodies) == 1
        assert service.evidence(root)["local_workflow_duration_ms"] == 200


def test_disconnect_preserves_capture_and_cannot_resend(
    api_server: ApiServer, tmp_path: Path
) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        api_server.disconnect_paths.add("/trade-api/v2/portfolio/events/orders")
        with pytest.raises(RemoteProtocolError):
            service.submit(root)
        capture_bytes = (root / "submission-capture.json").read_bytes()
        assert (
            decode_submission_capture(read_document(root / "submission-capture.json"))["intent"]
            == read_execution_journal(root / "execution.sqlite")["orders"][0]["intent"]
        )
        assert not (root / "acknowledgement.json").exists()
        assert not (root / "submission-timing.json").exists()
        restarted = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        with pytest.raises(ValueError, match="DEMO-TRIAL-005"):
            restarted.submit(root)
        assert (root / "submission-capture.json").read_bytes() == capture_bytes
        assert len(api_server.bodies) == 1
        assert restarted.status(root)["exposure"] == 10000


@pytest.mark.parametrize("changed", ["capture", "rules", "identity", "start", "acknowledgement"])
def test_evidence_replay_rejects_record_disagreement(
    api_server: ApiServer, tmp_path: Path, changed: str
) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        service.submit(root)
        if changed in ("capture", "rules"):
            path = root / "submission-capture.json"
            document = read_document(path)
            if changed == "capture":
                require_dict(document, "intent")["fee_reserve"] = 199
            else:
                require_dict(document, "market")["rules_primary"] = "Changed rules."
        elif changed == "acknowledgement":
            path = root / "acknowledgement.json"
            document = read_document(path)
            document["order_id"] = "changed-native-id"
        else:
            path = root / "submission-timing.json"
            document = read_document(path)
            if changed == "identity":
                document["client_order_id"] = "another-intent"
            else:
                document["started_ms"] = 119999
        path.write_text(dump_json_str(document), encoding="utf-8")
        requests_before = tuple(api_server.requests)
        with pytest.raises(ValueError, match="DEMO-EVIDENCE-004"):
            service.evidence(root)
        assert tuple(api_server.requests) == requests_before
        assert len(api_server.bodies) == 1


@pytest.mark.parametrize("filename", ["submission-capture.json", "submission-timing.json"])
def test_real_file_collision_keeps_reservation(
    api_server: ApiServer, tmp_path: Path, filename: str
) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        (root / filename).write_text("existing evidence", encoding="utf-8")
        with pytest.raises(FileExistsError):
            service.submit(root)
        assert (root / filename).read_text("utf-8") == "existing evidence"
        assert len(api_server.bodies) == (0 if filename == "submission-capture.json" else 1)
        assert service.status(root)["exposure"] == 10000
        with pytest.raises(ValueError, match="DEMO-TRIAL-005"):
            service.submit(root)


def test_persistence_delay_expires_quote_before_post(api_server: ApiServer, tmp_path: Path) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"

    def clock() -> str:
        return (
            "1970-01-01T00:02:05.001Z"
            if (root / "submission-capture.json").exists()
            else account_clock()
        )

    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        with pytest.raises(ValueError, match="DEMO-EVIDENCE-003"):
            service.submit(root)
        assert api_server.bodies == []
        assert service.status(root)["exposure"] == 10000
        assert (
            decode_submission_capture(read_document(root / "submission-capture.json"))["started_ms"]
            == 120000
        )


def test_invalid_native_ack_is_saved_before_rejection(
    api_server: ApiServer, tmp_path: Path
) -> None:
    _, signer = signing_key()
    root = tmp_path / "trial"
    with closing(build_client(5)) as client:
        service = build_demo_trial(client, signer, account_clock, origin=api_server.url)
        serve_account(api_server)
        service.prepare(root, "MARKET-1", 9800)
        serve_submission(api_server, root)
        ack = api_server.documents["/trade-api/v2/portfolio/events/orders"][0]
        ack["client_order_id"] = "foreign-id"
        with pytest.raises(ValueError, match="DEMO-TRIAL-002"):
            service.submit(root)
        assert read_document(root / "acknowledgement.json") == ack
        timing = decode_submission_timing(read_document(root / "submission-timing.json"))
        assert (
            timing["acknowledgement_sha256"]
            == sha256((root / "acknowledgement.json").read_bytes()).hexdigest()
        )
        assert service.status(root)["issues"] == ["EXECUTION-UNRESOLVED"]
        assert len(api_server.bodies) == 1
