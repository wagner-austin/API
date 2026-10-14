"""Public price collection and sealed offline reproduction for stock paper studies."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib.parse import quote

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from .archive import ResponseArchive, decode_capture, encode_capture
from .equity_history import decode_equity_chart
from .equity_outcome_records import encode_equity_window_result
from .equity_outcomes import evaluate_equity_window
from .equity_records import EquityPrice
from .equity_study import EquityStudy, encode_equity_study
from .validation import parse_timestamp, read_document, require_nonempty


def _acquire_histories(
    study: EquityStudy,
    archive: ResponseArchive,
    capture: bool,
    started_ms: int,
    ready: bool,
) -> tuple[dict[str, tuple[EquityPrice, ...]], tuple[Path, ...], int]:
    """Acquire every mature cohort member and enforce ordered receipt times.

    Args:
        study: Validated stock design and exact calendar.
        archive: Actual capture and offline replay dependencies.
        capture: Explicit acquisition mode.
        started_ms: Observation start in epoch milliseconds.
        ready: Whether the conservative acquisition date has elapsed.

    Returns:
        All native histories, exact evidence paths and last receipt timestamp.

    Raises:
        ValueError: EQUITY-019 for clock regression; native errors propagate.
        HTTPStatusError: A required public request failed.
        OSError: Required evidence cannot be read or persisted.
    """
    histories: dict[str, tuple[EquityPrice, ...]] = {}
    evidence: list[Path] = []
    previous = started_ms
    if ready:
        params: dict[str, str | int] = {
            "period1": parse_timestamp(study["expected_days"][0] + "T00:00:00Z") // 1000,
            "period2": parse_timestamp(study["expected_days"][-1] + "T00:00:00Z") // 1000 + 86400,
            "interval": "1d",
            "events": "div,splits",
        }
        for symbol in study["cohort"]:
            path = "/v8/finance/chart/" + quote(symbol, safe="")
            document = archive.capture(path, params) if capture else archive.replay(path, params)
            receipt = archive.load_capture(path, params)
            received = parse_timestamp(receipt["captured_at"])
            if received < previous:
                raise ValueError("[EQUITY-019] price receipt clock regressed")
            previous = received
            histories[symbol] = decode_equity_chart(
                document, symbol, "USD", received_at=receipt["captured_at"]
            )
            evidence.append(archive.file_for(archive.request_url(path, params)))
    return histories, tuple(evidence), previous


def collect_equity_outcomes(
    study: EquityStudy,
    archive: ResponseArchive,
    capture: bool,
) -> tuple[JSONObject, tuple[Path, ...]]:
    """Capture the whole cohort or replay a sealed observation without network I/O.

    Args:
        study: Validated immutable snapshot-bound design.
        archive: Actual HTTP, clock, pacing and immutable storage dependencies.
        capture: True for a new observation; False for mandatory offline replay.

    Returns:
        Explicit pending or complete report and every acquisition evidence path.
        Acquisition starts only at UTC midnight two days after the exit date,
        conservatively exceeding the native decoder's completed-bar requirement.

    Raises:
        FileExistsError: EQUITY-018 when capture would reuse an existing directory.
        ValueError: EQUITY-019 for clock regression or changed replay identity;
            native decoding and incomplete outcome failures propagate.
        HTTPStatusError: A required public response failed, without retry.
        OSError: Required evidence cannot be persisted or read.
    """
    seal_path = archive.root / "observation.json"
    identity = "equity://" + sha256(dump_json_str(encode_equity_study(study)).encode()).hexdigest()
    if capture:
        if archive.root.exists():
            raise FileExistsError("[EQUITY-018] capture requires a new observation directory")
        started = archive.clock()
        saved: JSONObject | None = None
        finished = started
    else:
        seal = decode_capture(read_document(seal_path))
        if seal["url"] != identity:
            raise ValueError("[EQUITY-019] observation belongs to a different study")
        saved = narrow_json_to_dict(load_json_str(seal["body"]))
        started = require_nonempty(saved, "started_at")
        finished = seal["captured_at"]
    first_ms = parse_timestamp(started)
    exit_ms = parse_timestamp(study["expected_days"][-1] + "T00:00:00Z")
    ready = first_ms >= exit_ms + 172800000
    histories, evidence, previous = _acquire_histories(study, archive, capture, first_ms, ready)
    if capture:
        finished = archive.clock()
    if parse_timestamp(finished) < previous:
        raise ValueError("[EQUITY-019] observation finished before its receipts")
    scores: list[JSONValue] = []
    returns: list[JSONValue] = []
    if ready:
        for cost in study["costs"]:
            result = evaluate_equity_window(
                histories,
                cohort=study["cohort"],
                selected=study["selected"],
                expected_days=study["expected_days"],
                decision_at=study["decision_at"],
                as_of=finished,
                round_trip_cost=cost,
            )
            # Eligibility and the non-regressing clock prove the exit has elapsed.
            assert result is not None
            returns = [{"symbol": symbol, "adjusted_return": value} for symbol, value in result[0]]
            scores.append(encode_equity_window_result(result[1]))
    report: JSONObject = {
        "study": encode_equity_study(study),
        "started_at": started,
        "evaluated_at": finished,
        "complete": ready,
        "returns": returns,
        "scores": scores,
        "orders_submitted": 0,
        "advance_to_funded_trading": False,
    }
    if capture:
        body = dump_json_str(report)
        sealed = decode_capture(
            {
                "url": identity,
                "captured_at": finished,
                "body": body,
                "digest": sha256(body.encode()).hexdigest(),
            }
        )
        archive.root.mkdir(parents=True, exist_ok=True)
        with seal_path.open("x", encoding="utf-8", newline="\n") as destination:
            destination.write(dump_json_str(encode_capture(sealed)))
    elif saved != report:
        raise ValueError("[EQUITY-019] replay differs from the sealed observation")
    return report, (*evidence, seal_path)
