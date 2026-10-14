"""Explicit official release acquisition and deterministic offline validation."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONObject

from .archive import ResponseArchive
from .bls_records import encode_cpi_release
from .bls_release import decode_bls_release
from .validation import parse_timestamp, require_day

RELEASE_PATH = "/news.release/cpi.nr0.htm"
RESEARCH_CLIENT = "covenant-market-research/1.0 (public CPI release observation)"


def collect_cpi_release(
    archive: ResponseArchive,
    *,
    target_month: str,
    published_at: str,
    capture: bool,
) -> tuple[JSONObject, tuple[Path, ...]]:
    """Acquire one expected official publication or replay its original receipt.

    Args:
        archive: Injected HTTP, clock, pacing and immutable storage dependencies.
        target_month: Required reference month in YYYY-MM format.
        published_at: Expected scheduled publication, including timezone.
        capture: True for a new capture; False for network-free replay.

    Returns:
        Validated release report and exact envelope source path. The receipt
        proves local observation, not first publication or exchange settlement.

    Raises:
        ValueError: CPI-008 for unexpected publication; CPI-010 for premature
            acquisition or a regressing receipt; native decoder errors propagate.
        FileExistsError: CPI-009 when acquisition reuses an existing directory.
        HTTPStatusError: Public GET failed, without retries or a replacement source.
        OSError: Required evidence cannot be persisted or read.
    """
    require_day({"day": target_month + "-01"}, "day")
    publication_ms = parse_timestamp(published_at)
    if capture:
        if archive.root.exists():
            raise FileExistsError("[CPI-009] capture requires a new observation directory")
        started_ms = parse_timestamp(archive.clock())
        if started_ms < publication_ms:
            raise ValueError("[CPI-010] acquisition precedes the expected publication")
        receipt = archive.capture_response(
            RELEASE_PATH, {}, headers={"Accept": "text/html", "User-Agent": RESEARCH_CLIENT}
        )
        if parse_timestamp(receipt["captured_at"]) < started_ms:
            raise ValueError("[CPI-010] release receipt clock regressed")
    else:
        receipt = archive.load_capture(RELEASE_PATH, {})
    release = decode_bls_release(
        receipt["body"], target_month=target_month, received_at=receipt["captured_at"]
    )
    if parse_timestamp(release["published_at"]) != publication_ms:
        raise ValueError("[CPI-008] observed publication differs from the expected release")
    report: JSONObject = {
        "release": encode_cpi_release(release),
        "received_at": receipt["captured_at"],
        "source_url": receipt["url"],
        "decoded_body_sha256": receipt["digest"],
        "orders_submitted": 0,
        "advance_to_funded_trading": False,
    }
    return report, (archive.file_for(receipt["url"]),)
