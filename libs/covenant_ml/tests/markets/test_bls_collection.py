"""Real socket acquisition preserves native HTML and fails closed on wrong releases."""

from __future__ import annotations

from contextlib import closing
from hashlib import sha256
from pathlib import Path

import pytest
from httpx import HTTPStatusError, RemoteProtocolError
from platform_core.json_utils import dump_json_str

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.bls_collection import RELEASE_PATH, collect_cpi_release
from covenant_ml.markets.validation import read_document

from ._server import ApiServer, api_server
from .test_bls_records import release_document
from .test_bls_release import native_html

__all__ = ["api_server"]
PUBLICATION = "2026-09-11T12:30:00Z"


def test_capture_replay_exact_content_and_no_replacement(
    api_server: ApiServer, tmp_path: Path
) -> None:
    html = native_html()
    api_server.text_documents[RELEASE_PATH] = [html]
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path / "capture", api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        report, paths = collect_cpi_release(
            archive, target_month="2026-08", published_at=PUBLICATION, capture=True
        )
        receipt = archive.load_capture(RELEASE_PATH, {})
        assert receipt["body"] == html
        assert report == {
            "release": release_document(),
            "received_at": receipt["captured_at"],
            "source_url": api_server.url + RELEASE_PATH + "?",
            "decoded_body_sha256": sha256(html.encode()).hexdigest(),
            "orders_submitted": 0,
            "advance_to_funded_trading": False,
        }
        assert paths == (archive.file_for(receipt["url"]),)
        assert collect_cpi_release(
            archive, target_month="2026-08", published_at=PUBLICATION, capture=False
        ) == (report, paths)
        with pytest.raises(FileExistsError, match="CPI-009"):
            collect_cpi_release(
                archive, target_month="2026-08", published_at=PUBLICATION, capture=True
            )
        assert api_server.requests == [RELEASE_PATH + "?"]
        assert api_server.request_headers[0]["Accept"] == "text/html"
        assert api_server.request_headers[0]["User-Agent"] == (
            "covenant-market-research/1.0 (public CPI release observation)"
        )
        saved = read_document(paths[0])
        saved["body"] = "changed"
        paths[0].write_text(dump_json_str(saved), encoding="utf-8")
        with pytest.raises(ValueError, match="MARKET-015"):
            collect_cpi_release(
                archive, target_month="2026-08", published_at=PUBLICATION, capture=False
            )
        assert api_server.requests == [RELEASE_PATH + "?"]


@pytest.mark.parametrize(
    "target,published,code",
    [("2026-09", PUBLICATION, "CPI-005"), ("2026-08", "2026-09-10T12:30:00Z", "CPI-008")],
)
def test_wrong_release_preserves_original_receipt(
    api_server: ApiServer, tmp_path: Path, target: str, published: str, code: str
) -> None:
    api_server.text_documents[RELEASE_PATH] = [native_html()]
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path / "capture", api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        for capture in (True, False):
            with pytest.raises(ValueError, match=code):
                collect_cpi_release(
                    archive, target_month=target, published_at=published, capture=capture
                )
        assert archive.load_capture(RELEASE_PATH, {})["body"] == native_html()
        assert api_server.requests == [RELEASE_PATH + "?"]


def test_premature_acquisition_has_no_effect(api_server: ApiServer, tmp_path: Path) -> None:
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        root = tmp_path / "capture"
        archive = ResponseArchive(
            root, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        with pytest.raises(ValueError, match="CPI-010"):
            collect_cpi_release(
                archive, target_month="2099-08", published_at="2099-09-11T12:30:00Z", capture=True
            )
        assert root.exists() is False
        assert api_server.requests == []


def test_regressing_receipt_retains_evidence(api_server: ApiServer, tmp_path: Path) -> None:
    api_server.text_documents[RELEASE_PATH] = [native_html()]
    instants = iter(("2026-09-11T12:30:01Z", PUBLICATION))
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path / "capture", api_server.url, client, lambda: next(instants), _test_hooks.pause
        )
        with pytest.raises(ValueError, match="CPI-010"):
            collect_cpi_release(
                archive, target_month="2026-08", published_at=PUBLICATION, capture=True
            )
        assert archive.load_capture(RELEASE_PATH, {})["captured_at"] == PUBLICATION
        assert api_server.requests == [RELEASE_PATH + "?"]


@pytest.mark.parametrize("disconnect", [False, True])
def test_real_transport_failure_creates_no_evidence(
    api_server: ApiServer, tmp_path: Path, disconnect: bool
) -> None:
    if disconnect:
        api_server.disconnect_paths.add(RELEASE_PATH)
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        root = tmp_path / "capture"
        archive = ResponseArchive(
            root, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        with pytest.raises(RemoteProtocolError if disconnect else HTTPStatusError):
            collect_cpi_release(
                archive, target_month="2026-08", published_at=PUBLICATION, capture=True
            )
        assert root.exists() is False
        assert api_server.requests == [RELEASE_PATH + "?"]
