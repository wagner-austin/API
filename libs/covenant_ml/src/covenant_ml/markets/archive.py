"""Explicit capture and offline replay of hashed public HTTP responses."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from threading import Lock
from urllib.parse import urlencode

from platform_core.http_client import HttpxClient
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from .validation import parse_timestamp, require_nonempty


class Capture(TypedDict):
    """Request identity, acquisition time and exact decoded response text."""

    url: ReadOnly[str]
    captured_at: ReadOnly[str]
    body: ReadOnly[str]
    digest: ReadOnly[str]


def encode_capture(record: Capture) -> JSONObject:
    """Encode an archived response.

    Args:
        record: Validated capture.

    Returns:
        JSON envelope.
    """
    return {
        "url": record["url"],
        "captured_at": record["captured_at"],
        "body": record["body"],
        "digest": record["digest"],
    }


def decode_capture(document: JSONObject) -> Capture:
    """Verify required fields, timestamp and content hash before parsing payload.

    Args:
        document: JSON envelope.

    Returns:
        Verified immutable capture.

    Raises:
        ValueError: MARKET-015 for a changed payload or invalid timestamp.
        TypeError: Missing or incorrectly typed fields.
    """
    body = require_str(document, "body")
    digest = require_nonempty(document, "digest")
    timestamp = require_nonempty(document, "captured_at")
    parse_timestamp(timestamp)
    if sha256(body.encode("utf-8")).hexdigest() != digest:
        raise ValueError("[MARKET-015] archived response hash mismatch")
    return Capture(
        url=require_nonempty(document, "url"), captured_at=timestamp, body=body, digest=digest
    )


class ResponseArchive:
    """Read-only HTTP acquisition with explicit, separate offline replay.

    A capture never overwrites an existing request. Offline replay never accesses
    the network. HTTP errors propagate and cannot become empty evidence.
    """

    def __init__(
        self,
        root: Path,
        base_url: str,
        client: HttpxClient,
        clock: Callable[[], str],
        pause: Callable[[float], None],
    ) -> None:
        """Bind archive storage and the actual I/O dependencies.

        Args:
            root: Directory of immutable request envelopes.
            base_url: Public API origin and prefix.
            client: Owned HTTP client supplied by the factory.
            clock: Aware UTC timestamp reader.
            pause: Request pacing implementation.
        """
        self.root = root
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.clock = clock
        self.pause = pause
        self._request_lock = Lock()

    def request_url(self, path: str, params: dict[str, str | int]) -> str:
        """Construct a canonical request identity from sorted query parameters.

        Args:
            path: Absolute API path suffix.
            params: Query arguments.

        Returns:
            Complete URL.
        """
        return self.base_url + path + "?" + urlencode(sorted(params.items()))

    def file_for(self, url: str) -> Path:
        """Resolve the content-independent request filename.

        Args:
            url: Canonical request URL.

        Returns:
            Local envelope path.
        """
        return self.root / (sha256(url.encode("utf-8")).hexdigest() + ".json")

    def capture(self, path: str, params: dict[str, str | int]) -> JSONObject:
        """Acquire, persist and validate one response without retries.

        Args:
            path: API path suffix.
            params: Query arguments.

        Returns:
            Parsed response object.

        Raises:
            FileExistsError: The request is already captured.
            HTTPStatusError: The server returned an unsuccessful status.
            ValueError: Response JSON is malformed.
            OSError: Persistence failed.
        """
        url = self.request_url(path, params)
        target = self.file_for(url)
        if target.exists():
            raise FileExistsError("[MARKET-016] request already captured; use replay explicitly")
        # All workers share one start-rate limit, not one limit per worker.
        with self._request_lock:
            self.pause(0.3)
        response = self.client.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        body = response.text
        capture = decode_capture(
            {
                "url": url,
                "captured_at": self.clock(),
                "body": body,
                "digest": sha256(body.encode("utf-8")).hexdigest(),
            }
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as output:
            output.write(dump_json_str(encode_capture(capture)))
        return narrow_json_to_dict(load_json_str(body))

    def replay(self, path: str, params: dict[str, str | int]) -> JSONObject:
        """Read verified evidence without a network request.

        Args:
            path: API path suffix.
            params: Query arguments.

        Returns:
            Parsed response object.

        Raises:
            ValueError: MARKET-017 for a changed request identity, or invalid hash.
            OSError: Required evidence cannot be read.
        """
        capture = self.load_capture(path, params)
        return narrow_json_to_dict(load_json_str(capture["body"]))

    def load_capture(self, path: str, params: dict[str, str | int]) -> Capture:
        """Read verified response bytes and their actual acquisition timestamp.

        Args:
            path: API path suffix.
            params: Canonical query arguments.

        Returns:
            Immutable envelope including receipt time and payload digest.

        Raises:
            ValueError: MARKET-017 for changed identity, MARKET-015 for changed bytes.
            OSError: Evidence cannot be read.
        """
        url = self.request_url(path, params)
        document = narrow_json_to_dict(load_json_str(self.file_for(url).read_text("utf-8")))
        capture = decode_capture(document)
        if capture["url"] != url:
            raise ValueError("[MARKET-017] archived request identity mismatch")
        return capture

    def resume_capture(self, path: str, params: dict[str, str | int]) -> JSONObject:
        """Continue an explicitly requested capture using verified checkpoints.

        Args:
            path: API path suffix.
            params: Query arguments.

        Returns:
            Existing verified response, or a newly captured response.

        Raises:
            ValueError: Existing evidence is corrupt; it is never refetched.
            HTTPStatusError: A new request failed; it is never softened.
        """
        if self.file_for(self.request_url(path, params)).exists():
            return self.replay(path, params)
        return self.capture(path, params)


__all__ = ["Capture", "ResponseArchive", "decode_capture", "encode_capture"]
