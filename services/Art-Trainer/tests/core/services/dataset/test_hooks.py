"""Tests for dataset service hooks.

These tests exercise the default hook implementations and reset functionality.
"""

from __future__ import annotations

import http.server
import socketserver
import threading

import pytest

from art_trainer.core.services.dataset import _test_hooks
from art_trainer.core.services.dataset._test_hooks import (
    DataBankDownloadError,
    DataBankUploadError,
    UploadResult,
    _default_http_get,
    _default_http_upload,
    reset_hooks,
)


def test_reset_hooks_resets_to_defaults() -> None:
    """Test reset_hooks restores default implementations."""
    # Store original hooks
    original_get = _test_hooks.http_get
    original_upload = _test_hooks.http_upload

    # Set hooks to something else
    def fake_get(url: str, headers: dict[str, str]) -> bytes:
        return b"fake"

    def fake_upload(
        url: str,
        headers: dict[str, str],
        filename: str,
        content: bytes,
    ) -> UploadResult:
        return {"file_id": "fake", "filename": "fake.txt"}

    _test_hooks.http_get = fake_get
    _test_hooks.http_upload = fake_upload

    # Verify hooks were changed
    assert _test_hooks.http_get is not original_get
    assert _test_hooks.http_upload is not original_upload

    # Reset hooks
    reset_hooks()

    # Verify hooks are restored to defaults
    assert _test_hooks.http_get is _default_http_get
    assert _test_hooks.http_upload is _default_http_upload


class _TestHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """Simple HTTP request handler for testing."""

    response_body: bytes = b"test content"
    response_code: int = 200
    upload_response_json: bytes = b'{"file_id": "test-file-id"}'

    def log_message(self, format_str: str, *args: str) -> None:
        """Suppress logging.

        Args:
            format_str: Format string.
            args: Arguments for format string.
        """
        pass

    def do_GET(self) -> None:
        """Handle GET requests."""
        self.send_response(self.response_code)
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(self.response_body)

    def do_POST(self) -> None:
        """Handle POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        # Read body but don't use it (just to consume the request)
        _ = self.rfile.read(content_length)

        self.send_response(self.response_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.upload_response_json)


def _start_test_server(
    response_body: bytes = b"test content",
    response_code: int = 200,
    upload_response_json: bytes = b'{"file_id": "test-file-id"}',
) -> tuple[socketserver.TCPServer, threading.Thread, int]:
    """Start a test HTTP server.

    Args:
        response_body: Body to return for GET requests.
        response_code: HTTP status code to return.
        upload_response_json: JSON bytes for POST response.

    Returns:
        Tuple of (server, thread, port).
    """

    class Handler(_TestHTTPRequestHandler):
        pass

    Handler.response_body = response_body
    Handler.response_code = response_code
    Handler.upload_response_json = upload_response_json

    # Find an available port
    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    return server, thread, port


def test_default_http_get_success() -> None:
    """Test _default_http_get downloads content successfully."""
    expected_content = b"downloaded file content"
    server, _thread, port = _start_test_server(
        response_body=expected_content,
        response_code=200,
    )

    url = f"http://127.0.0.1:{port}/test-file"
    headers = {"Authorization": "Bearer test-token"}

    result = _default_http_get(url, headers)

    assert result == expected_content

    server.shutdown()


def test_default_http_get_failure() -> None:
    """Test _default_http_get raises error on non-200 response."""
    server, _thread, port = _start_test_server(
        response_body=b"error",
        response_code=404,
    )

    url = f"http://127.0.0.1:{port}/test-file"
    headers = {"Authorization": "Bearer test-token"}

    with pytest.raises(DataBankDownloadError) as exc_info:
        _default_http_get(url, headers)

    assert "404" in str(exc_info.value)

    server.shutdown()


def test_default_http_upload_success() -> None:
    """Test _default_http_upload uploads content successfully."""
    server, _thread, port = _start_test_server(
        response_code=200,
        upload_response_json=b'{"file_id": "uploaded-file-123"}',
    )

    url = f"http://127.0.0.1:{port}/upload"
    headers = {"Authorization": "Bearer test-token"}
    filename = "test.safetensors"
    content = b"lora file content"

    result = _default_http_upload(url, headers, filename, content)

    assert result["file_id"] == "uploaded-file-123"
    assert result["filename"] == filename

    server.shutdown()


def test_default_http_upload_success_201() -> None:
    """Test _default_http_upload accepts 201 response."""
    server, _thread, port = _start_test_server(
        response_code=201,
        upload_response_json=b'{"file_id": "created-file-456"}',
    )

    url = f"http://127.0.0.1:{port}/upload"
    headers = {"Authorization": "Bearer test-token"}
    filename = "new.safetensors"
    content = b"new lora content"

    result = _default_http_upload(url, headers, filename, content)

    assert result["file_id"] == "created-file-456"
    assert result["filename"] == filename

    server.shutdown()


def test_default_http_upload_failure() -> None:
    """Test _default_http_upload raises error on non-200/201 response."""
    server, _thread, port = _start_test_server(
        response_code=500,
        upload_response_json=b'{"error": "server error"}',
    )

    url = f"http://127.0.0.1:{port}/upload"
    headers = {"Authorization": "Bearer test-token"}
    filename = "test.safetensors"
    content = b"lora file content"

    with pytest.raises(DataBankUploadError) as exc_info:
        _default_http_upload(url, headers, filename, content)

    assert "500" in str(exc_info.value)

    server.shutdown()


def test_default_http_upload_invalid_response_not_dict() -> None:
    """Test _default_http_upload raises error when response is not a dict."""

    # Create a custom handler that returns an array
    class ArrayHandler(_TestHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(content_length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'["not", "a", "dict"]')

    server = socketserver.TCPServer(("127.0.0.1", 0), ArrayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    url = f"http://127.0.0.1:{port}/upload"
    headers = {"Authorization": "Bearer test-token"}
    filename = "test.safetensors"
    content = b"lora file content"

    with pytest.raises(DataBankUploadError) as exc_info:
        _default_http_upload(url, headers, filename, content)

    assert "not a dict" in str(exc_info.value)

    server.shutdown()


def test_default_http_upload_invalid_response_missing_file_id() -> None:
    """Test _default_http_upload raises error when file_id is missing."""
    server, _thread, port = _start_test_server(
        response_code=200,
        upload_response_json=b'{"other_field": "value"}',
    )

    url = f"http://127.0.0.1:{port}/upload"
    headers = {"Authorization": "Bearer test-token"}
    filename = "test.safetensors"
    content = b"lora file content"

    with pytest.raises(DataBankUploadError) as exc_info:
        _default_http_upload(url, headers, filename, content)

    assert "missing file_id" in str(exc_info.value)

    server.shutdown()


def test_default_http_upload_invalid_response_file_id_not_string() -> None:
    """Test _default_http_upload raises error when file_id is not a string."""

    # Create a custom handler that returns file_id as a number
    class NumericFileIdHandler(_TestHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(content_length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"file_id": 12345}')

    server = socketserver.TCPServer(("127.0.0.1", 0), NumericFileIdHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    url = f"http://127.0.0.1:{port}/upload"
    headers = {"Authorization": "Bearer test-token"}
    filename = "test.safetensors"
    content = b"lora file content"

    with pytest.raises(DataBankUploadError) as exc_info:
        _default_http_upload(url, headers, filename, content)

    assert "missing file_id" in str(exc_info.value)

    server.shutdown()
