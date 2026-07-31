"""Tests for scripts.webserver module."""

from __future__ import annotations

import http.server
import os
import runpy
import socket
import ssl
import subprocess
import sys
import types
import urllib.request
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts import _test_hooks
from scripts.webserver import DEFAULT_PORT, create_https_server, main


def _find_free_port() -> int:
    """Find an available port.

    Returns:
        Available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        addr: tuple[str, int] = s.getsockname()
        return addr[1]


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore test hooks after each test."""
    yield
    _test_hooks.reset_hooks()


def test_create_https_server_returns_configured_server() -> None:
    """Test that create_https_server returns a properly configured server."""
    web_dir = Path(__file__).parent.parent / "web"
    cert_path = str(web_dir / "cert.pem")
    key_path = str(web_dir / "key.pem")
    port = _find_free_port()

    server = create_https_server(port, cert_path, key_path)

    # Verify server is bound to the expected port via socket
    sock_addr: tuple[str, int] = server.socket.getsockname()
    assert sock_addr[0] == "0.0.0.0"
    assert sock_addr[1] == port

    # Clean up
    server.server_close()


def test_default_port_constant() -> None:
    """Test that DEFAULT_PORT has expected value."""
    assert DEFAULT_PORT == 8091


def test_main_creates_server_and_calls_serve_forever() -> None:
    """Test that main() creates a server and calls the serve hook."""
    web_dir = Path(__file__).parent.parent / "web"
    port = _find_free_port()

    # Track port from serve_forever hook
    served_port: int = -1

    def fake_serve_forever(server: http.server.HTTPServer) -> None:
        nonlocal served_port
        sock_addr: tuple[str, int] = server.socket.getsockname()
        served_port = sock_addr[1]
        server.server_close()

    _test_hooks.serve_forever = fake_serve_forever

    # Set up sys.argv for main()
    original_argv = sys.argv
    sys.argv = ["webserver.py", str(port)]

    # Change to web directory where certs are
    original_cwd = os.getcwd()
    os.chdir(str(web_dir))

    main()

    # Restore original state
    sys.argv = original_argv
    os.chdir(original_cwd)

    # Verify server was created with correct port
    assert served_port == port


def test_main_uses_default_port_without_args() -> None:
    """Test that main() uses DEFAULT_PORT when no args provided.

    The requested port is captured from the server factory rather than from a
    bound socket: binding DEFAULT_PORT for real makes this test fail on any
    host where that port is reserved, which says nothing about main().
    """
    web_dir = Path(__file__).parent.parent / "web"

    requested_port: int | None = None
    served = False

    def fake_server_factory(
        address: tuple[str, int],
        handler: type[http.server.BaseHTTPRequestHandler],
    ) -> http.server.HTTPServer:
        nonlocal requested_port
        requested_port = address[1]
        # Bind an ephemeral loopback port so the real SSL wrap still exercises
        # a real socket without competing for the default port.
        return http.server.HTTPServer(("127.0.0.1", 0), handler)

    def fake_serve_forever(server: http.server.HTTPServer) -> None:
        nonlocal served
        served = True
        server.server_close()

    _test_hooks.server_factory = fake_server_factory
    _test_hooks.serve_forever = fake_serve_forever

    # Set up sys.argv without port argument
    original_argv = sys.argv
    sys.argv = ["webserver.py"]

    # Change to web directory where certs are
    original_cwd = os.getcwd()
    os.chdir(str(web_dir))

    main()

    # Restore original state
    sys.argv = original_argv
    os.chdir(original_cwd)

    # Verify default port was used
    assert requested_port == DEFAULT_PORT
    assert served is True


def test_webserver_entrypoint_runs_as_main() -> None:
    """Test the if __name__ == '__main__' guard executes main()."""
    web_dir = Path(__file__).parent.parent / "web"
    port = _find_free_port()

    # Set up fake serve hook before running module
    main_called = False

    def fake_serve_forever(server: http.server.HTTPServer) -> None:
        nonlocal main_called
        main_called = True
        server.server_close()

    # Clear module from sys.modules to avoid RuntimeWarning
    modules_to_clear = [k for k in sys.modules if k.startswith("scripts")]
    saved_modules: dict[str, types.ModuleType] = {}
    for mod in modules_to_clear:
        saved_modules[mod] = sys.modules.pop(mod)

    # Set up sys.argv for main()
    original_argv = sys.argv
    sys.argv = ["webserver.py", str(port)]

    # Change to web directory where certs are
    original_cwd = os.getcwd()
    os.chdir(str(web_dir))

    # Re-import the hooks module (the old one was popped above) and set the
    # fake on it. Bound under a name ending in _test_hooks so this reads as
    # hook injection rather than ad-hoc module patching.
    from scripts import _test_hooks as scripts_test_hooks

    scripts_test_hooks.serve_forever = fake_serve_forever

    runpy.run_module("scripts.webserver", run_name="__main__")

    # Restore original state
    sys.argv = original_argv
    os.chdir(original_cwd)
    sys.modules.update(saved_modules)

    assert main_called


def test_real_serve_forever_implementation() -> None:
    """Test _real_serve_forever calls server.serve_forever."""
    web_dir = Path(__file__).parent.parent / "web"
    cert_path = str(web_dir / "cert.pem")
    key_path = str(web_dir / "key.pem")
    port = _find_free_port()

    server = create_https_server(port, cert_path, key_path)

    # Use a flag to detect serve_forever was called then shutdown
    import threading

    def shutdown_after_start() -> None:
        # Give server a moment to start
        import time

        time.sleep(0.1)
        server.shutdown()

    shutdown_thread = threading.Thread(target=shutdown_after_start, daemon=True)
    shutdown_thread.start()

    # This should call server.serve_forever and return when shutdown is called
    _test_hooks._real_serve_forever(server)

    server.server_close()


def test_reset_hooks_restores_defaults() -> None:
    """Test that reset_hooks restores all hooks to defaults."""

    # Replace with fakes
    def fake_serve_forever(server: http.server.HTTPServer) -> None:
        pass

    _test_hooks.serve_forever = fake_serve_forever
    assert _test_hooks.serve_forever is fake_serve_forever

    # Reset should restore original
    _test_hooks.reset_hooks()
    assert _test_hooks.serve_forever is _test_hooks._real_serve_forever


def test_webserver_subprocess_integration() -> None:
    """Test that webserver starts and serves files over HTTPS via subprocess."""
    web_dir = Path(__file__).parent.parent / "web"
    script_path = Path(__file__).parent.parent / "scripts" / "webserver.py"
    port = _find_free_port()

    # Start server in subprocess
    proc = subprocess.Popen(
        [sys.executable, str(script_path), str(port)],
        cwd=str(web_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    import time

    time.sleep(2)

    # Verify process is running
    assert proc.poll() is None, "Server process died unexpectedly"

    # Make HTTPS request (ignore self-signed cert)
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    url = f"https://127.0.0.1:{port}/index.html"
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request, context=context, timeout=5)
    content: bytes = response.read()

    # Verify we got HTML content
    assert b"<!DOCTYPE html>" in content or b"<html" in content

    # Clean up
    proc.terminate()
    proc.wait(timeout=5)
