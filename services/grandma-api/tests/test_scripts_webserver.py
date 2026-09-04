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
import trustme
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


@pytest.fixture(scope="session")
def served_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding TLS material and one static file to serve.

    Six tests in this file used to point at ``web/`` for ``cert.pem`` and
    ``key.pem``. Those two are gitignored -- ``web/*.pem``, and correctly
    so, because a committed private key is a private key nobody can rotate
    -- so the files exist only where somebody once ran the generator. On CI
    the directory has no ``.pem`` in it at all and every one of the six died
    on ``FileNotFoundError`` inside ``load_cert_chain``.

    Minting the pair per session fixes that and says something truer
    besides: what these tests exercise is ``create_https_server`` wrapping a
    socket in a context built from A certificate, not from THE developer's
    certificate.

    ``index.html`` is here for the same reason. The subprocess test asked
    the real ``web/`` tree for a page, which coupled a webserver test to the
    frontend's build output.

    Args:
        tmp_path_factory: pytest's session-scoped temp directory factory.

    Returns:
        The directory, containing ``cert.pem``, ``key.pem`` and
        ``index.html``.
    """
    directory = tmp_path_factory.mktemp("served")
    authority = trustme.CA()
    certificate = authority.issue_cert("127.0.0.1", "localhost")
    certificate.private_key_pem.write_to_path(str(directory / "key.pem"))
    certificate.cert_chain_pems[0].write_to_path(str(directory / "cert.pem"))
    (directory / "index.html").write_text(
        "<!DOCTYPE html>\n<html><body>served</body></html>\n", encoding="utf-8"
    )
    return directory


def test_create_https_server_returns_configured_server(served_dir: Path) -> None:
    """Test that create_https_server returns a properly configured server."""
    cert_path = str(served_dir / "cert.pem")
    key_path = str(served_dir / "key.pem")
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


def test_main_creates_server_and_calls_serve_forever(served_dir: Path) -> None:
    """Test that main() creates a server and calls the serve hook."""
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

    # Serve from the fixture directory: main() defaults cert/key to
    # <cwd>/cert.pem and <cwd>/key.pem, so the cwd IS how it finds them.
    original_cwd = os.getcwd()
    os.chdir(str(served_dir))

    main()

    # Restore original state
    sys.argv = original_argv
    os.chdir(original_cwd)

    # Verify server was created with correct port
    assert served_port == port


def test_main_uses_default_port_without_args(served_dir: Path) -> None:
    """Test that main() uses DEFAULT_PORT when no args provided.

    The requested port is captured from the server factory rather than from a
    bound socket: binding DEFAULT_PORT for real makes this test fail on any
    host where that port is reserved, which says nothing about main().
    """

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

    # Serve from the fixture directory: main() defaults cert/key to
    # <cwd>/cert.pem and <cwd>/key.pem, so the cwd IS how it finds them.
    original_cwd = os.getcwd()
    os.chdir(str(served_dir))

    main()

    # Restore original state
    sys.argv = original_argv
    os.chdir(original_cwd)

    # Verify default port was used
    assert requested_port == DEFAULT_PORT
    assert served is True


def test_webserver_entrypoint_runs_as_main(served_dir: Path) -> None:
    """Test the if __name__ == '__main__' guard executes main()."""
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

    # Serve from the fixture directory: main() defaults cert/key to
    # <cwd>/cert.pem and <cwd>/key.pem, so the cwd IS how it finds them.
    original_cwd = os.getcwd()
    os.chdir(str(served_dir))

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


def test_real_serve_forever_implementation(served_dir: Path) -> None:
    """Test _real_serve_forever calls server.serve_forever."""
    cert_path = str(served_dir / "cert.pem")
    key_path = str(served_dir / "key.pem")
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


def test_webserver_subprocess_integration(served_dir: Path) -> None:
    """Test that webserver starts and serves files over HTTPS via subprocess."""
    project_root = Path(__file__).parent.parent
    port = _find_free_port()

    # Launched exactly as scripts/start.ps1 launches it: `-m` from the project
    # root, with the directory to serve passed as argv[2] (main() chdirs to it).
    # Running the file BY PATH with cwd=served_dir instead put scripts/ on
    # sys.path[0] rather than the project root, so `from scripts import
    # _test_hooks` at the top of webserver.py could only resolve against an
    # INSTALLED top-level `scripts` package -- one that every package here
    # shipped, so the copy found was whichever installed last.
    proc = subprocess.Popen(
        [sys.executable, "-m", "scripts.webserver", str(port), str(served_dir)],
        cwd=str(project_root),
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
