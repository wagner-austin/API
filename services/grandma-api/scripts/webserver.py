"""Simple HTTPS server for grandma-api web frontend.

Serves static files over HTTPS using a self-signed certificate.
"""

from __future__ import annotations

import http.server
import os
import ssl
import sys

from platform_core.logging import get_logger

from scripts import _test_hooks

logger = get_logger(__name__)

DEFAULT_PORT = 8091


def create_https_server(
    port: int,
    cert_path: str,
    key_path: str,
) -> http.server.HTTPServer:
    """Create an HTTPS server configured with SSL.

    Args:
        port: Port number to listen on.
        cert_path: Path to SSL certificate file.
        key_path: Path to SSL private key file.

    Returns:
        Configured HTTPServer ready to serve.

    Raises:
        FileNotFoundError: If cert or key files are not found.
        ssl.SSLError: If SSL configuration fails.
    """
    server = _test_hooks.server_factory(
        ("0.0.0.0", port),
        http.server.SimpleHTTPRequestHandler,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    return server


def main() -> None:
    """Start the HTTPS server.

    Args (via sys.argv):
        port: Port number (default 8091).
        work_dir: Working directory to serve files from (default current dir).
        cert_path: Path to SSL certificate (default work_dir/cert.pem).
        key_path: Path to SSL key (default work_dir/key.pem).
    """
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    work_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    cert_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(work_dir, "cert.pem")
    key_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join(work_dir, "key.pem")

    # Change to working directory for serving files
    os.chdir(work_dir)

    server = create_https_server(port, cert_path, key_path)
    logger.info(
        "HTTPS server running",
        extra={"port": port, "work_dir": work_dir, "url": f"https://0.0.0.0:{port}"},
    )
    _test_hooks.serve_forever(server)


if __name__ == "__main__":
    main()
