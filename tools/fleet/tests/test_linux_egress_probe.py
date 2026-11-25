"""The capacity probe's egress lines, run under ``sh`` (MCPs board task e70cdf05).

The lines are RUN, with curl, against servers each case starts on this host's
loopback: one that answers, a port nothing listens on, a socket that accepts
and never answers, and a name under ``.invalid``, which RFC 6761 reserves so
that no resolver answers it. Each is a way the world ends an ask: an answer,
a refused connection, a timeout and a name that does not resolve, which was
lavender's own failure at 2026-11-25T04:00Z. The reader is then held to what
the lines print, and refuses what they could never mean.
"""

from __future__ import annotations

import pathlib
import socket
import subprocess
import threading
from collections.abc import Generator

import pytest

from fleet.core.dialect import PROLOGUE, SH_INVOCATION
from fleet.core.linux_egress_probe import (
    BOUNDS_LINE,
    COUNT_FIELD,
    EGRESS_PROBE_PART,
    TARGETS,
    TARGETS_LINE,
    read_egress,
)
from tests.test_dialect_linux import fields_of

#: Bounds short enough that a target that never answers costs a case a second.
QUICK_BOUNDS = "egress_connect=1 egress_max=1\n"

#: An HTTP answer refusing what was asked, which is still an answer: the name
#: resolved, a route existed and the server spoke HTTP.
NOT_FOUND = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"


def _answer_one(listener: socket.socket) -> None:
    """Take one connection, read its request and answer it 404.

    Args:
        listener: The bound, listening socket.
    """
    connection, _peer = listener.accept()
    with connection:
        connection.recv(65536)
        connection.sendall(NOT_FOUND)


@pytest.fixture(name="answering")
def _answering() -> Generator[str, None, None]:
    """A server on this host's loopback that answers the one HEAD a case sends.

    Yields:
        Its URL.
    """
    listener = socket.create_server(("127.0.0.1", 0))
    port: int = listener.getsockname()[1]
    worker = threading.Thread(target=_answer_one, args=(listener,), daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{port}/"
    worker.join(timeout=5)
    listener.close()
    assert worker.is_alive() is False


@pytest.fixture(name="silent")
def _silent() -> Generator[str, None, None]:
    """A socket the kernel completes connections to and nothing ever answers on.

    Yields:
        Its URL.
    """
    listener = socket.create_server(("127.0.0.1", 0))
    port: int = listener.getsockname()[1]
    yield f"http://127.0.0.1:{port}/"
    listener.close()


def _closed_port() -> str:
    """A loopback port nothing listens on: one the kernel just handed out and took back.

    Returns:
        Its URL.
    """
    taken = socket.create_server(("127.0.0.1", 0))
    port: int = taken.getsockname()[1]
    taken.close()
    return f"http://127.0.0.1:{port}/"


def _run(
    tmp_path: pathlib.Path,
    targets: tuple[str, ...],
    bounds: str = QUICK_BOUNDS,
    path: str | None = None,
) -> dict[str, str]:
    """Run the egress lines after the dialect's prologue, and read what they print.

    Args:
        tmp_path: Where the script goes.
        targets: What the targets line names instead of the package indexes.
        bounds: What the bounds line holds instead of the fleet's.
        path: The ``PATH`` the shell starts with, or None for this process's.

    Returns:
        The lines' key=value fields.
    """
    script = tmp_path / "egress.sh"
    script.write_text(
        PROLOGUE
        + EGRESS_PROBE_PART.replace(
            TARGETS_LINE, f"egress_targets='{' '.join(targets)}'\n", 1
        ).replace(BOUNDS_LINE, bounds, 1),
        encoding="utf-8",
    )
    environment = None if path is None else {"PATH": path, "HOME": str(tmp_path)}
    completed = subprocess.run(
        [*SH_INVOCATION, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    return fields_of(completed.stdout)


def test_the_lines_ask_both_package_indexes_within_ten_seconds() -> None:
    """poetry's index and npm's (pypi.org, registry.npmjs.org), each bounded."""
    assert TARGETS == ("https://pypi.org/simple/pip/", "https://registry.npmjs.org/")
    assert (
        TARGETS_LINE
        == "egress_targets='https://pypi.org/simple/pip/ https://registry.npmjs.org/'\n"
    )
    assert BOUNDS_LINE == "egress_connect=3 egress_max=5\n"
    assert EGRESS_PROBE_PART.startswith(TARGETS_LINE + BOUNDS_LINE)
    assert EGRESS_PROBE_PART.endswith(f"printf '{COUNT_FIELD}=%s\\n' \"$asked\"\n")


@pytest.mark.host_linux
class TestTheLinesUnderSh:
    def test_any_http_answer_is_reach_and_each_target_is_printed_in_order(
        self, tmp_path: pathlib.Path, answering: str
    ) -> None:
        closed = _closed_port()

        fields = _run(tmp_path, (answering, closed))

        assert fields == {
            "egress_0": f"0 {answering}",
            "egress_1": f"7 {closed}",
            "egress_count": "2",
        }

    def test_a_server_that_never_answers_times_out_within_the_bound(
        self, tmp_path: pathlib.Path, silent: str
    ) -> None:
        assert _run(tmp_path, (silent,)) == {"egress_0": f"28 {silent}", "egress_count": "1"}

    def test_a_name_no_resolver_answers_does_not_resolve(self, tmp_path: pathlib.Path) -> None:
        """lavender's failure: 'Could not resolve host' (curl exit 6)."""
        unknown = "https://egress-probe.invalid/"

        fields = _run(tmp_path, (unknown,), bounds="egress_connect=3 egress_max=5\n")

        assert fields == {"egress_0": f"6 {unknown}", "egress_count": "1"}

    def test_a_node_with_no_curl_prints_the_shell_s_not_found(self, tmp_path: pathlib.Path) -> None:
        empty = tmp_path / "no-tools"
        empty.mkdir()
        closed = _closed_port()

        fields = _run(tmp_path, (closed,), path=str(empty))

        assert fields == {"egress_0": f"127 {closed}", "egress_count": "1"}
        assert read_egress("lavender", fields) == (
            None,
            "lavender has no curl on its PATH (exit 127), so its probe cannot say whether it "
            "reaches the internet",
        )


class TestReadEgress:
    def test_every_target_is_read_in_the_order_asked(self) -> None:
        fields = {
            "egress_0": "6 https://pypi.org/simple/pip/",
            "egress_1": "0 https://registry.npmjs.org/",
            "egress_count": "2",
        }

        assert read_egress("lavender", fields) == (
            (
                {"target": "https://pypi.org/simple/pip/", "curl_exit": 6},
                {"target": "https://registry.npmjs.org/", "curl_exit": 0},
            ),
            "",
        )

    @pytest.mark.parametrize("curl_exit", [0, 1, 6, 7, 28, 35, 52, 60, 126, 128, 255])
    def test_any_exit_but_not_found_reads_back_as_printed(self, curl_exit: int) -> None:
        fields = {"egress_0": f"{curl_exit} https://registry.npmjs.org/", "egress_count": "1"}

        assert read_egress("loki", fields) == (
            ({"target": "https://registry.npmjs.org/", "curl_exit": curl_exit},),
            "",
        )

    @pytest.mark.parametrize(
        ("count", "shown"), [(None, "None"), ("two", "'two'"), ("0", "'0'"), ("-1", "'-1'")]
    )
    def test_a_count_that_is_not_a_positive_whole_number_is_unreadable(
        self, count: str | None, shown: str
    ) -> None:
        fields = {"egress_0": "0 https://pypi.org/simple/pip/"}
        if count is not None:
            fields["egress_count"] = count

        assert read_egress("loki", fields) == (
            None,
            f"loki reported egress_count={shown}, which is not a positive whole number of "
            "targets, so whether it reaches the internet is unknown",
        )

    def test_a_counted_target_with_no_line_is_unreadable(self) -> None:
        """A probe that ended among the targets prints no count; one that lost a line is caught."""
        fields = {"egress_0": "0 https://pypi.org/simple/pip/", "egress_count": "2"}

        assert read_egress("loki", fields) == (
            None,
            "loki counted 2 egress target(s) and printed no egress_1 line",
        )

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "0",
            "x https://pypi.org/simple/pip/",
            "07 https://pypi.org/",
            "1000 https://a/",
            "0  https://pypi.org/simple/pip/",
            "0 https://a/ trailing",
        ],
    )
    def test_a_line_not_an_exit_and_a_target_is_unreadable(self, line: str) -> None:
        fields = {"egress_0": line, "egress_count": "1"}

        assert read_egress("loki", fields) == (
            None,
            f"loki printed an egress line that cannot be read: {line!r}",
        )
