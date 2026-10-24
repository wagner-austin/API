"""The session observer a Linux node is handed: its text, its python3 body
run by this interpreter, and the whole script run under ``sh``.

Split from ``test_dialect_linux.py`` when the observer gained the node's own
``pidDomain`` (board task 5cee2f65) and that module passed the 600-line
ceiling. Claude Code 2.1.294 spells ``pidDomain`` on Linux as ``linux:``, the
``/etc/machine-id``, then the pid namespace ``/proc/self/ns/pid`` links to,
so every case that runs the body reads those two files and is bound to a
Linux host, where ``tools/fleet-execution-linux`` runs it.
"""

from __future__ import annotations

import os
import pathlib
import re
import socket
import subprocess
import sys

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, load_json_str

from fleet.core.dialect_linux import (
    OBSERVE_SESSIONS_SCRIPT,
    PROLOGUE,
    SH_INVOCATION,
    LinuxDialect,
)
from fleet.core.linux_observe_sessions import OBSERVE_SESSIONS_PYTHON

DIALECT = LinuxDialect()

#: The shape of the pidDomain 2.1.294 wrote on ziggurat on 2026-10-16,
#: ``linux:f191c99155c14a719632bdc88e77f914:pid:[4026531836]``.
HARNESS_PID_DOMAIN = re.compile(r"linux:[0-9a-f]{32}:pid:\[\d+\]")

#: One registration document in the shape the harness writes, as a Linux
#: node would hold it: a POSIX cwd, a socket under the user's runtime dir and
#: the 2.1.294 pidDomain (diphtheria's machine-id, measured 2026-10-16).
SESSION_RECORD: JSONObject = {
    "sessionId": "d31e5228-3269-4a22-a27f-e535cbef1894",
    "pid": 4242,
    "startedAt": 1_789_000_000_000,
    "updatedAt": 1_789_000_030_500,
    "name": "api-7e",
    "nameSource": "derived",
    "cwd": "/home/corvis/PROJECTS/API",
    "messagingSocketPath": "/run/user/1000/claude-cc-msg-abc",
    "version": "2.1.294",
    "status": "idle",
    "pidDomain": "linux:212dd22057a34d6587f019c387979599:pid:[4026531836]",
}


def home_with_sessions(tmp_path: pathlib.Path, *records: JSONObject) -> dict[str, str]:
    """Lay out a home directory holding these session records.

    Args:
        tmp_path: The directory that becomes the home.
        *records: The registration documents, written as ``<pid>.json``.

    Returns:
        The environment that makes ``~`` resolve there under both
        ``os.path.expanduser`` spellings (``HOME`` on POSIX, ``USERPROFILE``
        on Windows), over the parent environment read through the
        monorepo's one permitted reader so the interpreter still starts.
    """
    if records:
        sessions = tmp_path / ".claude" / "sessions"
        sessions.mkdir(parents=True)
        for record in records:
            (sessions / f"{record['pid']}.json").write_text(dump_json_str(record), encoding="utf-8")
    parent = config_test_hooks.get_environment()
    return {**parent, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}


def this_node_pid_domain() -> str:
    """The pidDomain this Linux machine's harness writes, read directly.

    Returns:
        ``linux:<machine-id>:<pid namespace link>``.
    """
    machine_id = pathlib.Path("/etc/machine-id").read_text(encoding="utf-8").strip()
    return f"linux:{machine_id}:{os.readlink('/proc/self/ns/pid')}"


def test_the_observe_script_feeds_python3_through_a_quoted_heredoc() -> None:
    """sh cannot join files into one JSON array that survives an empty
    directory or a quote in a field; python3 can, and it is on every
    Linux node by the toolchain contract (board task cd5010c4)."""
    body = DIALECT.observe_sessions_script()

    assert body == OBSERVE_SESSIONS_SCRIPT
    assert body == f"{PROLOGUE}python3 - <<'PY'\n{OBSERVE_SESSIONS_PYTHON}PY\n"
    assert "'.claude', 'sessions'" in OBSERVE_SESSIONS_PYTHON
    assert "if os.path.isdir(directory):" in OBSERVE_SESSIONS_PYTHON
    assert "sorted(os.listdir(directory))" in OBSERVE_SESSIONS_PYTHON
    assert "'platform': sys.platform" in OBSERVE_SESSIONS_PYTHON
    assert "'hostname': socket.gethostname().lower()" in OBSERVE_SESSIONS_PYTHON
    assert "open('/etc/machine-id', encoding='utf-8')" in OBSERVE_SESSIONS_PYTHON
    assert (
        "'pidDomain': 'linux:' + machine_id + ':' + os.readlink('/proc/self/ns/pid')"
        in OBSERVE_SESSIONS_PYTHON
    )
    assert "separators=(',', ':')" in OBSERVE_SESSIONS_PYTHON


@pytest.mark.host_linux
class TestObserveBodyForReal:
    """The python3 body of the observe script, run by this interpreter.

    The sh around it is one heredoc, run whole by the class below; the body
    is what reads the directory and builds the document.
    """

    def run_body(self, environment: dict[str, str]) -> JSONObject:
        """Run the body exactly as the heredoc feeds it, and decode its line.

        Args:
            environment: The environment, with the home to read.

        Returns:
            The decoded document.
        """
        completed = subprocess.run(
            [sys.executable, "-"],
            input=OBSERVE_SESSIONS_PYTHON,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr
        document: JSONValue = load_json_str(completed.stdout)
        if not isinstance(document, dict):
            raise AssertionError(f"the script printed {type(document).__name__}, not an object")
        return document

    def test_it_reports_every_record_verbatim_with_the_node_s_own_pid_domain(
        self, tmp_path: pathlib.Path
    ) -> None:
        second: JSONObject = {
            **SESSION_RECORD,
            "pid": 31868,
            "sessionId": "06e0b3cd-6de1-45ec-8aa2-c3b48a7f60fc",
        }
        expected_pid_domain = this_node_pid_domain()

        document = self.run_body(home_with_sessions(tmp_path, second, SESSION_RECORD))

        assert HARNESS_PID_DOMAIN.fullmatch(expected_pid_domain), expected_pid_domain
        assert document == {
            "platform": sys.platform,
            "hostname": socket.gethostname().lower(),
            "pidDomain": expected_pid_domain,
            "records": [second, SESSION_RECORD],
        }

    def test_it_reports_zero_records_for_a_home_that_never_ran_claude_code(
        self, tmp_path: pathlib.Path
    ) -> None:
        """diphtheria on 2026-09-21: no ``~/.claude`` at all. That is a node
        with no sessions, not a fault."""
        document = self.run_body(home_with_sessions(tmp_path))

        assert document["records"] == []
        assert document["platform"] == sys.platform
        assert document["pidDomain"] == this_node_pid_domain()


@pytest.mark.host_linux
def test_the_whole_observe_script_under_sh_reports_the_records_under_the_home(
    tmp_path: pathlib.Path,
) -> None:
    """The whole script under sh: the prologue, the heredoc, the body."""
    home = tmp_path / "home"
    home.mkdir()
    script = tmp_path / "observe-sessions.sh"
    script.write_text(DIALECT.observe_sessions_script(), encoding="utf-8")

    completed = subprocess.run(
        [*SH_INVOCATION, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=home_with_sessions(home, SESSION_RECORD),
    )

    assert completed.returncode == 0, completed.stderr
    assert load_json_str(completed.stdout) == {
        "platform": sys.platform,
        "hostname": socket.gethostname().lower(),
        "pidDomain": this_node_pid_domain(),
        "records": [SESSION_RECORD],
    }
