"""Tests for remote command execution and byte transfer.

The quoting tests matter more than they look: a corpus filename holding a
quote or a semicolon would otherwise be interpreted by the remote shell, and
the resulting failure would present as a corrupt file rather than as a
quoting defect.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core.remote import make_directory, put_bytes, remote_digest, run_remote
from tests.conftest import FakeRun


class TestRunRemote:
    def test_it_returns_stdout_on_success(self, fake_run: FakeRun) -> None:
        fake_run.add("hostname", stdout="login-i16\n")
        assert run_remote("hpc3", "hostname") == "login-i16\n"

    def test_it_invokes_ssh_in_batch_mode(self, fake_run: FakeRun) -> None:
        run_remote("hpc3", "hostname")
        assert fake_run.calls[0].argv == ("ssh", "-o", "BatchMode=yes", "hpc3", "hostname")

    def test_a_non_zero_exit_carries_the_clusters_stderr(self, fake_run: FakeRun) -> None:
        fake_run.add("sbatch", returncode=1, stderr="Invalid account\n")
        with pytest.raises(AppError) as excinfo:
            run_remote("hpc3", "sbatch job.sbatch")
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED
        assert "Invalid account" in excinfo.value.message

    def test_a_silent_failure_says_so_rather_than_showing_nothing(self, fake_run: FakeRun) -> None:
        fake_run.add("sbatch", returncode=2)
        with pytest.raises(AppError) as excinfo:
            run_remote("hpc3", "sbatch job.sbatch")
        assert "<no stderr>" in excinfo.value.message
        assert "exited 2" in excinfo.value.message


class TestPutBytes:
    def test_it_streams_the_exact_bytes_through_stdin(self, fake_run: FakeRun) -> None:
        payload = b"\x00\x01binary\r\nnot-text\n"
        put_bytes("hpc3", "/pub/x/armB.txt", payload)
        assert fake_run.calls[0].stdin_bytes == payload

    def test_it_redirects_into_the_quoted_destination(self, fake_run: FakeRun) -> None:
        put_bytes("hpc3", "/pub/x/armB.txt", b"data")
        assert fake_run.calls[0].remote_command == "cat > '/pub/x/armB.txt'"

    def test_a_quote_in_the_path_is_escaped_not_interpreted(self, fake_run: FakeRun) -> None:
        put_bytes("hpc3", "/pub/it's/armB.txt", b"data")
        assert fake_run.calls[0].remote_command == "cat > '/pub/it'\\''s/armB.txt'"

    def test_a_failed_write_names_the_path(self, fake_run: FakeRun) -> None:
        fake_run.add("cat >", returncode=1, stderr="No space left on device\n")
        with pytest.raises(AppError) as excinfo:
            put_bytes("hpc3", "/pub/x/armB.txt", b"data")
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED
        assert "/pub/x/armB.txt" in excinfo.value.message
        assert "No space left" in excinfo.value.message


class TestMakeDirectory:
    def test_it_creates_parents(self, fake_run: FakeRun) -> None:
        make_directory("hpc3", "/pub/wagnera3/jobs")
        assert fake_run.calls[0].remote_command == "mkdir -p '/pub/wagnera3/jobs'"

    def test_a_failure_propagates(self, fake_run: FakeRun) -> None:
        fake_run.add("mkdir", returncode=1, stderr="Permission denied\n")
        with pytest.raises(AppError) as excinfo:
            make_directory("hpc3", "/root/nope")
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED


class TestRemoteDigest:
    def test_it_asks_for_sha256sum_of_the_quoted_path(self, fake_run: FakeRun) -> None:
        fake_run.add("sha256sum", stdout="abc  /pub/x/armB.txt\n")
        assert remote_digest("hpc3", "/pub/x/armB.txt") == "abc  /pub/x/armB.txt\n"
        assert fake_run.calls[0].remote_command == "sha256sum '/pub/x/armB.txt'"

    def test_a_missing_file_fails_the_command(self, fake_run: FakeRun) -> None:
        fake_run.add("sha256sum", returncode=1, stderr="No such file or directory\n")
        with pytest.raises(AppError) as excinfo:
            remote_digest("hpc3", "/pub/x/gone.txt")
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED
