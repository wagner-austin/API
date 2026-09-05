"""Tests for remote command execution and byte transfer.

The quoting tests matter more than they look: a corpus filename holding a
quote or a semicolon would otherwise be interpreted by the remote shell, and
the resulting failure would present as a corrupt file rather than as a
quoting defect.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core.remote import (
    MAX_COMMAND_CHARS,
    make_directory,
    put_bytes,
    remote_digest,
    run_remote,
    run_remote_batched,
    token_batches,
)
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


class TestTokenBatches:
    def test_a_list_that_fits_stays_one_batch(self) -> None:
        """The ordinary case has to rebuild exactly the command this
        replaced: splitting a small query would turn one consistent moment
        into several."""
        assert token_batches(["101", "102", "103"], overhead=80, separator=",") == [
            ["101", "102", "103"]
        ]

    def test_no_tokens_produce_no_batches(self) -> None:
        """Callers refuse an empty list by name first; this must not invent
        a batch that would query everything."""
        assert token_batches([], overhead=80, separator=",") == []

    def test_it_splits_on_measured_width_not_on_a_token_count(self) -> None:
        """A count is a guess about token length. These two lists hold the
        same NUMBER of tokens and must split differently."""
        narrow = token_batches(["1234567890"] * 800, overhead=80, separator=",")
        wide = token_batches(["1234567890" * 4] * 800, overhead=80, separator=",")
        assert len(narrow) < len(wide)

    def test_every_batch_fits_the_limit_once_its_command_is_built(self) -> None:
        overhead = 80
        batches = token_batches(
            [f"55{index:06d}" for index in range(5000)], overhead=overhead, separator=","
        )
        assert all(overhead + len(",".join(batch)) <= MAX_COMMAND_CHARS for batch in batches)

    def test_the_separator_is_charged_between_tokens_and_not_before_them(self) -> None:
        """Charging a separator the joined command will not carry loses a
        token from a batch that had exactly enough room for it."""
        token = "a" * 100
        overhead = MAX_COMMAND_CHARS - 201
        assert token_batches([token, token], overhead=overhead, separator=",") == [[token, token]]
        assert token_batches([token, token], overhead=overhead + 1, separator=",") == [
            [token],
            [token],
        ]

    def test_order_survives_the_split(self) -> None:
        tokens = [f"55{index:06d}" for index in range(5000)]
        batches = token_batches(tokens, overhead=80, separator=",")
        assert [token for batch in batches for token in batch] == tokens

    def test_a_token_too_long_to_send_alone_names_itself_and_the_limit(self) -> None:
        """Emitting it anyway would rebuild the over-long command this exists
        to split, and fail again naming neither."""
        with pytest.raises(ValueError) as excinfo:
            token_batches(["z" * MAX_COMMAND_CHARS], overhead=80, separator=",")
        assert f"over the {MAX_COMMAND_CHARS}-character limit" in str(excinfo.value)
        assert "carrying 80 of overhead" in str(excinfo.value)


class TestRunRemoteBatched:
    def test_it_runs_every_batch(self, fake_run: FakeRun) -> None:
        run_remote_batched("hpc3", ["sacct -j 1", "sacct -j 2"])
        assert fake_run.commands() == ["sacct -j 1", "sacct -j 2"]

    def test_the_batches_come_back_as_one_stream(self, fake_run: FakeRun) -> None:
        fake_run.add("sacct -j 1", stdout="row-a\nrow-b\n", once=True)
        fake_run.add("sacct -j 2", stdout="row-c\n", once=True)
        assert run_remote_batched("hpc3", ["sacct -j 1", "sacct -j 2"]) == "row-a\nrow-b\nrow-c\n"

    def test_a_batch_without_a_trailing_newline_does_not_fuse_two_rows(
        self, fake_run: FakeRun
    ) -> None:
        """Concatenating raw stdout would make 'row-b' and 'row-c' one
        malformed row, which reads as a parse defect and eats both."""
        fake_run.add("sacct -j 1", stdout="row-a\nrow-b", once=True)
        fake_run.add("sacct -j 2", stdout="row-c\n", once=True)
        assert run_remote_batched("hpc3", ["sacct -j 1", "sacct -j 2"]) == "row-a\nrow-b\nrow-c\n"

    def test_no_batches_report_nothing(self, fake_run: FakeRun) -> None:
        assert run_remote_batched("hpc3", []) == ""
        assert fake_run.calls == []

    def test_one_failed_batch_fails_the_whole_query(self, fake_run: FakeRun) -> None:
        """A partial answer reads as the complete one, and the rows it is
        missing are the ones nobody goes looking for."""
        fake_run.add("sacct -j 2", returncode=1, stderr="Invalid job id specified\n")
        with pytest.raises(AppError) as excinfo:
            run_remote_batched("hpc3", ["sacct -j 1", "sacct -j 2"])
        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED
        assert "Invalid job id specified" in excinfo.value.message


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
