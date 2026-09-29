"""The launcher's own log stream writes any character (MCPs board task 88b8fe61).

On 2026-09-29 :mod:`fleet.cli.rolled` relayed a node agent's traceback that
held U+FFFD, its standard output was a cp1252 pipe, and logging printed
``--- Logging error --- UnicodeEncodeError`` in place of the traceback. Each
test here gives the launcher a real cp1252 text stream over bytes as its
standard output, runs its real logging setup, logs through the handler that
setup attaches, and reads the bytes back.
"""

from __future__ import annotations

import io
import logging
import sys
import types

import pytest

from fleet.cli import rolled as rolled_cli

#: A relayed line no cp1252 stream can write: the replacement character, a
#: CJK ideograph, and the em dash the queue's refusal carried.
OUTSIDE_CP1252 = "stderr � 中 — end"


class _Cp1252Stdout:
    """Make standard output a cp1252 text stream, then restore it and the root logger.

    Entered inside the test body rather than set up as a fixture: pytest's
    capture re-installs its own ``sys.stdout`` when it resumes after fixture
    setup, so a stream swapped in by a fixture is gone by the time the test
    runs. A class rather than a ``@contextmanager`` generator because the
    ``import-collections-iterator`` guard forbids the annotation a generator
    needs.
    """

    def __init__(self) -> None:
        """Prepare the stream and remember what to restore."""
        self.raw = io.BytesIO()
        self._saved_stdout = sys.stdout
        root = logging.getLogger()
        self._saved_handlers = list(root.handlers)
        self._saved_level = root.level

    def __enter__(self) -> io.BytesIO:
        """Install the cp1252 stream as standard output.

        Returns:
            The bytes under the stream.
        """
        sys.stdout = io.TextIOWrapper(self.raw, encoding="cp1252", errors="strict")
        return self.raw

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Restore standard output and the root logger's handlers and level.

        Args:
            exc_type: The exception's type, if the body raised.
            exc: The exception, if the body raised.
            traceback: Its traceback, if the body raised.
        """
        sys.stdout = self._saved_stdout
        root = logging.getLogger()
        root.handlers[:] = self._saved_handlers
        root.setLevel(self._saved_level)


def test_a_relayed_line_outside_cp1252_is_written_whole_as_utf8() -> None:
    # Read inside the block: restoring stdout drops the wrapper, which closes
    # the bytes under it when it is collected.
    with _Cp1252Stdout() as raw:
        rolled_cli.configure_logging()
        logging.getLogger(rolled_cli.__name__).info("%s", OUTSIDE_CP1252)
        sys.stdout.flush()
        written = raw.getvalue().decode(rolled_cli.LOG_ENCODING)

    assert OUTSIDE_CP1252 in written
    assert "Logging error" not in written


def test_without_it_the_same_line_cannot_be_written() -> None:
    """The failure this guards, on the same stream shape, so the test above can fail."""
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")

    with pytest.raises(UnicodeEncodeError):
        stream.write(OUTSIDE_CP1252)


def test_a_stream_whose_encoding_cannot_be_set_is_refused_by_name() -> None:
    with pytest.raises(TypeError) as excinfo:
        rolled_cli.encode_log_stream(io.StringIO())

    assert str(excinfo.value) == (
        "FLEET_LOG_STREAM_UNCONFIGURABLE: the launcher's output is a StringIO, "
        "whose encoding cannot be set to utf-8"
    )
