"""Streaming file wrapper for run artifact downloads."""

from __future__ import annotations

import io
import types
from typing import Protocol


class _BinaryFileProto(Protocol):
    """Protocol for file-like objects needed by SSE streaming."""

    def seek(self, offset: int, whence: int = 0) -> int: ...
    def readline(self) -> bytes: ...
    def __enter__(self) -> _BinaryFileProto: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool | None: ...
    def __iter__(self) -> _BinaryFileProto: ...
    def __next__(self) -> bytes: ...


class _BinaryFileWrapper:
    """Wrapper around BufferedReader that properly implements _BinaryFileProto."""

    _f: io.BufferedReader

    def __init__(self, path: str, mode: str) -> None:
        raw = io.FileIO(path, mode)
        self._f = io.BufferedReader(raw)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._f.seek(offset, whence)

    def readline(self) -> bytes:
        return self._f.readline()

    def __enter__(self) -> _BinaryFileWrapper:
        self._f.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool | None:
        self._f.__exit__(exc_type, exc_val, exc_tb)
        return None

    def __iter__(self) -> _BinaryFileWrapper:
        return self

    def __next__(self) -> bytes:
        line = self._f.readline()
        if not line:
            raise StopIteration
        return line
