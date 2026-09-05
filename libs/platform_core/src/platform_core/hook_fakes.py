"""Fakes for the hook seams a command-line tool binds.

Split from :mod:`platform_core.testing` when that module crossed the 600-line
ceiling. The seam is by ROLE rather than by size: everything here fakes an
effect a CLI performs -- an HTTP call, the clock, the filesystem, the console
-- while ``testing`` holds the httpx client doubles, environment and host
probes that a service uses.

platform_calendar and platform_email each had their own copy of every fake
below -- thirteen functions, bodies identical -- and each declared its own
six hook type aliases to spell their signatures. Two libraries, one set of
seams, and no way for a fix in one to reach the other.

Spelled with plain Callable types rather than aliases. The aliases were the
duplication in its purest form: `ReadFileHook = Callable[[str], str]` written
twice, and the workspace bans a type alias for exactly the reason it bans
these bodies. A caller that wants a name for one binds the returned function
to its own annotated hook slot, which is what it does anyway.
"""

from __future__ import annotations

from collections.abc import Callable


def make_fake_http_get(response: str) -> Callable[[str, dict[str, str]], str]:
    """Build a hook for the GET shape that answers with a fixed body.

    Three shapes cover the four verbs: GET reads with no body, POST and PATCH
    send one, DELETE returns nothing. Spelled out rather than collapsed into a
    single ``Callable[..., str]``, which is an implicit Any and refused here.

    Args:
        response: Body every call returns.

    Returns:
        A hook taking (url, headers).
    """

    def _hook(url: str, headers: dict[str, str]) -> str:
        return response

    return _hook


def make_fake_http_send(response: str) -> Callable[[str, dict[str, str], str], str]:
    """Build a hook for the POST and PATCH shape.

    Args:
        response: Body every call returns.

    Returns:
        A hook taking (url, headers, body).
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        return response

    return _hook


def make_fake_http_delete() -> Callable[[str, dict[str, str]], None]:
    """Build a hook for the DELETE shape, which answers nothing.

    Returns:
        A hook taking (url, headers) and returning None.
    """

    def _hook(url: str, headers: dict[str, str]) -> None:
        return None

    return _hook


def make_raising_http_get(exc: BaseException) -> Callable[[str, dict[str, str]], str]:
    """Build a GET-shaped hook that raises instead of answering.

    Args:
        exc: The exception to raise.

    Returns:
        A hook taking (url, headers).

    Raises:
        BaseException: Always, when called. Putting a transport failure in
            front of the code under test is the hook's only purpose.
    """

    def _hook(url: str, headers: dict[str, str]) -> str:
        raise exc

    return _hook


def make_raising_http_send(exc: BaseException) -> Callable[[str, dict[str, str], str], str]:
    """Build a POST/PATCH-shaped hook that raises instead of answering.

    Args:
        exc: The exception to raise.

    Returns:
        A hook taking (url, headers, body).

    Raises:
        BaseException: Always, when called.
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        raise exc

    return _hook


def make_raising_http_delete(exc: BaseException) -> Callable[[str, dict[str, str]], None]:
    """Build a DELETE-shaped hook that raises instead of answering.

    Args:
        exc: The exception to raise.

    Returns:
        A hook taking (url, headers).

    Raises:
        BaseException: Always, when called.
    """

    def _hook(url: str, headers: dict[str, str]) -> None:
        raise exc

    return _hook


def make_fake_current_time(timestamp: int) -> Callable[[], int]:
    """Build a clock hook that always reports the same instant.

    Args:
        timestamp: Unix seconds every call returns.

    Returns:
        A callable returning ``timestamp``.
    """

    def _hook() -> int:
        return timestamp

    return _hook


def make_fake_file_system(
    files: dict[str, str],
) -> tuple[Callable[[str], str], Callable[[str, str], None], Callable[[str], bool]]:
    """Build read, write and exists hooks over an in-memory file system.

    Args:
        files: Initial contents keyed by path.

    Returns:
        A tuple of (read, write, exists) hooks sharing one store, so a write
        is visible to a later read -- which is the only reason to have the
        three from one call rather than separately.
    """
    storage = dict(files)

    def _read(path: str) -> str:
        if path not in storage:
            raise FileNotFoundError(f"File not found: {path}")
        return storage[path]

    def _write(path: str, content: str) -> None:
        storage[path] = content

    def _exists(path: str) -> bool:
        return path in storage

    return _read, _write, _exists


def make_fake_console(
    inputs: list[str],
) -> tuple[Callable[[str], None], Callable[[str], str]]:
    """Build output and input hooks for a scripted console.

    Args:
        inputs: Answers the input hook returns, in order.

    Returns:
        A tuple of (output, input) hooks.
    """
    outputs: list[str] = []
    index = [0]

    def _output(message: str) -> None:
        outputs.append(message)

    def _input(prompt: str) -> str:
        # Both copies used to return "" forever once the script ran out, and
        # both had a test pinning that: a test consuming more prompts than it
        # scripted passed while the code under test read an empty answer no
        # person would have given it.
        if index[0] >= len(inputs):
            raise AssertionError(
                f"console_input asked for {prompt!r} but the fake was scripted "
                f"with only {len(inputs)} answer(s)"
            )
        answer = inputs[index[0]]
        index[0] += 1
        return answer

    return _output, _input


__all__ = [
    "make_fake_console",
    "make_fake_current_time",
    "make_fake_file_system",
    "make_fake_http_delete",
    "make_fake_http_get",
    "make_fake_http_send",
    "make_raising_http_delete",
    "make_raising_http_get",
    "make_raising_http_send",
]
