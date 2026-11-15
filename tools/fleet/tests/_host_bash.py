"""What a host tool prints, asked of the tool under bash rather than retyped.

Support module, not a test module. A case whose expectation is a host
tool's own output (``sha256sum``'s line, what a rendered script prints) asks
the tool here, so the expectation is the tool's format and never a copy of
it typed into the test.
"""

from __future__ import annotations

import subprocess

#: How long one short command under :func:`bash_says` may take.
BASH_SAYS_SECONDS = 30


def bash_says(script: str, *arguments: str) -> str:
    """What a short script prints under bash.

    Args:
        script: The script text, reading its arguments as ``$1``, ``$2``...
        arguments: The script's arguments.

    Returns:
        Its standard output, stripped.

    Raises:
        AssertionError: When it exits non-zero, naming the script, the
            status and bash's stderr.
    """
    ran = subprocess.run(
        ["bash", "-c", script, "bash", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=BASH_SAYS_SECONDS,
    )
    if ran.returncode != 0:
        raise AssertionError(f"bash exited {ran.returncode} on {script!r}: {ran.stderr.strip()}")
    return ran.stdout.strip()
