"""Dependency-injection seam for the CLI layer.

The CLI's only impure act beyond what the core already routes through hooks
is writing its summary to stdout. It goes through a hook so a test can assert
what a run reported without capturing process output, which keeps the
assertions about the summary's content rather than about pytest's capture
behaviour.

The distributions a run fingerprints are here for a different reason. A real
scoring run must record the corpus group (see ``core.provenance``), and
``installed_version`` refuses a distribution that is absent -- correctly, since
a record naming a narrower instrument than the one that ran is the defect the
refusal exists to prevent. But ``poetry sync --with dev`` removes that optional
group before every ``make check``, so a test exercising this CLI would have to
install ~2 GB to reach one line of it. Rebinding the seam is how the test names
a set it has, without the production default becoming a lie.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from code_style_eval.core.provenance import FINGERPRINT_DISTRIBUTIONS


def _default_emit(line: str) -> None:
    """Write one summary line to stdout.

    Args:
        line: Line to write, without a trailing newline.
    """
    sys.stdout.write(line + "\n")


emit: Callable[[str], None] = _default_emit

#: Distributions a comparison records. Production names the real set.
record_distributions: tuple[str, ...] = FINGERPRINT_DISTRIBUTIONS


def reset_hooks() -> None:
    """Rebind every hook to its production implementation."""
    global emit, record_distributions
    emit = _default_emit
    record_distributions = FINGERPRINT_DISTRIBUTIONS


__all__ = ["emit", "record_distributions", "reset_hooks"]
