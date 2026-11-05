"""Which NixOS VM tests a check ran, and each one's test-script seconds.

MCPs board task d6b03ba3. MCPs/fleet-nixos's checks are VM tests, each built
by nix as a ``vm-test-run-<name>`` derivation whose driver ends with ``test
script finished in <s>s``, a line nix prefixes with the derivation's name.
Only a test the run built prints it: one already in the store, or
substituted, is not run. Those lines are spread through a transcript of some
40,000 lines, past the tail the verdict reads, so the audit of that task's
closure (7de4c7a5) found which tests a cold run built, and how long each
took, in a session's grep of the log and in no record.

So a check's build reads its own transcript once the recipe has ended and
prints one :data:`VM_TEST_MARKER` line per finished test and a
:data:`VM_TESTS_MARKER` line counting them, which the verdict
(:func:`read_vm_tests`, in :func:`fleet.core.verdict.judge`) copies onto the
queue row and the thread as ``vmtests=<count>:<name>=<s>s,...``.
"""

from __future__ import annotations

import re
from typing import Final

#: One finished test's line.
VM_TEST_MARKER: Final = "FLEET_VM_TEST"

#: The count's line, printed whether or not any test ran.
VM_TESTS_MARKER: Final = "FLEET_VM_TESTS"

#: nix's prefix and the NixOS test driver's last line, as ``sed -E`` reads them.
FINISHED_SED: Final = r"^vm-test-run-([^>[:space:]]+)> test script finished in ([0-9.]+)s$"

#: A finished test's line, as the build prints it.
VM_TEST_LINE: Final[re.Pattern[str]] = re.compile(
    rf"^{VM_TEST_MARKER}: (\S+) (\d+(?:\.\d+)?)$", re.MULTILINE
)

#: The count's line.
VM_TESTS_LINE: Final[re.Pattern[str]] = re.compile(
    rf"^{VM_TESTS_MARKER}: (\d+) finished$", re.MULTILINE
)


def sh_vm_test_lines(log: str) -> list[str]:
    """The ``sh`` lines that print the transcript's finished VM tests into it.

    The transcript is read whole before anything is appended, so the lines
    never read their own output.

    Args:
        log: The transcript's path, read and appended to.

    Returns:
        The lines, run after the recipe's phase and before the status is written.
    """
    return [
        f"vm_tests=\"$(sed -nE 's/{FINISHED_SED}/{VM_TEST_MARKER}: \\1 \\2/p' '{log}')\"",
        f"if [ -n \"$vm_tests\" ]; then printf '%s\\n' \"$vm_tests\" >> '{log}'; fi",
        f"printf '{VM_TESTS_MARKER}: %s finished\\n' "
        f"\"$(printf '%s' \"$vm_tests\" | grep -c '^{VM_TEST_MARKER}: ' || true)\" >> '{log}'",
    ]


def read_vm_tests(tail: str) -> str:
    """Read the VM tests a check ran off its transcript tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        ``<count>:<name>=<s>s,...``, the tests by name, from the last
        :data:`VM_TESTS_LINE` and the :data:`VM_TEST_LINE` lines between it
        and the count line before it; ``0`` when it ran none; else
        ``unread`` (a build from before the lines, or one stopped before
        them).
    """
    counts = list(VM_TESTS_LINE.finditer(tail))
    if not counts:
        return "unread"
    counted = counts[-1]
    since = counts[-2].end() if len(counts) > 1 else 0
    tests = sorted(
        (match.group(1), match.group(2))
        for match in VM_TEST_LINE.finditer(tail, since, counted.start())
    )
    if not tests:
        return counted.group(1)
    return f"{counted.group(1)}:" + ",".join(f"{name}={seconds}s" for name, seconds in tests)


__all__ = [
    "FINISHED_SED",
    "VM_TESTS_LINE",
    "VM_TESTS_MARKER",
    "VM_TEST_LINE",
    "VM_TEST_MARKER",
    "read_vm_tests",
    "sh_vm_test_lines",
]
