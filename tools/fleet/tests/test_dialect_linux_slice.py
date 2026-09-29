"""The Linux capacity probe's ``runners.slice`` lines (MCPs board task 5d6e57e7).

A node whose CI runners run in ``runners.slice`` also reports the slice's
``memory.current`` and ``memory.high``, so a reservation refusal can name
CI rather than the owner. The cgroup path is fixed, so the text is pinned
here; run against the real nodes on 2026-09-29, lavender-wsl printed
``ci_slice_current_gb=16.000`` and ``ci_slice_high_gb=16.000`` and
diphtheria, which has no such cgroup, printed neither.
"""

from __future__ import annotations

from fleet.core.dialect_linux import LinuxDialect


def test_the_capacity_probe_reads_runners_slice_only_where_it_has_a_numeric_high() -> None:
    body = LinuxDialect().capacity_probe_script()
    lines = body.splitlines()
    start = lines.index("s=/sys/fs/cgroup/runners.slice")

    assert lines[start:] == [
        "s=/sys/fs/cgroup/runners.slice",
        'if [ -r "$s/memory.current" ] && [ -r "$s/memory.high" ]; then',
        '  h="$(cat "$s/memory.high")"',
        '  case "$h" in',
        "    ''|*[!0-9]*) ;;",
        '    *) awk -v c="$(cat "$s/memory.current")" -v h="$h" \'BEGIN { printf '
        '"ci_slice_current_gb=%.3f\\nci_slice_high_gb=%.3f\\n", c / 1073741824, '
        "h / 1073741824 }' ;;",
        "  esac",
        "fi",
    ]
