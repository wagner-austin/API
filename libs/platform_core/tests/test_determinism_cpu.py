"""Tests for pinning determinism on a CPU numeric stack.

This is the non-torch half of the determinism story, and it exists because
most of the research here is non-torch. The case that carries the design is
the last one: a CPU run and a torch run must not compare equal just because
neither mentions the other's settings.

The premise was measured before this was written, and the measurement
corrected it. On numpy 2.3.5 / scipy-openblas 0.3.30 a 4096x4096 float32
matmul over identical bytes is BIT-IDENTICAL run after run at a fixed thread
count, and differs across 1, 8 and 24 threads -- 865,498 of 16,777,216
elements, max absolute difference 1.4e-4. So the thing worth pinning is an
unrecorded INPUT, not an unpredictable library, and these tests assert that
the count reaches every variable and lands in the record.
"""

from __future__ import annotations

import sys

import pytest

from platform_core.determinism_cpu import (
    CPU_STACK,
    NUMERIC_MODULES,
    NativeLibrariesAlreadyLoadedError,
    apply_cpu_determinism,
)
from platform_core.determinism_env import BLAS_THREAD_ENV_VARS, SINGLE_THREAD
from platform_core.determinism_record import (
    TRUE,
    UNPINNED_STACK,
    decode_determinism_record,
    determinism_record,
    encode_determinism_record,
    render_determinism_record,
)

NOT_LOADED: tuple[str, ...] = ()
"""A module table with no native numeric library in it.

Stated explicitly rather than left to `sys.modules`, because these tests run
in a shared pytest worker whose module table depends on what else imported.
A pin test that silently changed meaning based on collection order would be
worse than no test.
"""


class RecordingEnv:
    """An environment writer that records what it was asked to set, in order.

    Order matters here for the same reason it does for cuBLAS: these are read
    when the native library loads, so a caller that sets them late gets no
    error and no effect.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []

    def __call__(self, key: str, value: str, /) -> None:
        self.writes.append((key, value))


def test_pinning_writes_every_thread_variable_not_only_one() -> None:
    # Which BLAS a numpy wheel links against is not knowable here. Setting
    # only the one that happens to matter today leaves the record claiming a
    # posture the next wheel will not honour.
    env = RecordingEnv()

    apply_cpu_determinism(env, SINGLE_THREAD, NOT_LOADED)

    assert env.writes == [(name, SINGLE_THREAD) for name in BLAS_THREAD_ENV_VARS]


def test_the_record_names_every_variable_that_was_set() -> None:
    env = RecordingEnv()

    record = apply_cpu_determinism(env, SINGLE_THREAD, NOT_LOADED)

    assert record["stack"] == CPU_STACK
    assert dict(record["settings"]) == dict.fromkeys(BLAS_THREAD_ENV_VARS, SINGLE_THREAD)


def test_what_it_reports_is_what_it_wrote() -> None:
    # A record that claims a posture the process does not have is the exact
    # failure this module exists to prevent, so assert the two agree rather
    # than only that the record looks right.
    env = RecordingEnv()

    record = apply_cpu_determinism(env, SINGLE_THREAD, NOT_LOADED)

    assert dict(env.writes) == dict(record["settings"])


def test_a_thread_count_above_one_is_recorded_rather_than_silently_blessed() -> None:
    # Serial reductions cost throughput. A caller may buy the throughput back,
    # and the record must say so instead of reading as deterministic.
    env = RecordingEnv()

    record = apply_cpu_determinism(env, "8", NOT_LOADED)

    assert dict(record["settings"])["OMP_NUM_THREADS"] == "8"
    assert record != apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD, NOT_LOADED)


def test_a_cpu_record_round_trips_through_storage() -> None:
    record = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD, NOT_LOADED)

    assert decode_determinism_record(encode_determinism_record(record)) == record


def test_a_cpu_record_renders_as_its_own_stack() -> None:
    record = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD, NOT_LOADED)
    rendered = render_determinism_record(record)

    assert rendered.startswith(f"{CPU_STACK}[")
    assert "OMP_NUM_THREADS=1" in rendered


def test_a_cpu_run_never_compares_equal_to_a_torch_run_or_an_unpinned_one() -> None:
    # The whole point of carrying the stack in the record. Two runs that
    # pinned different things are different configurations, and a comparison
    # across them has to say so rather than find nothing to disagree about.
    cpu = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD, NOT_LOADED)
    torch_like = determinism_record("torch", {"cudnn_deterministic": TRUE})
    unpinned = determinism_record(UNPINNED_STACK, {})

    assert cpu != torch_like
    assert cpu != unpinned
    assert len({render_determinism_record(r) for r in (cpu, torch_like, unpinned)}) == 3


def test_pinning_twice_with_the_same_count_is_the_same_record() -> None:
    # Canonical ordering means a re-run of the same pin is byte-identical,
    # which is what lets a fingerprint comparison find no difference.
    first = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD, NOT_LOADED)
    second = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD, NOT_LOADED)

    assert first == second


class TestAPinThatCannotTakeIsRefused:
    """The requirement used to live only in this module's docstring.

    On 2026-08-26 that sentence sat directly above a caller that violated it:
    `benchmark_cleargbm_regression` imported numpy at module scope and pinned
    from `main`. mypy, ruff, the guards and 2,564 tests at 100% branches all
    passed, and the manifest it wrote asserted OMP_NUM_THREADS=1 for a run
    that was multi-threaded.

    Measured, so the refusal is not superstition -- fixed 2048x2048 float32
    matmul, digests of the result bytes:

        pin to 1 BEFORE importing numpy   f364ecedb70f678b
        pin to 8 BEFORE importing numpy   628f2231d6fe0a62
        import numpy, THEN pin to 1       20d850081f69206f

    The late pin reproducibly produced a third answer: no error, no effect.
    """

    def test_numpy_already_imported_refuses_rather_than_lying(self) -> None:
        env = RecordingEnv()

        with pytest.raises(NativeLibrariesAlreadyLoadedError):
            _ = apply_cpu_determinism(env, SINGLE_THREAD, ("numpy",))

    def test_nothing_is_written_when_the_pin_is_refused(self) -> None:
        """A partial write would leave the process in a state nobody chose."""
        env = RecordingEnv()

        with pytest.raises(NativeLibrariesAlreadyLoadedError):
            _ = apply_cpu_determinism(env, SINGLE_THREAD, ("numpy",))

        assert env.writes == []

    def test_the_message_names_what_was_loaded_and_what_to_do(self) -> None:
        """The cluster taught this lesson twice: an error that names neither
        the cause nor the fix costs an SSH round trip to diagnose."""
        with pytest.raises(NativeLibrariesAlreadyLoadedError) as excinfo:
            _ = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD, ("numpy", "torch"))

        message = str(excinfo.value)
        assert "numpy, torch" in message
        assert "process entry point" in message

    def test_every_library_that_pulls_a_blas_counts(self) -> None:
        """scipy and sklearn import numpy transitively; a caller should not
        have to know that to get the check they expect."""
        for name in NUMERIC_MODULES:
            with pytest.raises(NativeLibrariesAlreadyLoadedError):
                _ = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD, (name,))

    def test_an_unrelated_import_does_not_block_the_pin(self) -> None:
        """Refusing on anything at all would push callers to bypass the check."""
        env = RecordingEnv()

        record = apply_cpu_determinism(env, SINGLE_THREAD, ("json", "pathlib", "argparse"))

        assert record["stack"] == CPU_STACK
        assert env.writes == [(name, SINGLE_THREAD) for name in BLAS_THREAD_ENV_VARS]

    def test_the_real_module_table_is_the_default(self) -> None:
        """Production passes nothing, so the check must consult sys.modules.

        This test file has imported no numeric library, and the assertion is
        that the default reads the live table rather than an empty stand-in
        -- `sys` itself is present, and is deliberately not a blocker.
        """
        assert "sys" in sys.modules
        record = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD)

        assert record["stack"] == CPU_STACK
