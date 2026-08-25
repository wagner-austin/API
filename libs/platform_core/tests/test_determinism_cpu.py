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

from platform_core.determinism_cpu import CPU_STACK, apply_cpu_determinism
from platform_core.determinism_env import BLAS_THREAD_ENV_VARS, SINGLE_THREAD
from platform_core.determinism_record import (
    TRUE,
    UNPINNED_STACK,
    decode_determinism_record,
    determinism_record,
    encode_determinism_record,
    render_determinism_record,
)


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

    apply_cpu_determinism(env, SINGLE_THREAD)

    assert env.writes == [(name, SINGLE_THREAD) for name in BLAS_THREAD_ENV_VARS]


def test_the_record_names_every_variable_that_was_set() -> None:
    env = RecordingEnv()

    record = apply_cpu_determinism(env, SINGLE_THREAD)

    assert record["stack"] == CPU_STACK
    assert dict(record["settings"]) == dict.fromkeys(BLAS_THREAD_ENV_VARS, SINGLE_THREAD)


def test_what_it_reports_is_what_it_wrote() -> None:
    # A record that claims a posture the process does not have is the exact
    # failure this module exists to prevent, so assert the two agree rather
    # than only that the record looks right.
    env = RecordingEnv()

    record = apply_cpu_determinism(env, SINGLE_THREAD)

    assert dict(env.writes) == dict(record["settings"])


def test_a_thread_count_above_one_is_recorded_rather_than_silently_blessed() -> None:
    # Serial reductions cost throughput. A caller may buy the throughput back,
    # and the record must say so instead of reading as deterministic.
    env = RecordingEnv()

    record = apply_cpu_determinism(env, "8")

    assert dict(record["settings"])["OMP_NUM_THREADS"] == "8"
    assert record != apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD)


def test_a_cpu_record_round_trips_through_storage() -> None:
    record = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD)

    assert decode_determinism_record(encode_determinism_record(record)) == record


def test_a_cpu_record_renders_as_its_own_stack() -> None:
    rendered = render_determinism_record(apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD))

    assert rendered.startswith(f"{CPU_STACK}[")
    assert "OMP_NUM_THREADS=1" in rendered


def test_a_cpu_run_never_compares_equal_to_a_torch_run_or_an_unpinned_one() -> None:
    # The whole point of carrying the stack in the record. Two runs that
    # pinned different things are different configurations, and a comparison
    # across them has to say so rather than find nothing to disagree about.
    cpu = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD)
    torch_like = determinism_record("torch", {"cudnn_deterministic": TRUE})
    unpinned = determinism_record(UNPINNED_STACK, {})

    assert cpu != torch_like
    assert cpu != unpinned
    assert len({render_determinism_record(r) for r in (cpu, torch_like, unpinned)}) == 3


def test_pinning_twice_with_the_same_count_is_the_same_record() -> None:
    # Canonical ordering means a re-run of the same pin is byte-identical,
    # which is what lets a fingerprint comparison find no difference.
    first = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD)
    second = apply_cpu_determinism(RecordingEnv(), SINGLE_THREAD)

    assert first == second
