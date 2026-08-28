"""Tests for kernel-level determinism controls.

The doubles here are real classes satisfying the module's Protocols, not
patches. Each records the ORDER of what it was asked to do, because ordering
carries the correctness here: ``CUBLAS_WORKSPACE_CONFIG`` is read once when
the cuBLAS handle is created, so writing it after a CUDA call is accepted
silently and does nothing. A test that only checked the final values would
pass on a version that set it last.
"""

from __future__ import annotations

from platform_core.determinism_cpu import BLAS_THREAD_ENV_VARS
from platform_core.determinism_record import (
    FALSE,
    TRUE,
    UNPINNED_STACK,
    DeterminismRecord,
    decode_determinism_record,
    determinism_record,
    encode_determinism_record,
)

from platform_ml.determinism import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    CUBLAS_WORKSPACE_ENV_VAR,
    CUBLASLT_NO_SPLIT_K,
    CUBLASLT_WORKSPACE_ENV_VAR,
    SPLIT_K_REMOVED,
    SPLIT_K_SETTING,
    TORCH_STACK,
    TORCH_THREAD_SETTING,
    apply_determinism,
    remove_cublaslt_split_k,
    set_cublas_workspace,
    with_torch_thread_count,
)


class RecordingEnv:
    """An environment writer that records writes into a SHARED log.

    The log is passed in rather than owned, so environment writes and torch
    writes interleave in one ordering. An earlier version of this double kept
    its own list, which made the ordering assertion structurally unable to
    fail -- it was comparing against a log the env had never written to.
    """

    def __init__(self, log: list[str]) -> None:
        self.values: dict[str, str] = {}
        self.log = log

    def __call__(self, key: str, value: str) -> None:
        self.values[key] = value
        self.log.append(f"env:{key}={value}")


class RecordingMatmul:
    """``torch.backends.cuda.matmul`` double."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self._allow_tf32 = True

    @property
    def allow_tf32(self) -> bool:
        return self._allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value: bool) -> None:
        self._allow_tf32 = value
        self._log.append(f"matmul.allow_tf32={value}")


class RecordingCudnn:
    """``torch.backends.cudnn`` double."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self._allow_tf32 = True
        self._deterministic = False
        self._benchmark = True

    @property
    def allow_tf32(self) -> bool:
        return self._allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value: bool) -> None:
        self._allow_tf32 = value
        self._log.append(f"cudnn.allow_tf32={value}")

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @deterministic.setter
    def deterministic(self, value: bool) -> None:
        self._deterministic = value
        self._log.append(f"cudnn.deterministic={value}")

    @property
    def benchmark(self) -> bool:
        return self._benchmark

    @benchmark.setter
    def benchmark(self, value: bool) -> None:
        self._benchmark = value
        self._log.append(f"cudnn.benchmark={value}")


class RecordingTorch:
    """The three leaf objects apply_determinism writes, bundled for tests.

    Not a ``torch`` module double: the function takes the leaves directly, so
    this only exists to build them from one shared log and to expose what was
    recorded.
    """

    def __init__(self, log: list[str]) -> None:
        self.cudnn = RecordingCudnn(log)
        self.matmul = RecordingMatmul(log)
        self._log = log
        self.deterministic_calls: list[bool] = []

    def use_deterministic_algorithms(self, mode: bool) -> None:
        self.deterministic_calls.append(mode)
        self._log.append(f"use_deterministic_algorithms={mode}")

    def apply(self, env: RecordingEnv, *, remove_split_k: bool) -> DeterminismRecord:
        """Call the function under test with this bundle's leaves."""
        return apply_determinism(
            self.cudnn,
            self.matmul,
            self.use_deterministic_algorithms,
            env,
            remove_split_k=remove_split_k,
        )


def test_apply_determinism_returns_every_field_it_set() -> None:
    log: list[str] = []
    torch = RecordingTorch(log)
    env = RecordingEnv(log)

    record = torch.apply(env, remove_split_k=False)

    assert record == {
        "stack": TORCH_STACK,
        "settings": (
            ("cublas_workspace_config", CUBLAS_DETERMINISTIC_WORKSPACE),
            ("cudnn_benchmark", FALSE),
            ("cudnn_deterministic", TRUE),
            ("cudnn_tf32", FALSE),
            ("deterministic_algorithms", TRUE),
            ("matmul_tf32", FALSE),
        ),
    }


def test_leaving_split_k_alone_adds_no_setting_and_writes_no_variable() -> None:
    # The property the deployed known-answer registry depends on. Every entry
    # in it was registered from a record with no split-K setting, so if the
    # control arm gained a key -- even one saying "not removed" -- every
    # future probe would report configuration_differs against all of them.
    # Absence is what keeps that registry gating, and it is load-bearing.
    log: list[str] = []
    env = RecordingEnv(log)

    record = RecordingTorch(log).apply(env, remove_split_k=False)

    assert SPLIT_K_SETTING not in dict(record["settings"])
    assert CUBLASLT_WORKSPACE_ENV_VAR not in env.values


def test_removing_split_k_writes_the_variable_and_records_that_it_did() -> None:
    log: list[str] = []
    env = RecordingEnv(log)

    record = RecordingTorch(log).apply(env, remove_split_k=True)

    assert dict(record["settings"])[SPLIT_K_SETTING] == SPLIT_K_REMOVED
    assert env.values[CUBLASLT_WORKSPACE_ENV_VAR] == CUBLASLT_NO_SPLIT_K


def test_the_two_split_k_postures_produce_different_records() -> None:
    # The point of recording it at all: two runs that compute different
    # numbers must not compare as identically configured. Before this
    # setting existed they did, and which arm produced which artifact was
    # recoverable only from the filename.
    left = RecordingTorch([]).apply(RecordingEnv([]), remove_split_k=True)
    right = RecordingTorch([]).apply(RecordingEnv([]), remove_split_k=False)

    assert left != right


def test_apply_determinism_writes_the_env_var_before_touching_cuda() -> None:
    # The whole point: CUBLAS_WORKSPACE_CONFIG is read when the cuBLAS handle
    # is created. If any CUDA-touching call precedes it, the setting is a
    # no-op and the run is nondeterministic while reporting that it is not.
    log: list[str] = []
    torch = RecordingTorch(log)
    env = RecordingEnv(log)

    torch.apply(env, remove_split_k=False)

    assert log[0] == f"env:{CUBLAS_WORKSPACE_ENV_VAR}={CUBLAS_DETERMINISTIC_WORKSPACE}"


def test_apply_determinism_pins_every_flag_the_report_claims() -> None:
    # Asserts the ACTUAL post-state of the doubles, not only the returned
    # report, so a report that lies about what it did fails here.
    log: list[str] = []
    torch = RecordingTorch(log)
    env = RecordingEnv(log)

    torch.apply(env, remove_split_k=False)

    assert torch.matmul.allow_tf32 is False
    assert torch.cudnn.allow_tf32 is False
    assert torch.cudnn.deterministic is True
    assert torch.cudnn.benchmark is False
    assert torch.deterministic_calls == [True]
    assert env.values[CUBLAS_WORKSPACE_ENV_VAR] == CUBLAS_DETERMINISTIC_WORKSPACE


def test_apply_determinism_touches_exactly_these_settings() -> None:
    # A shape-drift guard. If someone adds a flag to apply_determinism without
    # adding it to the returned record, the run record stops describing
    # the run, which is the failure this module exists to prevent.
    log: list[str] = []
    RecordingTorch(log).apply(RecordingEnv(log), remove_split_k=False)

    assert log == [
        f"env:{CUBLAS_WORKSPACE_ENV_VAR}={CUBLAS_DETERMINISTIC_WORKSPACE}",
        "matmul.allow_tf32=False",
        "cudnn.allow_tf32=False",
        "cudnn.deterministic=True",
        "cudnn.benchmark=False",
        "use_deterministic_algorithms=True",
    ]


def test_both_env_writes_precede_every_cuda_touching_call() -> None:
    # Ordering is the entire correctness condition for both variables: each is
    # read once, when its library creates its handle, and a write that lands
    # after is accepted in silence. So this asserts the full sequence rather
    # than only the first element -- a split-K write that slipped below
    # `matmul.allow_tf32` would still write the right value, still record the
    # right setting, and still do nothing.
    log: list[str] = []
    RecordingTorch(log).apply(RecordingEnv(log), remove_split_k=True)

    assert log == [
        f"env:{CUBLAS_WORKSPACE_ENV_VAR}={CUBLAS_DETERMINISTIC_WORKSPACE}",
        f"env:{CUBLASLT_WORKSPACE_ENV_VAR}={CUBLASLT_NO_SPLIT_K}",
        "matmul.allow_tf32=False",
        "cudnn.allow_tf32=False",
        "cudnn.deterministic=True",
        "cudnn.benchmark=False",
        "use_deterministic_algorithms=True",
    ]


def test_remove_cublaslt_split_k_writes_zero_workspace() -> None:
    env = RecordingEnv([])

    remove_cublaslt_split_k(env)

    assert env.values == {CUBLASLT_WORKSPACE_ENV_VAR: CUBLASLT_NO_SPLIT_K}


def test_set_cublas_workspace_returns_what_it_wrote() -> None:
    env = RecordingEnv([])

    written = set_cublas_workspace(env)

    assert written == CUBLAS_DETERMINISTIC_WORKSPACE
    assert env.values[CUBLAS_WORKSPACE_ENV_VAR] == CUBLAS_DETERMINISTIC_WORKSPACE


def test_encode_nests_the_settings_under_their_stack() -> None:
    # Nested rather than flattened so a setting can never collide with the
    # "stack" key, whatever a future stack decides to name one.
    applied = RecordingTorch([]).apply(RecordingEnv([]), remove_split_k=False)
    encoded = encode_determinism_record(applied)

    assert encoded == {
        "stack": TORCH_STACK,
        "settings": {
            "cublas_workspace_config": CUBLAS_DETERMINISTIC_WORKSPACE,
            "cudnn_benchmark": FALSE,
            "cudnn_deterministic": TRUE,
            "cudnn_tf32": FALSE,
            "deterministic_algorithms": TRUE,
            "matmul_tf32": FALSE,
        },
    }


def test_a_torch_record_round_trips() -> None:
    record = RecordingTorch([]).apply(RecordingEnv([]), remove_split_k=False)

    assert decode_determinism_record(encode_determinism_record(record)) == record


class TestTheThreadCountIsPartOfThePosture:
    """A pinned thread count that no record carries is a run nobody can repeat.

    `Model-Trainer.worker.job_utils.setup_env` pins `torch.set_num_threads`
    on every job, then used to return either an `apply_determinism` record
    that omitted the count or -- when determinism was declined --
    `determinism_record(UNPINNED_STACK, {})`, an EMPTY settings map for a run
    that had just pinned something. Measured on this stack: a 4096x4096
    matmul at one thread and at eight differs in 865,498 of 16,777,216
    elements, so those two runs need records that differ, and had records
    that did not.
    """

    def test_the_count_joins_a_pinned_record(self) -> None:
        pinned = RecordingTorch([]).apply(RecordingEnv([]), remove_split_k=False)

        recorded = with_torch_thread_count(pinned, 8)

        assert dict(recorded["settings"])[TORCH_THREAD_SETTING] == "8"
        assert recorded["stack"] == TORCH_STACK

    def test_the_count_joins_an_unpinned_record(self) -> None:
        """The declined branch pins threads too, so it records them too."""
        recorded = with_torch_thread_count(determinism_record(UNPINNED_STACK, {}), 8)

        assert recorded["stack"] == UNPINNED_STACK
        assert recorded["settings"] == ((TORCH_THREAD_SETTING, "8"),)

    def test_two_thread_counts_do_not_compare_equal(self) -> None:
        """The whole point: these two runs produce different numbers."""
        pinned = RecordingTorch([]).apply(RecordingEnv([]), remove_split_k=False)

        assert with_torch_thread_count(pinned, 1) != with_torch_thread_count(pinned, 8)

    def test_it_keeps_every_setting_the_stack_pinned(self) -> None:
        pinned = RecordingTorch([]).apply(RecordingEnv([]), remove_split_k=False)

        recorded = with_torch_thread_count(pinned, 4)

        assert dict(pinned["settings"]).items() <= dict(recorded["settings"]).items()

    def test_the_result_still_round_trips(self) -> None:
        applied = RecordingTorch([]).apply(RecordingEnv([]), remove_split_k=False)
        recorded = with_torch_thread_count(applied, 4)

        assert decode_determinism_record(encode_determinism_record(recorded)) == recorded

    def test_the_torch_spelling_is_not_the_blas_env_spelling(self) -> None:
        """`OMP_NUM_THREADS` is read when a BLAS LOADS; `set_num_threads` takes
        whenever it is called. A run pinned by one has not had the other done
        to it, so one name for both would compare them as equal."""
        assert TORCH_THREAD_SETTING not in BLAS_THREAD_ENV_VARS


# The stack-agnostic surface of the record -- construction, ordering,
# rendering and the decode rejections -- is tested in platform_core, beside
# the type itself. What stays here is the TORCH PRODUCER: that the pinning
# happens, in the right order, and that what it reports matches what it did.
