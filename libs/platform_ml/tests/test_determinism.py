"""Tests for kernel-level determinism controls.

The doubles here are real classes satisfying the module's Protocols, not
patches. Each records the ORDER of what it was asked to do, because ordering
carries the correctness here: ``CUBLAS_WORKSPACE_CONFIG`` is read once when
the cuBLAS handle is created, so writing it after a CUDA call is accepted
silently and does nothing. A test that only checked the final values would
pass on a version that set it last.
"""

from __future__ import annotations

from platform_ml.determinism import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    CUBLAS_WORKSPACE_ENV_VAR,
    DeterminismReport,
    apply_determinism,
    encode_determinism_report,
    set_cublas_workspace,
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

    def apply(self, env: RecordingEnv) -> DeterminismReport:
        """Call the function under test with this bundle's leaves."""
        return apply_determinism(self.cudnn, self.matmul, self.use_deterministic_algorithms, env)


def test_apply_determinism_returns_every_field_it_set() -> None:
    log: list[str] = []
    torch = RecordingTorch(log)
    env = RecordingEnv(log)

    report = torch.apply(env)

    assert report == DeterminismReport(
        deterministic_algorithms=True,
        cublas_workspace_config=CUBLAS_DETERMINISTIC_WORKSPACE,
        matmul_tf32=False,
        cudnn_tf32=False,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
    )


def test_apply_determinism_writes_the_env_var_before_touching_cuda() -> None:
    # The whole point: CUBLAS_WORKSPACE_CONFIG is read when the cuBLAS handle
    # is created. If any CUDA-touching call precedes it, the setting is a
    # no-op and the run is nondeterministic while reporting that it is not.
    log: list[str] = []
    torch = RecordingTorch(log)
    env = RecordingEnv(log)

    torch.apply(env)

    assert log[0] == f"env:{CUBLAS_WORKSPACE_ENV_VAR}={CUBLAS_DETERMINISTIC_WORKSPACE}"


def test_apply_determinism_pins_every_flag_the_report_claims() -> None:
    # Asserts the ACTUAL post-state of the doubles, not only the returned
    # report, so a report that lies about what it did fails here.
    log: list[str] = []
    torch = RecordingTorch(log)
    env = RecordingEnv(log)

    torch.apply(env)

    assert torch.matmul.allow_tf32 is False
    assert torch.cudnn.allow_tf32 is False
    assert torch.cudnn.deterministic is True
    assert torch.cudnn.benchmark is False
    assert torch.deterministic_calls == [True]
    assert env.values[CUBLAS_WORKSPACE_ENV_VAR] == CUBLAS_DETERMINISTIC_WORKSPACE


def test_apply_determinism_touches_exactly_these_settings() -> None:
    # A shape-drift guard. If someone adds a flag to apply_determinism without
    # adding it to DeterminismReport, the run record silently stops describing
    # the run, which is the failure this module exists to prevent.
    log: list[str] = []
    RecordingTorch(log).apply(RecordingEnv(log))

    assert log == [
        f"env:{CUBLAS_WORKSPACE_ENV_VAR}={CUBLAS_DETERMINISTIC_WORKSPACE}",
        "matmul.allow_tf32=False",
        "cudnn.allow_tf32=False",
        "cudnn.deterministic=True",
        "cudnn.benchmark=False",
        "use_deterministic_algorithms=True",
    ]


def test_set_cublas_workspace_returns_what_it_wrote() -> None:
    env = RecordingEnv([])

    written = set_cublas_workspace(env)

    assert written == CUBLAS_DETERMINISTIC_WORKSPACE
    assert env.values[CUBLAS_WORKSPACE_ENV_VAR] == CUBLAS_DETERMINISTIC_WORKSPACE


def test_encode_renders_both_boolean_spellings() -> None:
    encoded = encode_determinism_report(
        DeterminismReport(
            deterministic_algorithms=True,
            cublas_workspace_config=CUBLAS_DETERMINISTIC_WORKSPACE,
            matmul_tf32=False,
            cudnn_tf32=False,
            cudnn_deterministic=True,
            cudnn_benchmark=False,
        )
    )

    assert encoded == {
        "deterministic_algorithms": "true",
        "cublas_workspace_config": CUBLAS_DETERMINISTIC_WORKSPACE,
        "matmul_tf32": "false",
        "cudnn_tf32": "false",
        "cudnn_deterministic": "true",
        "cudnn_benchmark": "false",
    }


def test_encode_covers_every_report_field() -> None:
    # Keys, sorted, so adding a field to DeterminismReport without adding it
    # to the encoder fails here rather than silently dropping it from the
    # provenance record.
    encoded = encode_determinism_report(RecordingTorch([]).apply(RecordingEnv([])))

    assert sorted(encoded.keys()) == [
        "cublas_workspace_config",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_tf32",
        "deterministic_algorithms",
        "matmul_tf32",
    ]
