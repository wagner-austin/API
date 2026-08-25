"""Tests for kernel-level determinism controls.

The doubles here are real classes satisfying the module's Protocols, not
patches. Each records the ORDER of what it was asked to do, because ordering
carries the correctness here: ``CUBLAS_WORKSPACE_CONFIG`` is read once when
the cuBLAS handle is created, so writing it after a CUDA call is accepted
silently and does nothing. A test that only checked the final values would
pass on a version that set it last.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from platform_ml.determinism import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    CUBLAS_WORKSPACE_ENV_VAR,
    FALSE,
    TORCH_STACK,
    TRUE,
    UNPINNED_STACK,
    DeterminismRecord,
    apply_determinism,
    decode_determinism_record,
    determinism_record,
    encode_determinism_record,
    render_determinism_record,
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

    def apply(self, env: RecordingEnv) -> DeterminismRecord:
        """Call the function under test with this bundle's leaves."""
        return apply_determinism(self.cudnn, self.matmul, self.use_deterministic_algorithms, env)


def test_apply_determinism_returns_every_field_it_set() -> None:
    log: list[str] = []
    torch = RecordingTorch(log)
    env = RecordingEnv(log)

    record = torch.apply(env)

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
    # adding it to the returned record, the run record stops describing
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


def test_encode_nests_the_settings_under_their_stack() -> None:
    # Nested rather than flattened so a setting can never collide with the
    # "stack" key, whatever a future stack decides to name one.
    encoded = encode_determinism_record(RecordingTorch([]).apply(RecordingEnv([])))

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
    record = RecordingTorch([]).apply(RecordingEnv([]))

    assert decode_determinism_record(encode_determinism_record(record)) == record


# ---------------------------------------------------------------------------
# The stack-agnostic surface. These exist because most of this monorepo's
# research is NOT torch, and a determinism axis only one stack can fill makes
# every other stack's runs compare as though the question did not apply.
# ---------------------------------------------------------------------------


def test_a_non_torch_stack_can_describe_its_own_posture() -> None:
    # A gradient-boosting or BLAS-bound run pins different things entirely.
    record = determinism_record("numpy", {"threads": "1", "seed": "0"})

    assert record == {"stack": "numpy", "settings": (("seed", "0"), ("threads", "1"))}


def test_a_run_that_pinned_nothing_is_recordable() -> None:
    # "Nothing was pinned" is a fact about a run, and it must differ from a
    # pinned run rather than be absent.
    record = determinism_record(UNPINNED_STACK, {})

    assert record == DeterminismRecord(stack=UNPINNED_STACK, settings=())
    assert record != RecordingTorch([]).apply(RecordingEnv([]))


def test_settings_are_canonically_ordered_whatever_the_producer_did() -> None:
    # Two records of the same posture must be equal and render identically,
    # or a re-ordered producer would read as a configuration change.
    forwards = determinism_record("numpy", {"a": "1", "b": "2"})
    backwards = determinism_record("numpy", {"b": "2", "a": "1"})

    assert forwards == backwards
    assert render_determinism_record(forwards) == render_determinism_record(backwards)


def test_render_names_the_stack_and_its_settings() -> None:
    rendered = render_determinism_record(determinism_record("numpy", {"threads": "1"}))

    assert rendered == "numpy[threads=1]"


def test_two_stacks_pinning_the_same_setting_name_do_not_compare_equal() -> None:
    # The stack is part of the record precisely because a name like "threads"
    # can mean different things to different stacks.
    assert determinism_record("numpy", {"threads": "1"}) != determinism_record(
        "openblas", {"threads": "1"}
    )


def test_a_record_that_cannot_say_what_pinned_it_is_rejected() -> None:
    with pytest.raises(ValueError, match="UNPINNED_STACK"):
        determinism_record("", {"threads": "1"})


def test_decode_rejects_an_unnamed_stack() -> None:
    with pytest.raises(JSONTypeError, match="stack"):
        decode_determinism_record({"stack": "", "settings": {}})


def test_decode_rejects_a_setting_whose_value_is_not_a_string() -> None:
    # A bool here would decode into a record that renders "True" on one
    # producer and "true" on another, so two identical postures would compare
    # as different configurations.
    with pytest.raises(JSONTypeError, match="cudnn_benchmark"):
        decode_determinism_record({"stack": TORCH_STACK, "settings": {"cudnn_benchmark": False}})


def test_decode_rejects_a_non_object_and_a_missing_settings_block() -> None:
    with pytest.raises(JSONTypeError):
        decode_determinism_record("torch")
    with pytest.raises(JSONTypeError):
        decode_determinism_record({"stack": TORCH_STACK})
