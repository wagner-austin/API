"""Tests for capturing what a measured number was produced on.

The case that motivates the module is the last one here: a completed cloze
record with no fingerprint is refused rather than served. That refusal is
what makes the other measurements comparable, and it is deliberately not
softened for records written before the field existed -- the published
52.3030% floor is exactly such a record, and it genuinely cannot be compared
with anything, because nothing knows what it ran on.
"""

from __future__ import annotations

import re
from collections.abc import Generator

import pytest
from platform_ml import (
    FALSE,
    TORCH_STACK,
    TRUE,
    decode_run_fingerprint,
    determinism_record,
    encode_run_fingerprint,
)

from model_trainer.core import _test_hooks
from model_trainer.core._hook_defaults import _default_cuda_driver_version
from model_trainer.core.run_fingerprint import (
    CUDA_DEVICE,
    NO_GPU,
    capture_run_fingerprint,
    describe_run_fingerprint,
)

PINNED = determinism_record(
    TORCH_STACK,
    {
        "deterministic_algorithms": TRUE,
        "cublas_workspace_config": ":4096:8",
        "matmul_tf32": FALSE,
        "cudnn_tf32": FALSE,
        "cudnn_deterministic": TRUE,
        "cudnn_benchmark": FALSE,
    },
)


class _Recorder:
    """Records whether the CUDA accessors were reached at all."""

    def __init__(self, name: str, driver: str) -> None:
        self.name = name
        self.driver = driver
        self.calls: list[str] = []

    def device_name(self) -> str:
        self.calls.append("device_name")
        return self.name

    def driver_version(self) -> str:
        self.calls.append("driver_version")
        return self.driver


def _restore_hooks() -> Generator[None, None, None]:
    """Put the three hooks this module reads back after each test.

    They are module-global. Left swapped they would answer for every later
    test in the same worker, which is the failure mode a shared hook has and
    a fixture is the whole of the fix.

    Yields:
        None, for the duration of one test.
    """
    git = _test_hooks.env_git_commit
    name = _test_hooks.cuda_device_name
    driver = _test_hooks.cuda_driver_version
    yield
    _test_hooks.env_git_commit = git
    _test_hooks.cuda_device_name = name
    _test_hooks.cuda_driver_version = driver


restore_hooks = pytest.fixture(_restore_hooks)


@pytest.mark.usefixtures("restore_hooks")
def test_a_cuda_run_records_the_card_and_the_driver() -> None:
    rec = _Recorder("NVIDIA GeForce RTX 3090 Ti", "591.86")
    _test_hooks.env_git_commit = lambda: "sha256:abc"
    _test_hooks.cuda_device_name = rec.device_name
    _test_hooks.cuda_driver_version = rec.driver_version

    fingerprint = capture_run_fingerprint(CUDA_DEVICE, PINNED)

    assert fingerprint == {
        "image_digest": "sha256:abc",
        "gpu_model": "NVIDIA GeForce RTX 3090 Ti",
        "driver_version": "591.86",
        "determinism": PINNED,
    }


@pytest.mark.usefixtures("restore_hooks")
def test_a_cpu_run_never_touches_the_cuda_accessors() -> None:
    # Querying them would initialise a CUDA context to describe hardware the
    # run does not use, and would put a card in the record of a measurement
    # that never had one.
    rec = _Recorder("NVIDIA GeForce RTX 3090 Ti", "591.86")
    _test_hooks.env_git_commit = lambda: "sha256:abc"
    _test_hooks.cuda_device_name = rec.device_name
    _test_hooks.cuda_driver_version = rec.driver_version

    fingerprint = capture_run_fingerprint("cpu", PINNED)

    assert rec.calls == []
    assert fingerprint["gpu_model"] == NO_GPU
    assert fingerprint["driver_version"] == NO_GPU


@pytest.mark.usefixtures("restore_hooks")
def test_an_unstamped_build_records_an_unknown_digest() -> None:
    # Empty is not a wildcard: it differs from every known digest, so a run
    # from an unstamped image never compares equal to a stamped one.
    rec = _Recorder("NVIDIA GeForce RTX 3090 Ti", "591.86")
    _test_hooks.env_git_commit = lambda: None
    _test_hooks.cuda_device_name = rec.device_name
    _test_hooks.cuda_driver_version = rec.driver_version

    fingerprint = capture_run_fingerprint(CUDA_DEVICE, PINNED)

    assert fingerprint["image_digest"] == ""


@pytest.mark.usefixtures("restore_hooks")
def test_a_captured_fingerprint_round_trips_through_storage() -> None:
    rec = _Recorder("NVIDIA A100 80GB PCIe", "550.90.07")
    _test_hooks.env_git_commit = lambda: "sha256:def"
    _test_hooks.cuda_device_name = rec.device_name
    _test_hooks.cuda_driver_version = rec.driver_version

    fingerprint = capture_run_fingerprint(CUDA_DEVICE, PINNED)

    assert decode_run_fingerprint(encode_run_fingerprint(fingerprint)) == fingerprint


def test_the_driver_hook_reads_the_real_driver_not_the_cuda_toolkit() -> None:
    """The adapter reaches real nvidia-smi, like the torch adapters reach torch.

    It assumes CUDA is present because its only caller gates on the run's
    device being cuda; this suite runs on the GPU box.

    Asserted as a version number rather than as merely non-empty, because the
    failure this guards against is the field carrying the wrong thing:
    ``torch.version.cuda`` is the runtime toolkit the wheel was built against
    (12.4 here), not the driver (591.86), and it would sail through any
    non-empty check while making two different drivers compare equal.
    """
    version = _default_cuda_driver_version()

    match = re.fullmatch(r"\d+(\.\d+)+", version)
    if match is None:
        raise AssertionError(f"driver version is not a version number: {version!r}")
    assert match.group(0) == version


def test_describe_names_the_three_fields_a_reader_compares() -> None:
    line = describe_run_fingerprint(
        {
            "image_digest": "sha256:abc",
            "gpu_model": "NVIDIA GeForce RTX 3090 Ti",
            "driver_version": "591.86",
            "determinism": PINNED,
        }
    )

    assert line == "image=sha256:abc gpu=NVIDIA GeForce RTX 3090 Ti driver=591.86"


def test_describe_spells_absence_as_a_word_not_a_blank() -> None:
    # A blank in a log reads as a formatting fault rather than as the absence
    # it records.
    line = describe_run_fingerprint(
        {
            "image_digest": "",
            "gpu_model": "",
            "driver_version": "",
            "determinism": PINNED,
        }
    )

    assert line == "image=unknown gpu=none driver=none"
