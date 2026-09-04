"""Tests for capturing what a measured number was produced on.

The case that motivates the module is the last one here: a completed cloze
record with no fingerprint is refused rather than served. That refusal is
what makes the other measurements comparable, and it is deliberately not
softened for records written before the field existed -- the published
52.3030% floor is exactly such a record, and it genuinely cannot be compared
with anything, because nothing knows what it ran on.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
from collections.abc import Generator

import pytest
from platform_core.comparability import decode_run_fingerprint, encode_run_fingerprint
from platform_core.determinism_record import FALSE, TRUE, determinism_record
from platform_core.environment_record import PackageVersion, capture_host_record
from platform_core.testing import (
    SAMPLE_HOST,
    FakeHostProbe,
    FakeVersionReader,
    sample_run_fingerprint,
)
from platform_ml import TORCH_STACK

from model_trainer.core import _test_hooks
from model_trainer.core._hook_defaults import (
    _default_host_probe,
    _default_installed_version,
)
from model_trainer.core._hook_defaults_cuda import _default_cuda_driver_version
from model_trainer.core.run_fingerprint import (
    CUDA_DEVICE,
    FINGERPRINT_DISTRIBUTIONS,
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


#: The versions the fake reader reports, in the canonical order the
#: fingerprint stores them.
_SAMPLE_VERSIONS: tuple[PackageVersion, ...] = (
    PackageVersion(name="numpy", version="2.3.5"),
    PackageVersion(name="torch", version="2.6.0"),
    PackageVersion(name="transformers", version="4.46.3"),
)


def _sample_host_probe() -> FakeHostProbe:
    """Build the probe that reports the stated machine.

    Returns:
        A probe reporting :data:`SAMPLE_HOST`.
    """
    return FakeHostProbe(
        platform=SAMPLE_HOST["platform"],
        machine=SAMPLE_HOST["machine"],
        logical_cores=SAMPLE_HOST["logical_cores"],
    )


def _restore_hooks() -> Generator[None, None, None]:
    """State the machine and libraries, then put every hook back.

    The hooks are module-global. Left swapped they would answer for every
    later test in the same worker, which is the failure mode a shared hook
    has and a fixture is the whole of the fix.

    The host and version hooks are INSTALLED here rather than merely saved,
    because a fingerprint test that read the real machine would assert a
    different platform string on every developer's box and a different one
    again in CI.

    Yields:
        None, for the duration of one test.
    """
    digest = _test_hooks.env_image_digest
    name = _test_hooks.cuda_device_name
    driver = _test_hooks.cuda_driver_version
    probe = _test_hooks.host_probe
    version = _test_hooks.installed_version
    _test_hooks.host_probe = _sample_host_probe
    _test_hooks.installed_version = FakeVersionReader(
        {"numpy": "2.3.5", "torch": "2.6.0", "transformers": "4.46.3"}
    )
    yield
    _test_hooks.env_image_digest = digest
    _test_hooks.cuda_device_name = name
    _test_hooks.cuda_driver_version = driver
    _test_hooks.host_probe = probe
    _test_hooks.installed_version = version


restore_hooks = pytest.fixture(_restore_hooks)


@pytest.mark.usefixtures("restore_hooks")
def test_a_cuda_run_records_the_card_and_the_driver() -> None:
    rec = _Recorder("NVIDIA GeForce RTX 3090 Ti", "591.86")
    _test_hooks.env_image_digest = lambda: "sha256:abc"
    _test_hooks.cuda_device_name = rec.device_name
    _test_hooks.cuda_driver_version = rec.driver_version

    fingerprint = capture_run_fingerprint(CUDA_DEVICE, PINNED)

    assert fingerprint == {
        "image_digest": "sha256:abc",
        "gpu_model": "NVIDIA GeForce RTX 3090 Ti",
        "driver_version": "591.86",
        "determinism": PINNED,
        "host": SAMPLE_HOST,
        "packages": _SAMPLE_VERSIONS,
    }


@pytest.mark.usefixtures("restore_hooks")
def test_a_cpu_run_never_touches_the_cuda_accessors() -> None:
    # Querying them would initialise a CUDA context to describe hardware the
    # run does not use, and would put a card in the record of a measurement
    # that never had one.
    rec = _Recorder("NVIDIA GeForce RTX 3090 Ti", "591.86")
    _test_hooks.env_image_digest = lambda: "sha256:abc"
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
    _test_hooks.env_image_digest = lambda: None
    _test_hooks.cuda_device_name = rec.device_name
    _test_hooks.cuda_driver_version = rec.driver_version

    fingerprint = capture_run_fingerprint(CUDA_DEVICE, PINNED)

    assert fingerprint["image_digest"] == ""


@pytest.mark.usefixtures("restore_hooks")
def test_a_captured_fingerprint_round_trips_through_storage() -> None:
    rec = _Recorder("NVIDIA A100 80GB PCIe", "550.90.07")
    _test_hooks.env_image_digest = lambda: "sha256:def"
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
        sample_run_fingerprint(
            image_digest="sha256:abc",
            gpu_model="NVIDIA GeForce RTX 3090 Ti",
            driver_version="591.86",
            determinism=PINNED,
        )
    )

    assert line == "image=sha256:abc gpu=NVIDIA GeForce RTX 3090 Ti driver=591.86"


def test_describe_spells_absence_as_a_word_not_a_blank() -> None:
    # A blank in a log reads as a formatting fault rather than as the absence
    # it records.
    line = describe_run_fingerprint(
        sample_run_fingerprint(
            image_digest="",
            gpu_model="",
            driver_version="",
            determinism=PINNED,
        )
    )

    assert line == "image=unknown gpu=none driver=none"


@pytest.mark.usefixtures("restore_hooks")
def test_the_fingerprint_carries_the_machine_it_ran_on() -> None:
    # The axis a cpu-only research stack has instead of a card, and the one
    # a torch run needs too: two nodes of one cluster differ here and in
    # nothing else a fingerprint used to record.
    _test_hooks.env_image_digest = lambda: "sha256:abc"

    fingerprint = capture_run_fingerprint("cpu", PINNED)

    assert fingerprint["host"] == SAMPLE_HOST


@pytest.mark.usefixtures("restore_hooks")
def test_the_fingerprint_carries_the_libraries_that_decide_its_numbers() -> None:
    _test_hooks.env_image_digest = lambda: "sha256:abc"

    fingerprint = capture_run_fingerprint("cpu", PINNED)

    assert fingerprint["packages"] == _SAMPLE_VERSIONS


def test_the_recorded_libraries_are_the_ones_that_can_change_a_number() -> None:
    # Not every installed distribution: a fingerprint over all of them
    # differs on a dev-dependency bump that cannot reach a matmul, and every
    # spurious difference makes a real one harder to see.
    assert FINGERPRINT_DISTRIBUTIONS == ("numpy", "torch", "transformers")


class TestTheProductionHooks:
    def test_the_host_probe_reads_this_real_machine(self) -> None:
        # The default hook, exercised against the real stdlib. A test that
        # replaced the reads would assert nothing about the production path.
        record = capture_host_record(_default_host_probe())

        assert record["platform"] == platform.platform()
        assert record["machine"] == platform.machine()
        assert record["logical_cores"] == os.cpu_count()

    def test_the_version_reader_reads_really_installed_metadata(self) -> None:
        assert _default_installed_version("torch") == importlib.metadata.version("torch")

    def test_the_version_reader_propagates_a_missing_distribution(self) -> None:
        # Rather than returning "unknown" as `pkg_version` does. "unknown" is
        # a non-empty string, so it would pass every validator and then
        # compare EQUAL between two environments that each failed to find the
        # library for different reasons.
        with pytest.raises(importlib.metadata.PackageNotFoundError):
            _default_installed_version("a-distribution-that-is-not-installed")

    def test_the_two_version_readers_disagree_deliberately(self) -> None:
        # The soft one is for a human-readable manifest and the strict one is
        # for a comparability axis. Asserting the difference so that a later
        # "cleanup" that unified them fails here rather than silently
        # softening a fingerprint.
        assert _test_hooks._default_pkg_version("a-distribution-that-is-not-installed") == (
            "unknown"
        )
