"""Image inspection refuses incomplete native selfchecks and changed build evidence."""

import sys
from collections.abc import Sequence

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue, require_dict

from hpc3.contracts.ledger import encode_ledger_entry
from hpc3.core import _test_hooks
from hpc3.core.image_evidence import measure_build
from tests.conftest import FakeRun, cluster
from tests.execution_fixtures import INSTANT
from tests.image_evidence_fixtures import IMAGE_PATH, LOG, build_entry, pins, serve_build


def test_verified_build_reads_all_pins_accounting_and_native_checks(fake_run: FakeRun) -> None:
    serve_build(fake_run)
    report = measure_build(build_entry(), cluster(), pins())
    assert report["kind"] == "HPC3_IMAGE_BUILD_EVIDENCE"
    assert require_dict(report, "ledger") == encode_ledger_entry(build_entry())
    assert report["verified_files"] == pins()
    assert (report["image_mtime"], report["log_mtime"]) == (INSTANT, INSTANT)
    assert (report["selfcheck_passes"], report["dependency_check"]) == (2, "passed")
    assert require_dict(require_dict(report, "accounting"), "job")["state"] == "COMPLETED"
    assert require_dict(report, "accounting")["exit_code"] == "0:0"


@pytest.mark.parametrize(
    "changes", [{"experiment": {}}, {"artifact": None}, {"artifact": "/other/runtime.sif"}]
)
def test_non_build_or_unrelated_artifact_refused(changes: dict[str, JSONValue]) -> None:
    with pytest.raises(ValueError, match="require a recorded image build"):
        measure_build(build_entry(**changes), cluster(), pins())


@pytest.mark.parametrize("path", list(pins()))
def test_every_pin_is_required(path: str) -> None:
    checksums = pins()
    del checksums[path]
    with pytest.raises(ValueError, match="pin image, log"):
        measure_build(build_entry(), cluster(), checksums)


@pytest.mark.parametrize("removed", LOG.strip().splitlines())
def test_missing_selfcheck_dependency_or_checksum_receipt_refused(
    fake_run: FakeRun, removed: str
) -> None:
    log = LOG.replace(removed + "\n", "", 1)
    serve_build(fake_run, log)
    with pytest.raises(ValueError, match="require both selfchecks"):
        measure_build(build_entry(), cluster(), pins(log))


def test_changed_image_refused_even_with_successful_build_log(fake_run: FakeRun) -> None:
    serve_build(fake_run)
    checksums = pins()
    checksums[IMAGE_PATH] = "c" * 64
    with pytest.raises(AppError, match="digests to"):
        measure_build(build_entry(), cluster(), checksums)


def test_remote_native_failure_propagates_real_process_exit() -> None:
    """Run the real transport process boundary through a failed native read."""
    real_run = _test_hooks.run

    def failed_transport(
        argv: Sequence[str], *, stdin_bytes: bytes | None = None
    ) -> _test_hooks.CommandResult:
        assert argv[0] == "ssh"
        return real_run(
            [sys.executable, "-c", "import sys; sys.stderr.write('build-read-failed'); sys.exit(7)"],
            stdin_bytes=stdin_bytes,
        )

    _test_hooks.run = failed_transport
    with pytest.raises(AppError, match="build-read-failed"):
        measure_build(build_entry(), cluster(), pins())
