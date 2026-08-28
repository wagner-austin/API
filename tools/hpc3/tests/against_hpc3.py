"""Production decoders with HPC3's measured facts already bound.

Every rule this package enforces is a rule about a specific machine, so
every decoder takes the cluster it is checking against. Most tests are about
the rule rather than about the machine, and threading the same constant
through several hundred call sites would bury the assertion under the
plumbing.

So the cluster is bound once, here. A test that imports from this module is
saying "against HPC3, the real one". The separate claim -- that the rules
follow a *different* cluster's numbers rather than HPC3's -- is proved in
``test_cluster.py`` against a synthetic cluster whose every value differs,
and it is proved by calling the production functions directly.

This is binding, not faking. The functions below are the production ones with
one argument supplied.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

from platform_core.json_utils import JSONValue

from hpc3.clusters.hpc3 import HPC3
from hpc3.contracts.budget import Budget, Consumption
from hpc3.contracts.job import JobSpec
from hpc3.contracts.job import decode_job_spec as _decode_job_spec
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.ledger import decode_ledger_entry as _decode_ledger_entry
from hpc3.contracts.preflight import PreflightResult
from hpc3.contracts.preflight import decode_preflight_result as _decode_preflight_result
from hpc3.contracts.status import JobStatus
from hpc3.contracts.status import decode_job_status as _decode_job_status
from hpc3.contracts.status import service_units as _service_units
from hpc3.contracts.sweep import SweepSpec
from hpc3.contracts.sweep import decode_sweep_spec as _decode_sweep_spec
from hpc3.contracts.workspace import ProjectConfig
from hpc3.contracts.workspace import decode_project_config as _decode_project_config
from hpc3.core import ledger as _ledger
from hpc3.core.budget import check_consumption as _check_consumption
from hpc3.core.budget import check_projection as _check_projection
from hpc3.core.budget import observe as _observe
from hpc3.core.budget import project as _project
from hpc3.core.cancel import CancelOutcome
from hpc3.core.cancel import cancel as _cancel
from hpc3.core.preflight import parse_test_only as _parse_test_only
from hpc3.core.status import parse_sacct_output as _parse_sacct_output
from hpc3.core.status import parse_sacct_row as _parse_sacct_row

PROJECT_CONFIG_DIR = pathlib.Path("/w")
"""Stand-in directory a test project's relative ``repo`` resolves against."""


def decode_job_spec(value: JSONValue) -> JobSpec:
    """Decode a job spec against HPC3, with no service-unit budget declared.

    Binds ``max_service_units=0.0``, which is the free-work-only posture and
    what nearly every test is about. A test that needs the billed path calls
    the production function directly and says so, the same way
    ``test_cluster.py`` calls it directly to prove the rules follow a
    cluster's numbers rather than HPC3's.

    Args:
        value: The value to decode.

    Returns:
        The validated spec.
    """
    return _decode_job_spec(value, HPC3, max_service_units=0.0)


def decode_job_status(value: JSONValue) -> JobStatus:
    """Decode an accounting row against HPC3.

    Args:
        value: The value to decode.

    Returns:
        The validated status.
    """
    return _decode_job_status(value, HPC3)


def decode_ledger_entry(value: JSONValue) -> LedgerEntry:
    """Decode a ledger record against HPC3.

    Args:
        value: The value to decode.

    Returns:
        The validated entry.
    """
    return _decode_ledger_entry(value, HPC3)


def decode_preflight_result(value: JSONValue) -> PreflightResult:
    """Decode a scheduler verdict against HPC3.

    Args:
        value: The value to decode.

    Returns:
        The validated result.
    """
    return _decode_preflight_result(value, HPC3)


def decode_sweep_spec(value: JSONValue) -> SweepSpec:
    """Decode a sweep against HPC3's QOS ceilings.

    Args:
        value: The value to decode.

    Returns:
        The validated sweep.
    """
    return _decode_sweep_spec(value, HPC3, max_service_units=0.0)


def decode_project_config(
    value: JSONValue, *, config_dir: pathlib.Path = PROJECT_CONFIG_DIR
) -> ProjectConfig:
    """Decode one project's defaults against HPC3.

    Args:
        value: The value to decode.
        config_dir: Directory a relative ``repo`` resolves against. Bound to
            a fixed stand-in by default, because nearly every test is about
            a resource rule rather than about where the code lives; the
            tests that ARE about the path pass their own.

    Returns:
        The validated defaults.
    """
    return _decode_project_config(value, HPC3, config_dir=config_dir)


def parse_sacct_row(line: str) -> JobStatus:
    """Parse one accounting row against HPC3.

    Args:
        line: The pipe-delimited row.

    Returns:
        The validated status.
    """
    return _parse_sacct_row(line, HPC3)


def parse_sacct_output(output: str) -> list[JobStatus]:
    """Parse an accounting query's output against HPC3.

    Args:
        output: The command's standard output.

    Returns:
        One status per row.
    """
    return _parse_sacct_output(output, HPC3)


def parse_test_only(output: str) -> PreflightResult:
    """Parse a ``sbatch --test-only`` verdict against HPC3.

    Args:
        output: The command's output.

    Returns:
        The validated result.
    """
    return _parse_test_only(output, HPC3)


def service_units(status: JobStatus) -> float:
    """Compute a job's real charge using HPC3's usage factors.

    Args:
        status: The job's accounting row.

    Returns:
        Service units consumed.
    """
    return _service_units(status, HPC3)


def project(specs: Sequence[JobSpec]) -> Consumption:
    """Project what specs would consume on HPC3.

    Args:
        specs: Specs to total.

    Returns:
        Projected consumption.
    """
    return _project(specs, HPC3)


def observe(statuses: Sequence[JobStatus]) -> Consumption:
    """Total what jobs have consumed on HPC3.

    Args:
        statuses: Accounting rows to total.

    Returns:
        Observed consumption.
    """
    return _observe(statuses, HPC3)


def check_projection(budget: Budget, specs: Sequence[JobSpec]) -> Consumption:
    """Enforce a budget against a projection on HPC3.

    Args:
        budget: The caps to enforce.
        specs: Specs about to be submitted.

    Returns:
        The projection.
    """
    return _check_projection(budget, specs, HPC3)


def check_consumption(budget: Budget, statuses: Sequence[JobStatus]) -> Consumption:
    """Enforce a budget against observed usage on HPC3.

    Args:
        budget: The caps to enforce.
        statuses: Accounting rows to total.

    Returns:
        The observed consumption.
    """
    return _check_consumption(budget, statuses, HPC3)


def cancel(host: str, job_ids: Sequence[str]) -> list[CancelOutcome]:
    """Cancel jobs on HPC3 and report what actually stopped.

    Args:
        host: SSH destination.
        job_ids: Ids to cancel.

    Returns:
        One outcome per job accounting knows.
    """
    return _cancel(host, job_ids, HPC3)


def read_ledger(path: pathlib.Path) -> list[LedgerEntry]:
    """Read a ledger written against HPC3.

    Args:
        path: Ledger file.

    Returns:
        Every recorded entry, oldest first.
    """
    return _ledger.read(path, HPC3)


__all__ = [
    "cancel",
    "check_consumption",
    "check_projection",
    "decode_job_spec",
    "decode_job_status",
    "decode_ledger_entry",
    "decode_preflight_result",
    "decode_project_config",
    "decode_sweep_spec",
    "observe",
    "parse_sacct_output",
    "parse_sacct_row",
    "parse_test_only",
    "project",
    "read_ledger",
    "service_units",
]
