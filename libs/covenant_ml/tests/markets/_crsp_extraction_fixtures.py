"""Real protocol bytes, real evaluator keys and the database seam for writer tests."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from platform_core.json_utils import JSONObject, dump_json_str

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.crsp_extraction_records import (
    CrspExtractionReceipt,
    CrspLimits,
    decode_crsp_limits,
)
from covenant_ml.markets.crsp_writer import extract_crsp
from covenant_ml.markets.research_keys import ResearchAccess

from ._crsp_database import FakeDatabase, database_rows
from ._crsp_study_fixtures import study_protocol
from ._encrypted_fixtures import create_access

#: The writer's clock in every test: the day after the fixture's coverage.
NOW = "2026-10-31T09:00:00+00:00"


def _now() -> str:
    """The fixed writer clock."""
    return NOW


class Extraction(NamedTuple):
    """Inputs and directories of one writer invocation."""

    protocol: Path
    access: ResearchAccess
    root: Path
    scratch: Path
    output: Path


def extraction(root: Path) -> Extraction:
    """Write the study protocol, create real keys and empty private directories."""
    protocol = root / "crsp-protocol.json"
    protocol.write_text(dump_json_str(study_protocol()), encoding="utf-8")
    access = create_access(root)
    scratch = root / "scratch"
    scratch.mkdir(mode=0o700)
    output = root / "outputs"
    output.mkdir()
    return Extraction(protocol, access, root, scratch, output)


def limits(**changes: int) -> CrspLimits:
    """Bounds the synthetic database satisfies, with explicit changes."""
    document: JSONObject = {
        "max_rows": 20000,
        "max_relation_bytes": 2_000_000,
        "max_partition_bytes": 4_000_000,
        "seconds": 120,
    }
    document.update(changes)
    return decode_crsp_limits(document)


@contextmanager
def installed(database: FakeDatabase) -> Generator[FakeDatabase, None, None]:
    """Install the database and a fixed clock, restoring the real hooks afterwards."""
    runner, clock = _test_hooks.bounded_runner, _test_hooks.clock
    _test_hooks.bounded_runner = database
    _test_hooks.clock = _now
    yield database
    _test_hooks.bounded_runner = runner
    _test_hooks.clock = clock


def run_vendor(case: Extraction, bounds: CrspLimits) -> CrspExtractionReceipt:
    """Extract from the vendor schema with the given bounds."""
    return extract_crsp(
        case.protocol,
        case.access,
        schema="crsp_a_stock",
        setup=None,
        scratch=case.scratch,
        output=case.output,
        limits=bounds,
    )


__all__ = [
    "NOW",
    "Extraction",
    "FakeDatabase",
    "database_rows",
    "extraction",
    "installed",
    "limits",
    "run_vendor",
]
