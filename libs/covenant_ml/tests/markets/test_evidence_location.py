"""Evidence that moved hosts is verified where it is now, by the hash it was recorded with."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, narrow_json_to_dict, require_list

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.campaign_execution import tick_campaign
from covenant_ml.markets.campaign_inputs import load_campaign_inputs
from covenant_ml.markets.campaign_process import run_command
from covenant_ml.markets.campaign_records import Campaign
from covenant_ml.markets.evidence_location import (
    RELOCATIONS_FILE,
    EvidenceRelocation,
    decode_relocations,
    locate_evidence,
    read_host_relocations,
    read_relocations,
)
from covenant_ml.markets.reporting import read_verified_market_report

from ._campaign_fixtures import single_campaign
from ._paper_fixtures import serve_paper
from ._server import ApiServer, api_server

__all__ = ["api_server"]

WINDOWS = EvidenceRelocation(
    recorded_prefix="C:/Users/Test/PROJECTS/files", current_prefix="/home/corvis/PROJECTS/files"
)


def _declaration(*entries: tuple[str, str]) -> JSONObject:
    return {
        "schema_version": 1,
        "relocations": [{"recorded_prefix": r, "current_prefix": c} for r, c in entries],
    }


def test_a_windows_path_under_a_declared_prefix_is_read_under_its_current_prefix() -> None:
    recorded = "C:\\Users\\Test\\PROJECTS\\files\\market-edge-20261012\\protocol.json"
    assert locate_evidence(recorded, (WINDOWS,)) == Path(
        "/home/corvis/PROJECTS/files/market-edge-20261012/protocol.json"
    )
    assert locate_evidence("C:\\Users\\Test\\PROJECTS\\files", (WINDOWS,)) == Path(
        "/home/corvis/PROJECTS/files"
    )


def test_a_path_outside_every_prefix_is_read_where_it_was_recorded() -> None:
    sibling = "C:\\Users\\Test\\PROJECTS\\files-old\\protocol.json"
    assert locate_evidence(sibling, (WINDOWS,)) == Path(sibling)
    assert locate_evidence("/srv/evidence/a.json", ()) == Path("/srv/evidence/a.json")


def test_the_first_declared_prefix_containing_a_path_decides() -> None:
    narrow = EvidenceRelocation(
        recorded_prefix="C:/Users/Test/PROJECTS/files/stock-panel-20261014",
        current_prefix="/mnt/archive/stock-panel",
    )
    recorded = "C:\\Users\\Test\\PROJECTS\\files\\stock-panel-20261014\\pilot.json"
    assert locate_evidence(recorded, (narrow, WINDOWS)) == Path(
        "/mnt/archive/stock-panel/pilot.json"
    )
    assert locate_evidence(recorded, (WINDOWS, narrow)) == Path(
        "/home/corvis/PROJECTS/files/stock-panel-20261014/pilot.json"
    )


def test_a_declaration_is_decoded_with_its_prefixes_spelled_alike(tmp_path: Path) -> None:
    current = str(tmp_path / "files")
    decoded = decode_relocations(_declaration(("C:\\Users\\Test\\PROJECTS\\files\\", current)))
    assert decoded == (
        EvidenceRelocation(recorded_prefix="C:/Users/Test/PROJECTS/files", current_prefix=current),
    )


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        ({"schema_version": 2, "relocations": []}, "schema version"),
        (_declaration(("", "/home/corvis/files")), "prefix is empty"),
        (_declaration(("C:/a", "")), "prefix is empty"),
        (_declaration(("C:/a", "relative/files")), "not absolute"),
        (_declaration(("C:/a", "/x"), ("C:\\a\\", "/y")), "declared twice"),
    ],
)
def test_an_invalid_declaration_is_refused(
    document: JSONObject, reason: str, tmp_path: Path
) -> None:
    entries = require_list(document, "relocations")
    absolute = str(tmp_path)
    for value in entries:
        entry = narrow_json_to_dict(value)
        if entry["current_prefix"] in ("/home/corvis/files", "/x", "/y"):
            entry["current_prefix"] = absolute
    with pytest.raises(ValueError, match=rf"EVIDENCE-RELOCATION-001.*{reason}"):
        decode_relocations(document)


def test_a_host_without_a_declaration_relocates_nothing(tmp_path: Path) -> None:
    assert read_relocations(tmp_path / "absent.json") == ()
    declared = tmp_path / "evidence-relocations.json"
    declared.write_text(dump_json_str(_declaration(("C:/old", str(tmp_path)))), "utf-8")
    assert read_relocations(declared) == (
        EvidenceRelocation(recorded_prefix="C:/old", current_prefix=str(tmp_path)),
    )
    assert read_host_relocations() == read_relocations(Path.home() / RELOCATIONS_FILE)


INPUTS = ("protocol.json", "model.json")


def _campaign_under(directory: Path) -> tuple[Campaign, Path, str]:
    """A one-plan campaign whose frozen inputs are written into ``directory``."""
    directory.mkdir()
    return single_campaign(directory)


def _move(source: Path, destination: Path) -> tuple[EvidenceRelocation, ...]:
    """Move a campaign's frozen inputs, and declare each file's new place.

    The campaign's own directory stays under ``source``, so each input is
    declared by its own path rather than by the directory holding both.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.move(str(source / name), str(destination / name))
    return tuple(
        EvidenceRelocation(
            recorded_prefix=str(source / name), current_prefix=str(destination / name)
        )
        for name in INPUTS
    )


def test_a_campaign_whose_inputs_moved_is_verified_and_run_where_they_are_now(
    api_server: ApiServer, tmp_path: Path
) -> None:
    campaign, root, digest = _campaign_under(tmp_path / "old")
    moved = _move(tmp_path / "old", tmp_path / "new")
    with pytest.raises(FileNotFoundError):
        load_campaign_inputs(root, digest)
    _test_hooks.evidence_relocations = lambda: moved
    assert load_campaign_inputs(root, digest) == campaign
    plan = campaign["plans"][0]
    ticker = serve_paper(api_server, plan)
    market = narrow_json_to_dict(require_list(api_server.documents["/markets"][0], "markets")[0])
    api_server.documents["/markets/" + ticker] = [{"market": market}]
    counts = tick_campaign(
        root, digest, _test_hooks.clock, run_command, api_server.url, api_server.url
    )
    assert counts["complete"] == 1
    assert counts["failed"] == 0


def test_a_report_whose_sources_moved_verifies_by_their_recorded_hashes(
    api_server: ApiServer, tmp_path: Path
) -> None:
    campaign, root, digest = _campaign_under(tmp_path / "old")
    plan = campaign["plans"][0]
    ticker = serve_paper(api_server, plan)
    market = narrow_json_to_dict(require_list(api_server.documents["/markets"][0], "markets")[0])
    api_server.documents["/markets/" + ticker] = [{"market": market}]
    tick_campaign(root, digest, _test_hooks.clock, run_command, api_server.url, api_server.url)
    report = root / "reports" / (plan["name"] + "-capture.json")
    _, recorded, _ = read_verified_market_report(report, "TEST-001")
    moved = _move(tmp_path / "old", tmp_path / "new")
    with pytest.raises(FileNotFoundError):
        read_verified_market_report(report, "TEST-001")
    _test_hooks.evidence_relocations = lambda: moved
    _, located, _ = read_verified_market_report(report, "TEST-001")
    assert tmp_path / "new" / "protocol.json" in located
    assert len(located) == len(recorded)
    (tmp_path / "new" / "protocol.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match=r"TEST-001.*source evidence changed"):
        read_verified_market_report(report, "TEST-001")
