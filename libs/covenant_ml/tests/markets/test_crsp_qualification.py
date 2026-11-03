"""Every panel path's first refusal, counted by the production resolver on real files."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    narrow_json_to_dict,
    require_list,
    require_str,
)

from covenant_ml.markets.crsp_extract import CrspSources, load_crsp_sources
from covenant_ml.markets.crsp_holdings import (
    CrspPathRefusal,
    crsp_resolve_window_path,
    index_crsp_outcomes,
)
from covenant_ml.markets.crsp_qualification import (
    CrspQualification,
    CrspRefusedPath,
    crsp_qualification_counts,
    decode_crsp_qualification,
    encode_crsp_qualification,
    qualify_crsp_panel,
)
from covenant_ml.markets.crsp_study import crsp_known_panel

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import DELISTED, missing_price_rows, study_protocol, study_rows
from .test_crsp_execution_boundaries import event
from .test_crsp_holdings import W1, daily, outcomes


def _sources(directory: Path, rows: dict[str, list[JSONObject]]) -> CrspSources:
    """Write rows under the committed design and load them through the verified reader."""
    path, _ = write_sources(directory, rows, study_protocol())
    return load_crsp_sources(path)


def open_suspension_rows() -> dict[str, list[JSONObject]]:
    """Security 7 stays suspended from 2019-03-11 through its 2019-04-01 exit."""
    rows = missing_price_rows(price_flag="SU", gap=9)
    for row in rows["returns"]:
        if row["permno"] == 7 and "2019-03-11" <= require_str(row, "dlycaldt") <= "2019-04-01":
            row.update(
                dlyret=None,
                dlyprevdt=None,
                dlyretmissflg="MP",
                dlyretdurflg="MR",
                dlyprc=None,
                dlyprcflg="SU",
            )
    return rows


def _position(sources: CrspSources, decision: str, permno: int) -> tuple[int, int]:
    """Independently locate a path: its decision index and its one-based walk ordinal."""
    ordinal = 0
    for index, item in enumerate(crsp_known_panel(sources)):
        for feature in item.features:
            ordinal += 1
            if item.window.decision == decision and feature.permno == permno:
                return index, ordinal
    raise AssertionError("path not in the panel")


def test_a_complete_extract_has_no_refusal_and_counts_every_path(tmp_path: Path) -> None:
    sources = _sources(tmp_path, study_rows())
    qualification = qualify_crsp_panel(sources)
    panel = crsp_known_panel(sources)
    assert qualification == CrspQualification(
        members=tuple(len(item.features) for item in panel),
        unfilled=(0,) * len(panel),
        training_decisions=sum(1 for item in panel if item.window.decision < "2018-07-01"),
        refused=(),
    )
    counts = crsp_qualification_counts(qualification)
    assert counts == {
        "decisions": len(panel),
        "paths": sum(len(item.features) for item in panel),
        "refused": 0,
        "reasons": {},
        "periods": {"training": 0, "validation": 0},
        "unfilled_training": 0,
        "unfilled_validation": 0,
        "dependent_decisions": 0,
        "dependent_rows": 0,
    }


def test_an_open_suspension_is_one_refusal_that_leaves_its_decision_unlabelled(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path, open_suspension_rows())
    qualification = qualify_crsp_panel(sources)
    index, ordinal = _position(sources, "2019-02-28", 7)
    returns = index_crsp_outcomes(sources).returns
    assert qualification["refused"] == (
        CrspRefusedPath(
            ordinal=ordinal,
            decision=index,
            period="validation",
            reason="price gap tail",
            entry_sha256=sha256(dump_json_str(returns[(7, "2019-03-01")]).encode()).hexdigest(),
            exit_sha256=sha256(dump_json_str(returns[(7, "2019-04-01")]).encode()).hexdigest(),
        ),
    )
    # Security 8 delists on that decision date, so its order at the next
    # session's delisting-flagged record is unfilled cash in the same decision.
    assert qualification["unfilled"] == tuple(
        int(item == index) for item in range(len(qualification["members"]))
    )
    counts = crsp_qualification_counts(qualification)
    assert counts["reasons"] == {"price gap tail": 1}
    assert counts["periods"] == {"training": 0, "validation": 1}
    assert (counts["unfilled_training"], counts["unfilled_validation"]) == (0, 1)
    # One unknown member leaves every label of its decision unknown.
    assert (counts["dependent_decisions"], counts["dependent_rows"]) == (
        1,
        qualification["members"][index],
    )


def test_a_vendor_flagged_delisting_return_is_counted_by_its_flag(tmp_path: Path) -> None:
    rows = study_rows()
    rows["delists"][0].update(delretmisstype="DG", delret=None)
    sources = _sources(tmp_path, rows)
    (refused,) = qualify_crsp_panel(sources)["refused"]
    index, ordinal = _position(sources, "2019-02-28", DELISTED)
    assert (refused["decision"], refused["ordinal"], refused["reason"]) == (
        index,
        ordinal,
        "delisting DG",
    )


#: The message a closing-trade check gives for a record it cannot execute at.
UNEXECUTABLE = "[CRSP-004] execution requires a finite positive closing trade"
#: A delisting return whose event the partition does not hold, by its stored day.
EVENTLESS = "[CRSP-009] the delisting return for 1 on {day} has no terminal event in this partition"


def test_a_delisting_the_partition_omits_is_named_rather_than_valued(tmp_path: Path) -> None:
    # The extract a partition writes when security 8's event is valued after
    # its observation end: the flagged daily row on DelDlyDt stays, the event
    # does not. Its -25% value must never be read from anywhere.
    rows = study_rows()
    rows["delists"] = []
    sources = _sources(tmp_path, rows)
    index, ordinal = _position(sources, "2019-02-28", DELISTED)
    returns = index_crsp_outcomes(sources).returns
    assert qualify_crsp_panel(sources)["refused"] == (
        CrspRefusedPath(
            ordinal=ordinal,
            decision=index,
            period="validation",
            reason="delisting return without event",
            entry_sha256=sha256(
                dump_json_str(returns[(DELISTED, "2019-03-01")]).encode()
            ).hexdigest(),
            exit_sha256=None,
        ),
    )


@pytest.mark.parametrize(
    ("rows", "delists", "refusal"),
    [
        (
            [],
            [],
            CrspPathRefusal("entry absent", "[CRSP-009] no execution record for 1 on 2020-02-03"),
        ),
        (
            [daily(1, W1.entry, 0.5, dlydelflg="Y", dlyretdurflg="DD", dlyprcflg="TR")],
            [event(W1.decision)],
            CrspPathRefusal(
                "entry terminal inconsistent", "[CRSP-010] inconsistent terminal entry for 1"
            ),
        ),
        (
            [daily(1, W1.entry, 0.0, dlyprc=None)],
            [],
            CrspPathRefusal("entry not a closing trade", UNEXECUTABLE),
        ),
        (
            [daily(1, W1.entry, 0.0)],
            [],
            CrspPathRefusal("absent row", "[CRSP-009] held return for 1 on 2020-02-04 is absent"),
        ),
        (
            [
                daily(1, W1.entry, 0.0),
                daily(1, "2020-02-04", 0.1),
                daily(1, W1.exit, 0.0, dlyprcflg="BA"),
            ],
            [],
            CrspPathRefusal("exit not a closing trade", UNEXECUTABLE),
        ),
        (
            [daily(1, W1.entry, 0.0), daily(1, "2020-02-04", 0.1), daily(1, W1.exit, 0.0)],
            [event("2020-02-04"), event("2020-02-03")],
            CrspPathRefusal(
                "two terminal events", "[CRSP-009] one window cannot hold two terminal events"
            ),
        ),
        (
            [
                daily(1, W1.entry, 0.0),
                daily(1, "2020-02-04", -0.4, dlydelflg="Y", dlyretdurflg="DD", dlyprcflg="DA"),
            ],
            [],
            CrspPathRefusal("delisting return without event", EVENTLESS.format(day="2020-02-04")),
        ),
        (
            [
                daily(1, W1.entry, 0.0),
                daily(1, "2020-02-04", 0.1),
                daily(1, W1.exit, 0.0, dlyprcflg="BA"),
                daily(1, "2020-03-03", -0.4, dlydelflg="Y", dlyretdurflg="DD", dlyprcflg="DA"),
            ],
            [],
            CrspPathRefusal("delisting return without event", EVENTLESS.format(day="2020-03-03")),
        ),
    ],
)
def test_each_refusal_has_its_counting_key_and_message(
    rows: list[JSONObject], delists: list[JSONObject], refusal: CrspPathRefusal
) -> None:
    assert crsp_resolve_window_path(outcomes(rows, delists), 1, W1) == refusal


def _refusals() -> CrspQualification:
    """Two decisions of three and two members, one refusal in each.

    The first is a training decision with one unfilled order.
    """
    return CrspQualification(
        members=(3, 2),
        unfilled=(1, 0),
        training_decisions=1,
        refused=(
            CrspRefusedPath(
                ordinal=2,
                decision=0,
                period="training",
                reason="span NT/MR",
                entry_sha256="a" * 64,
                exit_sha256=None,
            ),
            CrspRefusedPath(
                ordinal=5,
                decision=1,
                period="validation",
                reason="delisting MV",
                entry_sha256=None,
                exit_sha256="b" * 64,
            ),
        ),
    )


def test_codec_round_trips_and_counts_dependent_rows() -> None:
    value = _refusals()
    assert decode_crsp_qualification(encode_crsp_qualification(value)) == value
    counts = crsp_qualification_counts(value)
    assert counts["reasons"] == {"delisting MV": 1, "span NT/MR": 1}
    assert (counts["dependent_decisions"], counts["dependent_rows"]) == (2, 5)
    assert (counts["unfilled_training"], counts["unfilled_validation"]) == (1, 0)


@pytest.mark.parametrize(
    ("path", "field", "value", "message"),
    [
        (None, "extra", 1, "exact qualification fields"),
        (None, "members", [3, -1], "nonnegative"),
        (None, "unfilled", [1], "fit their decisions"),
        (None, "unfilled", [1, 3], "fit their decisions"),
        (None, "unfilled", [3, 0], "more unfilled and refused"),
        (None, "training_decisions", 3, "outside the panel"),
        (None, "training_decisions", -1, "outside the panel"),
        (0, "period", "validation", "disagrees with its decision"),
        (1, "period", "training", "disagrees with its decision"),
        (0, "extra", 1, "exact refused-path fields"),
        (0, "period", "final", "no panel decision or period"),
        (0, "decision", 2, "no panel decision or period"),
        (0, "entry_sha256", "A" * 64, "SHA-256 or null"),
        (0, "exit_sha256", "a" * 63, "SHA-256 or null"),
        (0, "ordinal", 4, "outside its decision's walk"),
        (1, "ordinal", 3, "outside its decision's walk"),
        (1, "decision", 0, "disagrees with its decision"),
    ],
)
def test_decoder_refuses_inconsistent_documents(
    path: int | None, field: str, value: JSONValue, message: str
) -> None:
    document = encode_crsp_qualification(_refusals())
    if path is None:
        document[field] = value
    else:
        narrow_json_to_dict(require_list(document, "results")[path])[field] = value
    with pytest.raises(ValueError, match=message):
        decode_crsp_qualification(document)


def test_decoder_refuses_ordinals_out_of_walk_order() -> None:
    document = encode_crsp_qualification(_refusals())
    second = narrow_json_to_dict(require_list(document, "results")[1])
    second.update(ordinal=1, decision=0, period="training")
    with pytest.raises(ValueError, match="strictly increase"):
        decode_crsp_qualification(document)
