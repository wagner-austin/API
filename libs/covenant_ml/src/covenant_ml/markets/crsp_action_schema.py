"""Fixed corporate-action diagnostic projections, separate from the study protocol."""

from __future__ import annotations

from typing import NamedTuple


class CrspActionRelation(NamedTuple):
    """One exact projection with its source key and all observed date fields."""

    role: str
    table: str
    columns: tuple[str, ...]
    key: tuple[str, ...]
    dates: tuple[str, ...]


#: Event projections keep identity without importing out-of-bound payment values.
CRSP_ACTIONS = (
    CrspActionRelation(
        "delist-events",
        "stkdelists",
        ("permno", "delistingdt"),
        ("permno", "delistingdt"),
        ("delistingdt",),
    ),
    CrspActionRelation(
        "delist-details",
        "stkdelists",
        (
            "permno",
            "delistingdt",
            "deldtprc",
            "deldtprcflg",
            "delactiontype",
            "delstatustype",
            "delreasontype",
            "delpaymenttype",
            "delpermno",
            "delpermco",
            "delret",
            "delretmisstype",
            "delnextdt",
            "delnextprc",
            "delnextprcflg",
            "delamtdt",
            "deldivamt",
            "deldistype",
            "deldlydt",
        ),
        ("permno", "delistingdt"),
        ("delistingdt", "delnextdt", "delamtdt", "deldlydt"),
    ),
    CrspActionRelation(
        "distribution-events",
        "stkdistributions",
        ("permno", "disexdt", "disseqnbr"),
        ("permno", "disexdt", "disseqnbr"),
        ("disexdt",),
    ),
    CrspActionRelation(
        "distributions",
        "stkdistributions",
        (
            "permno",
            "disexdt",
            "disseqnbr",
            "disordinaryflg",
            "distype",
            "disfreqtype",
            "dispaymenttype",
            "disdetailtype",
            "distaxtype",
            "disorigcurtype",
            "disdivamt",
            "disfacpr",
            "disfacshr",
            "disdeclaredt",
            "disrecorddt",
            "dispaydt",
            "dispermno",
            "dispermco",
            "disamountsourcetype",
        ),
        ("permno", "disexdt", "disseqnbr"),
        ("disexdt", "disdeclaredt", "disrecorddt", "dispaydt"),
    ),
)


__all__ = ["CRSP_ACTIONS", "CrspActionRelation"]
