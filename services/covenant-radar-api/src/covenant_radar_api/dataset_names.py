"""The external datasets this service's API and scripts name by word.

Workers resolve a dataset through covenant_ml's registry, whose keys are open
(a test registers its own). This vocabulary is the closed set the edges
admit: the HTTP API and the explain script take the three bundled bankruptcy
datasets, and the optimize script every member.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class DatasetName(StrEnum):
    """A registry dataset an API request or script argument may name."""

    TAIWAN = "taiwan"
    US = "us"
    POLISH = "polish"
    KAGGLE_GIVE_ME_CREDIT = "kaggle_give_me_credit"
    KAGGLE_AMEX_DEFAULT = "kaggle_amex_default"


# The three bundled bankruptcy datasets, in the order messages list them.
BANKRUPTCY_DATASETS: Final[tuple[DatasetName, ...]] = (
    DatasetName.TAIWAN,
    DatasetName.US,
    DatasetName.POLISH,
)


__all__ = ["BANKRUPTCY_DATASETS", "DatasetName"]
