"""Tests for the DatasetName vocabulary the API and scripts admit."""

from __future__ import annotations

from covenant_radar_api.dataset_names import BANKRUPTCY_DATASETS, DatasetName


def test_members_are_their_registry_keys() -> None:
    """Each member equals the covenant_ml registry key it names."""
    assert [str(m) for m in DatasetName] == [
        "taiwan",
        "us",
        "polish",
        "kaggle_give_me_credit",
        "kaggle_amex_default",
    ]


def test_bankruptcy_datasets_are_the_three_bundled_ones_in_message_order() -> None:
    """The API and explain script admit exactly these, listed in this order."""
    assert BANKRUPTCY_DATASETS == (DatasetName.TAIWAN, DatasetName.US, DatasetName.POLISH)
