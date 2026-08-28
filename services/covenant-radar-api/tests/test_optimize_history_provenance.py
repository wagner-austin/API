"""A history row says which configuration produced it, or says it does not.

``optimization_history.jsonl`` is a longitudinal comparison: ``best_val_auc``
and ``duration_seconds`` across seven backends, four datasets and months of
runs, kept expressly to track progression. Until 2026-08-28 it recorded
neither the machine those numbers came from nor the versions of the libraries
whose arithmetic produced them -- so nothing said whether two of its rows were
comparable at all, which is the question
:mod:`platform_core.comparability` exists to answer.

Split from ``test_optimize_history.py`` when that file passed the 600-line
ceiling. The split is by role: everything here is about whether a row can
state its own provenance, and nothing here is about history management.
"""

from __future__ import annotations

import pytest
from platform_core.comparability import encode_run_fingerprint
from platform_core.json_utils import JSONObject, JSONTypeError
from platform_core.testing import sample_run_fingerprint
from scripts.optimize.history import (
    UnifiedHistoryEntry,
    _decode_history_entry,
    _encode_history_entry,
)


def _json_row(**overrides: JSONObject) -> JSONObject:
    """Build a decodable row carrying an explicit null fingerprint.

    Args:
        **overrides: Fields to replace.

    Returns:
        A fresh object each call, so a test may mutate it.
    """
    row: JSONObject = {
        "timestamp": "2024-01-01T00:00:00Z",
        "backend": "xgboost",
        "dataset": "taiwan",
        "feature_preset": "full",
        "n_trials": 50,
        "n_samples": 1000,
        "n_features": 100,
        "best_val_auc": 0.85,
        "best_trial_number": 25,
        "duration_seconds": 60.0,
        "fingerprint": None,
    }
    row.update(overrides)
    return row


def _entry() -> UnifiedHistoryEntry:
    """Build a decoded row with no fingerprint.

    Returns:
        The entry.
    """
    return _decode_history_entry(_json_row())


class TestTheFingerprintIsThreeState:
    """Carried, explicitly absent, or a bug -- and they are not the same."""

    def test_a_row_carrying_a_fingerprint_decodes_it(self) -> None:
        fingerprint = sample_run_fingerprint(gpu_model="L40S")
        row = _json_row(fingerprint=encode_run_fingerprint(fingerprint))

        assert _decode_history_entry(row)["fingerprint"] == fingerprint

    def test_an_explicit_null_decodes_to_none(self) -> None:
        """Which reads as "nobody recorded one", not "the configuration was
        unremarkable". 3,068 rows written before the field existed say this."""
        assert _decode_history_entry(_json_row())["fingerprint"] is None

    def test_a_missing_key_is_refused_rather_than_read_as_none(self) -> None:
        """The distinction the field exists for. A row that simply omits it
        cannot be told from a writer that forgot, so the reader must not have
        to guess which one it is looking at."""
        row = _json_row()
        del row["fingerprint"]

        with pytest.raises(JSONTypeError, match="Field 'fingerprint' is required"):
            _decode_history_entry(row)


class TestARowSurvivesBeingWritten:
    """The encoder exists because the row used to be dumped as itself.

    That worked while every field was a JSON scalar. A fingerprint is not: its
    determinism settings are sorted pairs and its package versions are a tuple
    of records, and ``json`` renders both as bare arrays that
    ``decode_run_fingerprint`` refuses. A row that cannot be read back is not
    a record.
    """

    def test_a_fingerprinted_row_round_trips(self) -> None:
        entry = _entry()
        entry["fingerprint"] = sample_run_fingerprint(image_digest="sha256:abc")

        assert _decode_history_entry(_encode_history_entry(entry)) == entry

    def test_an_unfingerprinted_row_encodes_the_null_rather_than_dropping_it(self) -> None:
        encoded = _encode_history_entry(_entry())

        assert encoded["fingerprint"] is None

    def test_the_encoded_row_carries_every_field_the_decoder_requires(self) -> None:
        """A dropped key would be indistinguishable from a row that predates
        whichever field was dropped, which is the failure this whole file is
        about."""
        assert sorted(_encode_history_entry(_entry())) == sorted(_json_row())
