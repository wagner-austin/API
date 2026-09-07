"""The record that admits a staged corpus to a training run.

``model_trainer.cluster.preflight.check_corpus_certified`` refuses a corpus
whose digest no ``*-digests.txt`` beside it names, and its docstring has always
said "``hpc3-stage`` writes one, and a file put there by hand has none". Until
2026-09-04 this package wrote nothing: the supported staging path produced
corpora the supported training path refused, and the only way through was to
hand-write the very file the rule exists to distinguish from a hand-placed one.

The consumer lives in another package that this one does not depend on, so the
contract cannot be shared by import. It is pinned here instead, by reading the
consumer's own source, so the pairing fails a test rather than a job.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from platform_core.json_utils import JSONValue

from hpc3.contracts.stage import StageManifest, decode_stage_manifest
from hpc3.core.stage import CERTIFICATION_SUFFIX, certification_path, certification_text

#: What the consumer scans for. A 64-character lowercase hex token.
CONSUMER_DIGEST_TOKEN = re.compile(r"\b[0-9a-f]{64}\b")

_A = "a" * 64
_B = "b" * 64


def _manifest(destination: str = "/pub/wagnera3/abl/redo") -> StageManifest:
    """Build a decoded manifest naming two files.

    Args:
        destination: Cluster directory to receive them.

    Returns:
        The validated manifest.
    """
    document: dict[str, JSONValue] = {
        "destination": destination,
        "files": [
            {"name": _A, "sha256": _A, "size_bytes": 10},
            {"name": "cloze.jsonl", "sha256": _B, "size_bytes": 20},
        ],
        "provenance": {"wiki_commit": "e2be2a3", "emitter": "emit_corpus"},
    }
    return decode_stage_manifest(document)


def _consumer_source() -> str:
    """Read the preflight module that consumes these records.

    Returns:
        Its source text.

    Raises:
        pytest.skip.Exception: When Model-Trainer is not checked out beside
            this package, which is the case in a packaged install.
    """
    path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "services"
        / "Model-Trainer"
        / "src"
        / "model_trainer"
        / "cluster"
        / "preflight.py"
    )
    if not path.is_file():
        pytest.skip(f"consumer not checked out at {path}")
    return path.read_text(encoding="utf-8")


class TestTheContractMatchesItsConsumer:
    def test_the_suffix_is_the_one_the_consumer_globs_for(self) -> None:
        assert f'CERTIFICATION_SUFFIX = "{CERTIFICATION_SUFFIX}"' in _consumer_source()

    def test_every_digest_this_writes_is_a_token_the_consumer_finds(self) -> None:
        # finditer, not findall, for the reason the consumer states at its own
        # call site: findall is typed list[Any], which strict mode rejects.
        found = {
            match.group(0)
            for match in CONSUMER_DIGEST_TOKEN.finditer(certification_text(_manifest()))
        }
        assert _A in found
        assert _B in found

    def test_the_consumer_still_requires_a_certification(self) -> None:
        """If this check were dropped upstream, writing the record would
        become dead weight rather than a fix, and this test says so."""
        assert "def check_corpus_certified" in _consumer_source()


class TestCertificationText:
    def test_it_names_every_staged_file(self) -> None:
        text = certification_text(_manifest())
        assert f"{_A}  {_A}" in text
        assert f"{_B}  cloze.jsonl" in text

    def test_it_carries_the_provenance(self) -> None:
        assert "provenance emitter=emit_corpus wiki_commit=e2be2a3" in certification_text(
            _manifest()
        )

    def test_it_ends_with_a_newline(self) -> None:
        assert certification_text(_manifest()).endswith("\n")

    def test_a_digest_absent_from_the_manifest_is_absent_from_the_record(self) -> None:
        assert "c" * 64 not in certification_text(_manifest())


class TestCertificationPath:
    def test_it_lands_beside_the_corpora_it_admits(self) -> None:
        assert (
            certification_path(_manifest(), "stage-abl-redo")
            == "/pub/wagnera3/abl/redo/stage-abl-redo-digests.txt"
        )

    def test_it_follows_the_destination(self) -> None:
        assert certification_path(_manifest("/pub/other"), "m").startswith("/pub/other/")
