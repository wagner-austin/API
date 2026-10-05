"""Tests for the code claim: the half of the repo_commit check a CI clone CAN run.

Whether a declared commit is the code a run executes can only be answered on
the cluster, at submission. Whether a run document makes that claim in a form
submission can check is a question about the document alone, so it is asked
at decode -- and therefore by ``test_committed_runs`` over every committed
document, in CI, at depth 1.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.code_claim import REPO_COMMIT_KEY, REPO_TREE_KEY, code_claim
from tests.against_hpc3 import decode_job_spec

_TREE = "/pub/wagnera3/api"


def _job(experiment: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Build an undecoded CPU job carrying an experiment record.

    Args:
        experiment: The record to carry.

    Returns:
        A job document that is valid in every other respect.
    """
    return {
        "project": "cleargbm",
        "name": "p6-rung5",
        "partition": "free",
        "gpu": None,
        "cpus": 4,
        "mem_gb": 16,
        "minutes": 60,
        "requeue": False,
        "resumes_from_checkpoint": False,
        "env_path": "/pub/wagnera3/envs/cleargbm",
        "pinned_packages": {},
        "deterministic": True,
        "experiment": experiment,
        "command": "python -m scripts.optimize",
        "artifact": None,
    }


class TestReadingTheClaim:
    def test_a_record_naming_neither_key_makes_no_claim(self) -> None:
        assert code_claim({"phase": "P6"}) is None

    def test_both_keys_make_a_claim_carrying_each_as_written(self) -> None:
        claim = code_claim({REPO_COMMIT_KEY: "80221ea", REPO_TREE_KEY: _TREE, "phase": "P6"})
        assert claim == {"tree": _TREE, "commit": "80221ea"}

    def test_a_full_forty_character_sha_is_accepted(self) -> None:
        sha = "80221ea1056e08aacd3f5ee01e6c599e166970be"
        claim = code_claim({REPO_COMMIT_KEY: sha, REPO_TREE_KEY: _TREE})
        assert claim is not None
        assert claim["commit"] == sha

    def test_a_commit_without_its_tree_is_refused_naming_the_tree(self) -> None:
        """Rung 5's shape: a commit, and nothing saying which checkout it describes."""
        with pytest.raises(JSONTypeError) as caught:
            code_claim({REPO_COMMIT_KEY: "20d9159"})
        assert "declares 'repo_commit' without 'repo_tree'" in str(caught.value)

    def test_a_tree_without_its_commit_is_refused_naming_the_commit(self) -> None:
        with pytest.raises(JSONTypeError) as caught:
            code_claim({REPO_TREE_KEY: _TREE})
        assert "declares 'repo_tree' without 'repo_commit'" in str(caught.value)

    def test_a_relative_tree_is_refused(self) -> None:
        """The submitter's relative path and the cluster's are different trees."""
        with pytest.raises(JSONTypeError) as caught:
            code_claim({REPO_COMMIT_KEY: "80221ea", REPO_TREE_KEY: "api"})
        assert "absolute cluster path, got 'api'" in str(caught.value)

    @pytest.mark.parametrize("commit", ["80221e", "80221EA", "80221eg", "8" * 41])
    def test_a_commit_that_is_not_seven_to_forty_lowercase_hex_is_refused(
        self, commit: str
    ) -> None:
        with pytest.raises(JSONTypeError) as caught:
            code_claim({REPO_COMMIT_KEY: commit, REPO_TREE_KEY: _TREE})
        assert f"7 to 40 lowercase hex characters, got {commit!r}" in str(caught.value)


class TestDecodingAJob:
    def test_a_job_with_half_a_claim_cannot_be_decoded(self) -> None:
        with pytest.raises(JSONTypeError) as caught:
            decode_job_spec(_job({REPO_COMMIT_KEY: "20d9159", "phase": "P6"}))
        assert "without 'repo_tree'" in str(caught.value)

    def test_a_job_with_a_whole_claim_keeps_its_record_unchanged(self) -> None:
        record: dict[str, JSONValue] = {
            REPO_COMMIT_KEY: "80221ea",
            REPO_TREE_KEY: _TREE,
            "phase": "P6",
        }
        assert decode_job_spec(_job(record))["experiment"] == record
