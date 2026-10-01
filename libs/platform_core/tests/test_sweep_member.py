"""Tests for one sweep member and the artifact every run declares.

The artifact rule carries the weight. A sweep is where an undeclared or
misdeclared output costs most: six arms that were all going to be compared,
and an index that cannot say where any of their results went.
"""

from __future__ import annotations

import pytest

from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.sweep_member import (
    SweepMember,
    decode_sweep_member,
    encode_sweep_member,
    require_artifact_in_command,
)


class TestMemberValidation:
    def test_a_non_object_member_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object, got str"):
            decode_sweep_member("s0")

    def test_an_empty_suffix_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="Field 'suffix' must not be empty"):
            decode_sweep_member({"suffix": "", "command": "x", "artifact": None})

    def test_an_empty_command_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="Field 'command' must not be empty"):
            decode_sweep_member({"suffix": "s0", "command": "", "artifact": None})

    def test_a_slashed_suffix_is_refused(self) -> None:
        """The suffix reaches a log filename; a separator would escape it."""
        with pytest.raises(JSONTypeError, match="path separator"):
            decode_sweep_member({"suffix": "a/b", "command": "x", "artifact": None})
        with pytest.raises(JSONTypeError, match="path separator"):
            decode_sweep_member({"suffix": "a\\b", "command": "x", "artifact": None})

    def test_a_valid_member_decodes(self) -> None:
        member = decode_sweep_member({"suffix": "s0", "command": "python x.py", "artifact": None})
        assert member == {"suffix": "s0", "command": "python x.py", "artifact": None}

    def test_a_member_that_never_mentions_an_artifact_is_refused(self) -> None:
        """Per member, because a sweep is where the omission costs most: six
        arms silently sharing no declared output is six results nobody can
        reach, and they were all going to be compared."""
        with pytest.raises(JSONTypeError, match="Field 'artifact' is required"):
            decode_sweep_member({"suffix": "s0", "command": "python x.py"})

    def test_a_member_declaring_an_artifact_its_command_writes_is_kept(self) -> None:
        """Per member, because six arms writing one path are five lost results."""
        member = decode_sweep_member(
            {"suffix": "s0", "command": "python x.py --out /r/s0.json", "artifact": "/r/s0.json"}
        )

        assert member["artifact"] == "/r/s0.json"

    def test_a_member_whose_artifact_its_command_never_writes_is_refused(self) -> None:
        """The suffix edited in one place and not the other -- the real failure."""
        with pytest.raises(JSONTypeError, match="does not appear in this run's command"):
            decode_sweep_member(
                {
                    "suffix": "s1",
                    "command": "python x.py --out /r/s0.json",
                    "artifact": "/r/s1.json",
                }
            )


class TestMemberEncoding:
    def test_a_member_round_trips(self) -> None:
        payload: dict[str, JSONValue] = {
            "suffix": "repA",
            "command": "python -m rw_bot.harness.campaign_match --result /pub/r/repA.json",
            "artifact": "/pub/r/repA.json",
        }
        assert encode_sweep_member(decode_sweep_member(payload)) == payload

    def test_a_null_artifact_is_written_rather_than_dropped(self) -> None:
        """The key is required on decode, so dropping a null would write a
        document the decoder refuses."""
        member = SweepMember(suffix="s0", command="python x.py", artifact=None)

        encoded = encode_sweep_member(member)

        assert "artifact" in encoded
        assert encoded["artifact"] is None
        assert decode_sweep_member(encoded) == member


class TestRequireArtifactInCommand:
    """Called directly, as ``hpc3``'s job spec calls it on a whole run document."""

    def test_a_path_the_command_writes_is_kept(self) -> None:
        assert (
            require_artifact_in_command({"artifact": "/r/a.json"}, "score --out /r/a.json")
            == "/r/a.json"
        )

    def test_null_states_the_run_writes_nothing_durable(self) -> None:
        assert require_artifact_in_command({"artifact": None}, "smoke") is None

    def test_an_absent_key_is_refused_rather_than_read_as_null(self) -> None:
        with pytest.raises(JSONTypeError, match="Write null to state"):
            require_artifact_in_command({}, "score --out /r/a.json")

    def test_a_non_string_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a string or null, got int"):
            require_artifact_in_command({"artifact": 7}, "score")

    def test_an_empty_string_is_refused_rather_than_read_as_null(self) -> None:
        """Null declares no artifact; empty declares one and names nowhere."""
        with pytest.raises(JSONTypeError, match="not an empty string"):
            require_artifact_in_command({"artifact": ""}, "score")

    def test_a_path_the_command_never_mentions_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match=r"'/r/b\.json', which does not appear"):
            require_artifact_in_command({"artifact": "/r/b.json"}, "score --out /r/a.json")

    def test_a_directory_the_command_writes_into_is_kept(self) -> None:
        """A run writing several files into one place names the place."""
        assert (
            require_artifact_in_command({"artifact": "/r/out"}, "score --out-dir /r/out")
            == "/r/out"
        )
