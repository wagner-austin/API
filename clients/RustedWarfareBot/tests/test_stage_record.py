"""Turning a packed archive into the manifest that stages it.

Small because the act is small, and here at all because the alternative was
keeping the manifest by hand -- which is what the two under ``provenance/``
were, and what let one of them go on naming a 407-entry archive after the
tree had grown to 409.
"""

from __future__ import annotations

import pytest
from hpc3.contracts.stage import decode_stage_manifest, encode_stage_manifest
from platform_core.json_utils import JSONTypeError

from rw_bot.stage_record import stage_manifest
from rw_bot.tree_archive import ArchiveResult

_DESTINATION = "/pub/wagnera3/rusted/staging"
_DIGEST = "bae37748c8b2539c81834f64179cbee86a4cff0dd70938131a80ea0788b1251c"
_ARCHIVE = ArchiveResult(sha256=_DIGEST, size_bytes=2232320, executables=("jvm-linux/bin/java",))
_PROVENANCE = {"git_commit": "7fc59eda039a06b395935bba00618b775744ec4b", "entries": "409"}


class TestComposingIt:
    def test_the_file_it_names_is_the_archive_it_was_given(self) -> None:
        manifest = stage_manifest(_DESTINATION, "rw-payload.tar", _ARCHIVE, _PROVENANCE)
        assert manifest["files"] == [
            {"name": "rw-payload.tar", "sha256": _DIGEST, "size_bytes": 2232320}
        ]

    def test_the_destination_is_carried_as_given(self) -> None:
        manifest = stage_manifest(_DESTINATION, "rw-payload.tar", _ARCHIVE, _PROVENANCE)
        assert manifest["destination"] == _DESTINATION

    def test_the_provenance_is_copied_not_aliased(self) -> None:
        """A caller that went on editing its own mapping would otherwise be
        editing the manifest, which is the kind of sharing that makes a
        record disagree with the act that produced it."""
        source = dict(_PROVENANCE)
        manifest = stage_manifest(_DESTINATION, "rw-payload.tar", _ARCHIVE, source)
        source["entries"] = "1"
        assert manifest["provenance"]["entries"] == "409"

    def test_what_it_composes_survives_hpc3s_own_decoder(self) -> None:
        """The manifest is written for ``hpc3-stage`` to read, so agreeing
        with the encoder alone would prove nothing."""
        manifest = stage_manifest(_DESTINATION, "rw-payload.tar", _ARCHIVE, _PROVENANCE)
        assert decode_stage_manifest(encode_stage_manifest(manifest)) == manifest

    def test_a_manifest_with_no_provenance_is_refused_by_that_decoder(self) -> None:
        """Not defended against here: the contract already refuses it, and a
        second check in front of the first is a second thing to keep in
        step. Asserted so the reliance is stated rather than assumed."""
        manifest = stage_manifest(_DESTINATION, "rw-payload.tar", _ARCHIVE, {})
        with pytest.raises(JSONTypeError, match="must record at least one fact"):
            decode_stage_manifest(encode_stage_manifest(manifest))
