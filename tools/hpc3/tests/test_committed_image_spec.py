"""The committed abl image spec, held to the contract that reads it.

Split from ``test_contracts_image.py``, which held both the ImageSpec
contract's unit tests and this class until the file crossed the size ceiling.
The two roles part cleanly: that file exercises the codec against synthetic
payloads, this one holds the artifact in ``specs/`` to it -- the same
committed-codegen shape ``test_committed_campaign.py`` guards one repo over,
and the same reason: a committed spec that stops decoding is not evidence of
anything.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import load_json_str

from hpc3.contracts.image_spec import decode_image_spec, encode_image_spec

_COMMITTED_SPEC = pathlib.Path(__file__).parent.parent / "specs" / "abl-image.json"


class TestTheCommittedSpec:
    """The spec in the repository must satisfy the contract that reads it.

    A spec is only reproducibility evidence if it still decodes. Validating
    it here means a rule tightened later fails in CI rather than on a cluster
    at build time, when the wheels are already staged and a GPU is already
    reserved.

    THE SYMBOL LIST THAT USED TO LIVE HERE IS GONE, AND THE REASON MATTERS.
    It was forty-six ``(module, attribute)`` pairs transcribed by hand, and
    its own docstring recorded fourteen recurrences of the same failure: the
    spec grew, the list did not, and ``make check`` in this package was red
    on ``main`` across nine consecutive commits while the signal it existed
    to give was stepped over rather than read.

    That docstring also explained why the check could not be generated: "the
    generator that would end this would derive the list from the code --
    import each module and resolve each attribute -- and this package cannot:
    ``hpc3`` is its own poetry project and ``import model_trainer`` fails in
    its venv."

    THE PREMISE WAS WRONG, and only in one word. The check does not have to
    IMPORT. :mod:`tests.test_committed_specs` resolves every declared symbol
    by PARSING the monorepo's source, which needs no shared virtualenv and no
    torch, and it holds the property the exact list was standing in for --
    that no wheel an image installs goes unnamed -- as an invariant rather
    than as a transcription. What nobody retypes cannot drift.
    """

    def test_it_decodes(self) -> None:
        raw = load_json_str(_COMMITTED_SPEC.read_text(encoding="utf-8"))
        spec = decode_image_spec(raw)
        assert spec["expected_versions"] == {
            "torch": "2.6.0+cu124",
            "transformers": "4.46.3",
            # Pinned when QLoRA gained real quantization (board note
            # 2026-09-01, opus-corpus-docmode-0901): the spec's runtime
            # probe must hold the image to the bitsandbytes it was built
            # with, or a rebuilt image could silently load un-quantized.
            "bitsandbytes": "0.45.5",
        }

    def test_it_round_trips_byte_for_byte_through_the_contract(self) -> None:
        raw = load_json_str(_COMMITTED_SPEC.read_text(encoding="utf-8"))
        assert encode_image_spec(decode_image_spec(raw)) == raw

    def test_its_environment_survives_the_cluster_bind_mounts(self) -> None:
        spec = decode_image_spec(load_json_str(_COMMITTED_SPEC.read_text(encoding="utf-8")))
        assert spec["env_prefix"] == "/opt/env"
