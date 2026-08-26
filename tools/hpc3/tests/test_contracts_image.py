"""Tests for the image contract.

Four rules carry the weight, and each corresponds to a failure that produces a
working-looking artifact rather than an error:

* an unpinned requirement resolves at build time, reintroducing the drift the
  image exists to remove, and the build still succeeds;
* an environment prefix under a bind-mounted root is replaced at runtime by
  the host directory, so the image's interpreter vanishes inside its own
  image;
* an empty commit is stamped as provenance the image does not have, which is
  worse than the null a missing stamp produces;
* an empty assertion list yields an image that cannot detect its own
  staleness until a job has already waited for a GPU.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str

from hpc3.contracts.image import (
    HOST_BOUND_ROOTS,
    decode_image_reference,
    encode_image_reference,
)
from hpc3.contracts.image_spec import (
    decode_image_spec,
    encode_image_spec,
    encode_symbol_check,
    require_symbol_check,
)

_COMMITTED_SPEC = pathlib.Path(__file__).parent.parent / "specs" / "abl-image.json"
_DIGEST = "9ed4e27fd0d8207de3f84e833b98e0cf7e6ab09af66726849ca1cf023326cd51"


def _spec(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid image-spec payload with optional overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        A JSON object suitable for :func:`decode_image_spec`.
    """
    base: dict[str, JSONValue] = {
        "base_image": "python:3.11.16-slim-bookworm",
        "env_prefix": "/opt/env",
        "git_commit": "d11efacd231ef92426eaf92483c33a8504bd770f",
        "extra_index_urls": ["https://download.pytorch.org/whl/cu124"],
        "requirements": ["torch==2.6.0+cu124", "transformers==4.46.3"],
        "wheels": ["model_trainer_server-0.1.0-py3-none-any.whl"],
        "expected_versions": {"torch": "2.6.0+cu124"},
        "required_symbols": [{"module": "model_trainer.cluster.preflight", "attribute": "check"}],
        "smoke_commands": [],
        "labels": {"org.corvis.captured": "2026-08-25"},
    }
    base.update(overrides)
    return base


class TestRoundTrip:
    """Encoding a decoded spec must reproduce the document."""

    def test_decode_then_encode_is_the_original(self) -> None:
        payload = _spec()
        assert encode_image_spec(decode_image_spec(payload)) == payload

    def test_every_field_survives(self) -> None:
        spec = decode_image_spec(_spec())
        assert sorted(spec.keys()) == [
            "base_image",
            "env_prefix",
            "expected_versions",
            "extra_index_urls",
            "git_commit",
            "labels",
            "required_symbols",
            "requirements",
            "smoke_commands",
            "wheels",
        ]

    def test_a_symbol_check_round_trips(self) -> None:
        payload: JSONValue = {"module": "pkg.mod", "attribute": "thing"}
        assert encode_symbol_check(require_symbol_check(payload, "x")) == payload

    def test_encoding_copies_rather_than_aliases(self) -> None:
        """A caller mutating the encoded form must not reach the spec."""
        spec = decode_image_spec(_spec())
        encoded = encode_image_spec(spec)
        requirements = encoded["requirements"]
        if not isinstance(requirements, list):
            raise AssertionError("requirements must encode to a list")
        requirements.append("injected==1.0")
        assert spec["requirements"] == ["torch==2.6.0+cu124", "transformers==4.46.3"]


class TestPinnedRequirements:
    """A requirement that is not exactly pinned is the drift, not a warning."""

    @pytest.mark.parametrize("line", ["torch", "torch>=2.6.0", "torch~=2.6.0", "torch<3"])
    def test_an_unpinned_requirement_is_refused(self, line: str) -> None:
        with pytest.raises(JSONTypeError, match="must pin an exact version"):
            _ = decode_image_spec(_spec(requirements=[line]))

    def test_an_empty_requirement_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be empty"):
            _ = decode_image_spec(_spec(requirements=[]))

    def test_a_blank_line_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be blank"):
            _ = decode_image_spec(_spec(requirements=["   "]))

    def test_a_non_string_requirement_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a string"):
            _ = decode_image_spec(_spec(requirements=[7]))

    def test_surrounding_whitespace_is_stripped(self) -> None:
        spec = decode_image_spec(_spec(requirements=["  torch==2.6.0+cu124  "]))
        assert spec["requirements"] == ["torch==2.6.0+cu124"]


class TestEnvPrefix:
    """The prefix must survive the cluster's bind mounts."""

    @pytest.mark.parametrize("root", sorted(HOST_BOUND_ROOTS))
    def test_a_bind_mounted_root_is_refused(self, root: str) -> None:
        with pytest.raises(JSONTypeError, match="bind-mounts over"):
            _ = decode_image_spec(_spec(env_prefix=f"/{root}/wagnera3/envs/abl"))

    def test_a_relative_prefix_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="absolute POSIX path"):
            _ = decode_image_spec(_spec(env_prefix="opt/env"))

    def test_a_backslashed_prefix_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="forward-slashed"):
            _ = decode_image_spec(_spec(env_prefix="/opt\\env"))

    def test_a_parent_segment_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not contain"):
            _ = decode_image_spec(_spec(env_prefix="/opt/../pub/env"))

    def test_the_filesystem_root_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="filesystem root"):
            _ = decode_image_spec(_spec(env_prefix="/"))

    def test_a_trailing_slash_is_trimmed(self) -> None:
        spec = decode_image_spec(_spec(env_prefix="/opt/env/"))
        assert spec["env_prefix"] == "/opt/env"


class TestWheels:
    """Wheel names are joined onto a build directory."""

    @pytest.mark.parametrize("name", ["../escape.whl", "sub/dir.whl", "back\\slash.whl"])
    def test_a_separator_is_refused(self, name: str) -> None:
        with pytest.raises(JSONTypeError, match="path separator"):
            _ = decode_image_spec(_spec(wheels=[name]))

    def test_an_empty_wheel_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be empty"):
            _ = decode_image_spec(_spec(wheels=[]))

    def test_an_empty_wheel_name_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be empty"):
            _ = decode_image_spec(_spec(wheels=[""]))

    def test_a_non_string_wheel_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a string"):
            _ = decode_image_spec(_spec(wheels=[None]))

    def test_a_bare_dot_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must name a file"):
            _ = decode_image_spec(_spec(wheels=["."]))


class TestAssertionsAreMandatory:
    """An image that checks nothing about itself cannot detect staleness."""

    def test_an_empty_version_map_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be empty"):
            _ = decode_image_spec(_spec(expected_versions={}))

    def test_an_empty_symbol_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be empty"):
            _ = decode_image_spec(_spec(required_symbols=[]))

    def test_a_non_string_version_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a string"):
            _ = decode_image_spec(_spec(expected_versions={"torch": 2}))

    def test_an_empty_version_string_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be empty"):
            _ = decode_image_spec(_spec(expected_versions={"torch": ""}))

    def test_a_symbol_check_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            _ = decode_image_spec(_spec(required_symbols=["torch"]))

    def test_a_symbol_check_missing_its_attribute_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="attribute"):
            _ = decode_image_spec(_spec(required_symbols=[{"module": "pkg"}]))

    def test_an_empty_symbol_module_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be empty"):
            _ = decode_image_spec(_spec(required_symbols=[{"module": "", "attribute": "x"}]))


class TestRemainingFields:
    """The fields whose only rule is that they were decided."""

    def test_an_empty_commit_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be empty"):
            _ = decode_image_spec(_spec(git_commit=""))

    def test_an_empty_base_image_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be empty"):
            _ = decode_image_spec(_spec(base_image=""))

    def test_labels_may_be_empty(self) -> None:
        spec = decode_image_spec(_spec(labels={}))
        assert spec["labels"] == {}

    def test_extra_index_urls_may_be_empty(self) -> None:
        spec = decode_image_spec(_spec(extra_index_urls=[]))
        assert spec["extra_index_urls"] == []

    def test_a_non_string_index_url_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a string"):
            _ = decode_image_spec(_spec(extra_index_urls=[3]))

    def test_a_non_object_document_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            _ = decode_image_spec(["not", "an", "object"])

    def test_a_non_object_symbol_check_is_refused_directly(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            _ = require_symbol_check("nope", "where")


def _ref(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid image-reference payload with optional overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        A JSON object suitable for :func:`decode_image_reference`.
    """
    base: dict[str, JSONValue] = {
        "path": "/pub/wagnera3/images/abl.sif",
        "sha256": _DIGEST,
        "binds": ["/pub/wagnera3"],
    }
    base.update(overrides)
    return base


class TestImageReference:
    """What a job carries to say which image it runs inside."""

    def test_it_round_trips(self) -> None:
        payload: JSONValue = {
            "path": "/pub/wagnera3/images/abl.sif",
            "sha256": _DIGEST,
            "binds": ["/pub/wagnera3"],
        }
        assert encode_image_reference(decode_image_reference(payload, "image")) == payload

    def test_binds_are_required_rather_than_defaulted(self) -> None:
        """An unbound job on HPC3 finds none of its data and still starts."""
        with pytest.raises(JSONTypeError, match="binds"):
            _ = decode_image_reference({"path": "/pub/abl.sif", "sha256": _DIGEST}, "image")

    def test_binds_may_be_empty_for_a_self_contained_image(self) -> None:
        reference = decode_image_reference(
            {"path": "/pub/abl.sif", "sha256": _DIGEST, "binds": []}, "image"
        )
        if reference is None:
            raise AssertionError("a populated reference must decode")
        assert reference["binds"] == []

    @pytest.mark.parametrize("bad", ["relative/path", "/pub/../etc", "/pub\\wagnera3"])
    def test_a_malformed_bind_is_refused(self, bad: str) -> None:
        with pytest.raises(JSONTypeError, match="binds"):
            _ = decode_image_reference(
                {"path": "/pub/abl.sif", "sha256": _DIGEST, "binds": [bad]}, "image"
            )

    def test_a_non_string_bind_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a string"):
            _ = decode_image_reference(
                {"path": "/pub/abl.sif", "sha256": _DIGEST, "binds": [7]}, "image"
            )

    def test_a_trailing_slash_is_trimmed(self) -> None:
        reference = decode_image_reference(
            {"path": "/pub/abl.sif", "sha256": _DIGEST, "binds": ["/pub/wagnera3/"]}, "image"
        )
        if reference is None:
            raise AssertionError("a populated reference must decode")
        assert reference["binds"] == ["/pub/wagnera3"]

    def test_absence_means_no_image(self) -> None:
        """A host run is a state, not a missing value."""
        assert decode_image_reference(None, "image") is None
        assert encode_image_reference(None) is None

    def test_a_bind_mounted_path_is_allowed(self) -> None:
        """Unlike env_prefix: the .sif is a file the HOST reads."""
        reference = decode_image_reference(
            {
                "path": "/pub/wagnera3/images/abl.sif",
                "sha256": _DIGEST,
                "binds": ["/pub/wagnera3"],
            },
            "image",
        )
        if reference is None:
            raise AssertionError("a populated reference must decode")
        assert reference["path"] == "/pub/wagnera3/images/abl.sif"

    def test_a_relative_path_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="absolute POSIX path"):
            _ = decode_image_reference(_ref(path="images/abl.sif"), "image")

    def test_a_parent_segment_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not contain"):
            _ = decode_image_reference(_ref(path="/pub/../etc/abl.sif"), "image")

    def test_a_backslashed_path_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="forward-slashed"):
            _ = decode_image_reference({"path": "/pub\\abl.sif", "sha256": _DIGEST}, "image")

    @pytest.mark.parametrize(
        "digest",
        ["", "abc", _DIGEST.upper(), _DIGEST[:-1], _DIGEST[:-1] + "g"],
    )
    def test_a_malformed_digest_is_refused(self, digest: str) -> None:
        """A re-cased or truncated digest names different bytes."""
        with pytest.raises(JSONTypeError, match="lowercase hex"):
            _ = decode_image_reference(_ref(sha256=digest), "image")

    def test_a_missing_digest_is_refused(self) -> None:
        """A path can be rebuilt in place; only the digest names bytes."""
        with pytest.raises(JSONTypeError, match="sha256"):
            _ = decode_image_reference({"path": "/pub/abl.sif", "binds": []}, "image")

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            _ = decode_image_reference("/pub/abl.sif", "image")


class TestTheCommittedSpec:
    """The spec in the repository must satisfy the contract that reads it.

    A spec is only reproducibility evidence if it still decodes. Validating
    it here means a rule tightened later fails in CI rather than on a cluster
    at build time, when the wheels are already staged and a GPU is already
    reserved.
    """

    def test_it_decodes(self) -> None:
        raw = load_json_str(_COMMITTED_SPEC.read_text(encoding="utf-8"))
        spec = decode_image_spec(raw)
        assert spec["expected_versions"] == {"torch": "2.6.0+cu124", "transformers": "4.46.3"}

    def test_it_round_trips_byte_for_byte_through_the_contract(self) -> None:
        raw = load_json_str(_COMMITTED_SPEC.read_text(encoding="utf-8"))
        assert encode_image_spec(decode_image_spec(raw)) == raw

    def test_it_asserts_every_symbol_this_image_exists_to_carry(self) -> None:
        """The symbols whose absence means a stale wheel was baked in.

        Asserted as an EXACT set, and on the (module, attribute) pair rather
        than the attribute alone. Exact, because adding a required symbol
        widens what the build refuses and should be a reviewed act rather
        than something a spec edit does quietly. Paired, because ``main`` is
        a name half this repository exports -- an attribute-only assertion
        would be satisfied by the wrong module's ``main`` and report the
        image carrying a scorer it does not have.

        The first two are the fixes the image was originally built to carry.
        The third is ``modeltrainer-score-baseline``, added 2026-08-25: the
        v2 image did not have it, which is why the A100 floor could not be
        measured until v3 was built.

        The fourth is ``score_with_outcomes``, and it is here because ``main``
        cannot do its job alone. ``main`` existed in v3 too, so a v4 built
        against a STALE wheel would carry it, pass the self-check, and then
        fail on the cluster with an unknown ``--outcomes`` flag. A required
        symbol only detects a stale wheel if it names something the new code
        introduced.

        The last two are the environment known-answer probe, added by
        another session on 2026-08-25. They arrived here as a FAILURE of
        this test rather than as a silent widening, which is the exactness
        earning its keep: the spec grew, and someone had to look.
        """
        spec = decode_image_spec(load_json_str(_COMMITTED_SPEC.read_text(encoding="utf-8")))
        symbols = sorted(
            (check["module"], check["attribute"]) for check in spec["required_symbols"]
        )
        assert symbols == [
            ("model_trainer.cli.known_answer_probe", "probe_run_record"),
            ("model_trainer.cli.score_baseline", "main"),
            ("model_trainer.cli.score_baseline", "score_with_outcomes"),
            ("model_trainer.cluster.preflight", "check_corpus_certified"),
            (
                "model_trainer.core.services.model.known_answer_probe",
                "probe_forward_loss",
            ),
            (
                "model_trainer.core.services.training.base_trainer_checkpoints",
                "_TrainerCheckpoints",
            ),
        ]

    def test_its_environment_survives_the_cluster_bind_mounts(self) -> None:
        spec = decode_image_spec(load_json_str(_COMMITTED_SPEC.read_text(encoding="utf-8")))
        assert spec["env_prefix"] == "/opt/env"
