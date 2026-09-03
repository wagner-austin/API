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

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

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

_DIGEST = "9ed4e27fd0d8207de3f84e833b98e0cf7e6ab09af66726849ca1cf023326cd51"


BASE_IMAGE = "python:3.11.16-slim-bookworm@sha256:" + "b3" * 32
"""A digest-pinned base, because the spec contract refuses a bare tag.

Composed rather than written out so the line fits, and so the 64-character
digest is obviously synthetic rather than mistaken for a real one.
"""


def _spec(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid image-spec payload with optional overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        A JSON object suitable for :func:`decode_image_spec`.
    """
    base: dict[str, JSONValue] = {
        "base_image": BASE_IMAGE,
        "env_prefix": "/opt/env",
        "git_commit": "d11efacd231ef92426eaf92483c33a8504bd770f",
        "system_packages": [],
        "extra_index_urls": ["https://download.pytorch.org/whl/cu124"],
        "requirements": ["torch==2.6.0+cu124", "transformers==4.46.3"],
        "wheels": ["model_trainer_server-0.1.0-py3-none-any.whl"],
        "expected_versions": {"torch": "2.6.0+cu124"},
        "required_symbols": [{"module": "model_trainer.cluster.preflight", "attribute": "check"}],
        "smoke_commands": [],
        "labels": {"org.corvis.captured": "2026-08-25"},
        "project": "abl",
    }
    base.update(overrides)
    return base


class TestTheBaseImageMustPinADigest:
    """A tag is a mutable pointer, and this document exists to pin things.

    Not hypothetical: rusted pinned python:3.11-slim-bookworm@sha256:0bee7276,
    and that same tag now resolves to sha256:528257d4. The tag moved under
    four specs that named it bare, and nothing in the workspace noticed.
    """

    def test_a_bare_tag_is_refused(self) -> None:
        """The whole point: two builds a week apart could differ in silence."""
        with pytest.raises(JSONTypeError, match="must pin a digest"):
            _ = decode_image_spec(_spec(base_image="python:3.11.16-slim-bookworm"))

    def test_a_reference_that_is_only_a_digest_is_refused(self) -> None:
        """An empty image half names nothing to pull."""
        with pytest.raises(JSONTypeError, match="must pin a digest"):
            _ = decode_image_spec(_spec(base_image="@sha256:" + "b3" * 32))

    def test_a_digest_of_the_wrong_length_is_refused(self) -> None:
        """A truncated digest is the one someone pasted by eye."""
        with pytest.raises(JSONTypeError, match="lowercase hex"):
            _ = decode_image_spec(_spec(base_image="python:3.11@sha256:b3b3b3"))

    def test_an_uppercase_digest_is_refused(self) -> None:
        """Registries emit lowercase; anything else was retyped."""
        with pytest.raises(JSONTypeError, match="lowercase hex"):
            _ = decode_image_spec(_spec(base_image="python:3.11@sha256:" + "B3" * 32))

    def test_a_pinned_reference_is_admitted_unchanged(self) -> None:
        """The tag is kept beside the digest; rusted's built images carry both."""
        pinned = "python:3.11.16-slim-bookworm@sha256:" + "b3" * 32

        assert decode_image_spec(_spec(base_image=pinned))["base_image"] == pinned


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
            "project",
            "required_symbols",
            "requirements",
            "smoke_commands",
            "system_packages",
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


class TestPinnedSystemPackages:
    """The same argument as the pip layer, in the distribution's syntax."""

    @pytest.mark.parametrize("entry", ["xvfb", "openjdk-17-jre-headless", "mesa-utils"])
    def test_an_unpinned_package_is_refused(self, entry: str) -> None:
        """An unpinned apt install resolves against whatever the distribution
        serves that day and SUCCEEDS, so two images built a week apart differ
        with nothing recording that they do."""
        with pytest.raises(JSONTypeError, match="must pin an exact version"):
            _ = decode_image_spec(_spec(system_packages=[entry]))

    def test_the_pin_is_the_distributions_and_not_pips(self) -> None:
        """``apt-get install xvfb==2`` installs nothing and reports success;
        one equals sign is the syntax."""
        spec = decode_image_spec(_spec(system_packages=["xvfb=2:21.1.4-2ubuntu1.7"]))
        assert spec["system_packages"] == ["xvfb=2:21.1.4-2ubuntu1.7"]

    def test_an_empty_layer_is_allowed_because_most_images_have_none(self) -> None:
        """Requiring a package would force one to be invented."""
        assert decode_image_spec(_spec(system_packages=[]))["system_packages"] == []

    @pytest.mark.parametrize(
        "entry",
        [
            "xvfb=1 ; rm -rf /",
            "xvfb=1 && curl x",
            "xvfb=$(id)",
            "xvfb=`id`",
            "xvfb=1|sh",
            "xvfb>=2",
        ],
    )
    def test_a_shell_metacharacter_is_refused(self, entry: str) -> None:
        """These are interpolated into the build script, so a stray separator
        becomes a second command rather than a package that does not exist --
        and the build reports success having installed nothing."""
        with pytest.raises(JSONTypeError, match="interpolated into the build script"):
            _ = decode_image_spec(_spec(system_packages=[entry]))

    def test_a_blank_entry_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be blank"):
            _ = decode_image_spec(_spec(system_packages=["   "]))

    def test_a_non_string_entry_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a string"):
            _ = decode_image_spec(_spec(system_packages=[7]))

    def test_surrounding_whitespace_is_stripped(self) -> None:
        spec = decode_image_spec(_spec(system_packages=["  xvfb=1.2  "]))
        assert spec["system_packages"] == ["xvfb=1.2"]

    def test_the_field_is_required_rather_than_defaulted(self) -> None:
        """Nothing is defaulted in this document: a field absent is a field
        the author did not decide, and deciding it here would put a value into
        an image that no document records."""
        payload = _spec()
        del payload["system_packages"]
        with pytest.raises(JSONTypeError):
            _ = decode_image_spec(payload)


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
