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
        "system_packages": [],
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

        The environment known-answer probe was added by another session on
        2026-08-25. It arrived here as a FAILURE of this test rather than as
        a silent widening, which is the exactness earning its keep: the spec
        grew, and someone had to look.

        THE LOOK WAS SKIPPED AFTER THAT, AND THIS LIST WENT STALE. The spec
        and this assertion last agreed at six symbols in ``108e3ef4``. They
        parted at ``d6cd17d7`` -- image v13, the forward trace -- and NINE
        further spec-growing commits followed without this list moving:
        v13, v14, v15, v16, v17, v18, v19, v20, v21, ending at twenty-five
        symbols against six asserted. The additions are the trace, SDPA,
        forward-cost, training-step and legacy-GEMM probes, plus the two
        ``environment_record`` captures ``c0ce20b7`` baked in so a run
        fingerprint could carry its host and packages.

        Every one of those nine commits left this assertion failing and was
        committed anyway, so ``make check`` in this package was red on
        ``main`` across nine changes and the signal this test exists to give
        was being stepped over rather than read. The list below is the whole
        of ``specs/abl-image.json``; the ritual only works if the failure is
        answered in the commit that causes it.

        IT WENT STALE AGAIN ON THE VERY NEXT SPEC COMMIT. ``b7da5cda`` -- v22,
        the determinism controls -- added three symbols
        (``CUBLASLT_WORKSPACE_ENV_VAR``, ``remove_cublaslt_split_k``,
        ``restrict_attention_to_math``) and did not touch this list, one
        commit after the paragraph above was written to stop exactly that.
        So the confession is not the fix, and neither is the tenth repetition
        of it: what this failure keeps proving is that a list transcribed by
        hand into a test drifts from the artifact it transcribes whenever
        those are edited by different people at different times. The
        assertion is worth keeping because the LOOK is worth forcing, but
        anyone tempted to explain the next recurrence should reach for the
        generator instead.

        WHY THE GENERATOR IS NOT REACHABLE FROM HERE, checked on 2026-08-30
        rather than assumed. The generator that would end this would derive
        the list from the code -- import each module and resolve each
        attribute -- and this package cannot: ``hpc3`` is its own poetry
        project and ``import model_trainer`` fails in its venv. Deriving the
        list from the spec instead would assert the spec against itself and
        check nothing. So the transcription is not laziness, it is the only
        thing available at this layer.

        WHERE THE REAL RESOLVE-CHECK LIVES. ``selfcheck.py``, rendered into
        the build directory and run INSIDE the image, which is where every
        module is installed. That is the check with teeth; this list is a
        local front-run of it, and its value is catching a moved symbol
        before a build job is spent rather than after. v24 spent one on a
        stale smoke, which is the same class of failure one layer over.

        Eleventh recurrence, 2026-08-30: ``require_control_arm`` MOVED --
        from ``cli.probe_trace`` to ``core.services.model.control_arms``,
        because the isolated GEMM probe needed the same four arms -- and
        three symbols were added for the kernel arms. A move is the case the
        confession above never covered: the transcribed list stayed
        internally consistent and pointed at an attribute that no longer
        existed, so the spec would have decoded, rendered, and failed in the
        container.
        """
        spec = decode_image_spec(load_json_str(_COMMITTED_SPEC.read_text(encoding="utf-8")))
        symbols = sorted(
            (check["module"], check["attribute"]) for check in spec["required_symbols"]
        )
        assert symbols == [
            ("model_trainer.cli.forward_benchmark", "measure_row"),
            ("model_trainer.cli.known_answer_probe", "probe_run_record"),
            ("model_trainer.cli.legacy_gemm_probe", "legacy_run_record"),
            ("model_trainer.cli.probe_trace", "trace_run_record"),
            ("model_trainer.cli.probe_trace", "workspace_observation"),
            ("model_trainer.cli.probe_trace_report", "report_lines"),
            ("model_trainer.cli.score_baseline", "main"),
            ("model_trainer.cli.score_baseline", "score_with_outcomes"),
            ("model_trainer.cli.sdpa_benchmark", "benchmark_run_record"),
            ("model_trainer.cli.sdpa_probe", "selected_backend"),
            ("model_trainer.cli.train_benchmark", "train_run_record"),
            ("model_trainer.cli.train_benchmark_report", "report_lines"),
            ("model_trainer.cluster.preflight", "check_corpus_certified"),
            (
                "model_trainer.core.services.model.control_arms",
                "require_control_arm",
            ),
            (
                "model_trainer.core.services.model.deterministic_gemm",
                "BLOCK_ARMS",
            ),
            (
                "model_trainer.core.services.model.deterministic_gemm",
                "blocked_matmul",
            ),
            (
                "model_trainer.core.services.model.deterministic_gemm",
                "gemm_by_arm",
            ),
            (
                "model_trainer.core.services.model.deterministic_gemm",
                "matmul_by_arm",
            ),
            (
                "model_trainer.core.services.model.deterministic_gemm",
                "rank1_addmm",
            ),
            ("model_trainer.core.services.model.forward_cost", "release_row"),
            ("model_trainer.core.services.model.forward_trace", "traced_forward"),
            ("model_trainer.core.services.model.gemm_shapes", "GEMM_BOUNDARY"),
            (
                "model_trainer.core.services.model.kernel_arm_modules",
                "ArmConv1D",
            ),
            (
                "model_trainer.core.services.model.kernel_arm_modules",
                "apply_kernel_arm_to_model",
            ),
            (
                "model_trainer.core.services.model.kernel_arm_modules",
                "use_kernel_arm",
            ),
            (
                "model_trainer.core.services.model.known_answer_probe",
                "probe_forward_loss",
            ),
            ("model_trainer.core.services.model.legacy_gemm_probe", "arm_outputs"),
            ("model_trainer.core.services.model.sdpa_probe", "probe_sdpa"),
            ("model_trainer.core.services.model.sdpa_timing", "backend_context"),
            ("model_trainer.core.services.model.sdpa_timing", "time_sdpa"),
            ("model_trainer.core.services.model.train_cost", "run_train_step"),
            ("model_trainer.core.services.model.train_cost", "train_step_setup"),
            (
                "model_trainer.core.services.training.base_trainer_checkpoints",
                "_TrainerCheckpoints",
            ),
            ("platform_core.determinism_env", "CUBLASLT_WORKSPACE_ENV_VAR"),
            ("platform_core.environment_record", "capture_host_record"),
            ("platform_core.environment_record", "capture_package_versions"),
            ("platform_ml.determinism", "remove_cublaslt_split_k"),
            ("platform_ml.determinism", "restrict_attention_to_math"),
        ]

    def test_its_environment_survives_the_cluster_bind_mounts(self) -> None:
        spec = decode_image_spec(load_json_str(_COMMITTED_SPEC.read_text(encoding="utf-8")))
        assert spec["env_prefix"] == "/opt/env"
