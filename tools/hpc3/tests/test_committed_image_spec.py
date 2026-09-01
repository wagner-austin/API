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

        Twelfth, 2026-08-31: four symbols for the v29 image -- the train-step
        probe (its plan table, its identity check, its record builder) and
        ``probed_shapes_hook``, which the batched-digest smokes install to
        keep a CPU smoke from walking ninety-three shapes. Each names
        something v28 did not carry, which is the property that makes a
        required symbol detect a stale wheel at all.

        Thirteenth, 2026-09-01: two symbols for v30's owned-backward arm --
        ``OWNED_ARM`` on the arm table and ``OwnedAddmm``, the autograd
        Function whose backward is the entire point. v29 carries neither.
        """
        spec = decode_image_spec(load_json_str(_COMMITTED_SPEC.read_text(encoding="utf-8")))
        symbols = sorted(
            (check["module"], check["attribute"]) for check in spec["required_symbols"]
        )
        assert symbols == [
            ("model_trainer.cli._test_hooks", "probed_shapes_hook"),
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
            ("model_trainer.cli.train_step_probe", "train_step_run_record"),
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
                "OWNED_ARM",
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
            ("model_trainer.core.services.model.owned_backward", "OwnedAddmm"),
            ("model_trainer.core.services.model.sdpa_probe", "probe_sdpa"),
            ("model_trainer.core.services.model.sdpa_timing", "backend_context"),
            ("model_trainer.core.services.model.sdpa_timing", "time_sdpa"),
            ("model_trainer.core.services.model.train_cost", "run_train_step"),
            ("model_trainer.core.services.model.train_cost", "train_step_setup"),
            (
                "model_trainer.core.services.model.train_step_plan",
                "TRAIN_STEP_RUNGS",
            ),
            (
                "model_trainer.core.services.model.train_step_probe",
                "train_step_identity",
            ),
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
