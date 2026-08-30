"""The control-arm table, shared by every command that can apply the controls.

It lived inside ``cli/probe_trace`` until 2026-08-29, when the isolated GEMM
probe needed the same four arms. These tests came with it: a second spelling
of the table would let the two drift, and then "the trace ran arm X" and "the
GEMM probe ran arm X" could name two different postures while reading
identically in both records.
"""

from __future__ import annotations

import pytest

from model_trainer.core.services.model.control_arms import (
    CONTROL_ARMS,
    CONTROLS_FLAG,
    require_control_arm,
)


class TestTheControlArms:
    """The flag exists so an instrument can reach the treated condition.

    Split-K had an environment escape and the attention pin has none -- it is
    four ``torch.backends.cuda`` calls -- so before this flag a measurement
    could only ever observe attention untreated. An instrument that cannot
    reach a condition cannot measure it.
    """

    def test_every_arm_names_a_distinct_posture(self) -> None:
        # Four arms because the two controls are disjoint: split-K governs
        # cuBLASLt matmuls, the math pin governs attention. The single-control
        # arms are what make attribution a run rather than a code change.
        assert CONTROL_ARMS == {
            "none": (False, False),
            "split-k": (True, False),
            "attention": (False, True),
            "both": (True, True),
        }

    def test_it_resolves_each_arm(self) -> None:
        assert require_control_arm("none") == (False, False)
        assert require_control_arm("both") == (True, True)

    def test_it_resolves_the_single_control_arms(self) -> None:
        assert require_control_arm("split-k") == (True, False)
        assert require_control_arm("attention") == (False, True)

    def test_an_unknown_arm_is_refused_by_name(self) -> None:
        # Refused rather than defaulted: a measurement whose arm was guessed
        # is one whose record names a condition it may not have run under.
        with pytest.raises(ValueError, match="must be one of attention, both, none, split-k"):
            require_control_arm("splitk")

    def test_the_refusal_names_the_flag(self) -> None:
        # The message is read on a cluster, detached from the source.
        with pytest.raises(ValueError, match="--controls"):
            require_control_arm("")

    def test_the_flag_is_declared_here_rather_than_per_command(self) -> None:
        # Two commands spell it; one constant means they cannot disagree.
        assert CONTROLS_FLAG == "--controls"
