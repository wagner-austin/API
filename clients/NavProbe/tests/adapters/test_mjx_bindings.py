"""Drift tests: the declared Protocols against the installed vendor API.

A Protocol is a claim about somebody else's code. Left unchecked it is a claim
that was true when it was written, and the failure mode is silent: the adapter
keeps type-checking against a signature the vendor has since changed, and the
first sign of trouble is a measurement that is wrong rather than a build that is
red.

So every declared name is **called**, by keyword, using the Protocol's own
parameter names, and driven to a result that could only be produced by the real
function. That is stronger than reading a signature: it checks presence, spelling
and behaviour in one assertion, and a vendor rename fails here rather than
mid-measurement. It is also the only form available — the monorepo bans
:mod:`inspect`, and reflecting over annotations would reintroduce ``Any`` into a
package that forbids it.
"""

from __future__ import annotations

from navprobe.adapters.jax_bindings import FlatArrayProtocol, load_jax_numpy
from navprobe.adapters.mjx_bindings import (
    MjxDataProtocol,
    StateBuilderProtocol,
    load_jax_state_transforms,
    load_jax_step_transforms,
    load_mjx,
)
from navprobe.adapters.mujoco_bindings import load_mujoco
from tests.adapters.models import FALLING_BALL_XML, FREE_JOINT_COORDINATE_COUNT

#: Worlds the batched drift checks build.
WORLD_COUNT = 2


def _state_builder(base: MjxDataProtocol) -> StateBuilderProtocol:
    """Build the single-world state builder the adapter vmaps.

    Written as an annotated closure rather than a lambda so its parameter keeps
    a declared type; an unannotated lambda would reintroduce ``Any`` at exactly
    the boundary these tests exist to keep typed.

    Args:
        base: State in the model's initial configuration.

    Returns:
        A builder producing one world's state from that world's positions.
    """

    def build_one(qpos: FlatArrayProtocol) -> MjxDataProtocol:
        """Build one world's state.

        Args:
            qpos: That world's generalised positions.

        Returns:
            The state for that world.
        """
        return base.replace(qpos=qpos)

    return build_one


class TestDeclaredKeywordNames:
    """Each vendor function accepts the parameter names its Protocol declares.

    Every call passes arguments by keyword using the Protocol's own names, so a
    vendor rename makes the call raise. That is the drift signal.
    """

    def test_from_xml_string_takes_xml(self) -> None:
        """``MjModel.from_xml_string(xml=...)`` compiles the model."""
        model = load_mujoco().MjModel.from_xml_string(xml=FALLING_BALL_XML)
        assert model.nq == FREE_JOINT_COORDINATE_COUNT

    def test_put_model_takes_m(self) -> None:
        """``mjx.put_model(m=...)`` places the model on the device."""
        compiled = load_mujoco().MjModel.from_xml_string(xml=FALLING_BALL_XML)
        assert load_mjx().put_model(m=compiled).nq == FREE_JOINT_COORDINATE_COUNT

    def test_make_data_takes_m(self) -> None:
        """``mjx.make_data(m=...)`` allocates state for the model."""
        mjx = load_mjx()
        model = mjx.put_model(m=load_mujoco().MjModel.from_xml_string(xml=FALLING_BALL_XML))
        assert len(mjx.make_data(m=model).qpos.tolist()) == FREE_JOINT_COORDINATE_COUNT

    def test_step_takes_m_and_d(self) -> None:
        """``mjx.step(m=..., d=...)`` advances the state.

        The returned positions differ from the initial ones, so this asserts a
        step was taken rather than that the call merely returned. The function
        is bound to a local name first, matching the adapter, which holds the
        binding rather than reaching through the module on every step.
        """
        mjx = load_mjx()
        step = mjx.step
        model = mjx.put_model(m=load_mujoco().MjModel.from_xml_string(xml=FALLING_BALL_XML))
        data = mjx.make_data(m=model)
        assert step(m=model, d=data).qpos.tolist() != data.qpos.tolist()

    def test_asarray_takes_a(self) -> None:
        """``jax.numpy.asarray(a=...)`` converts host values to a device array."""
        rows = [[1.0, 2.0], [3.0, 4.0]]
        assert load_jax_numpy().asarray(a=rows).tolist() == rows

    def test_replace_takes_qpos_and_carries_the_values_through(self) -> None:
        """``Data.replace(qpos=...)`` returns a copy carrying those positions.

        Exercised through the vmapped builder because that is how the adapter
        calls it, and because MJX's data type is a pytree whose leaves must all
        gain a batch axis together.
        """
        mjx = load_mjx()
        model = mjx.put_model(m=load_mujoco().MjModel.from_xml_string(xml=FALLING_BALL_XML))
        base = mjx.make_data(m=model)
        rows = [[0.25] * FREE_JOINT_COORDINATE_COUNT] * WORLD_COUNT
        build = load_jax_state_transforms().vmap(_state_builder(base))
        assert build(load_jax_numpy().asarray(a=rows)).qpos.tolist() == rows

    def test_vmap_takes_in_axes(self) -> None:
        """``jax.vmap(fun, in_axes=...)`` batches the step over its second argument.

        Driven to a result rather than merely constructed: a batched step of two
        worlds returns two rows, which is what ``in_axes`` was asked to produce.
        """
        mjx = load_mjx()
        model = mjx.put_model(m=load_mujoco().MjModel.from_xml_string(xml=FALLING_BALL_XML))
        base = mjx.make_data(m=model)
        rows = [base.qpos.tolist()] * WORLD_COUNT
        build = load_jax_state_transforms().vmap(_state_builder(base))
        state = build(load_jax_numpy().asarray(a=rows))
        batched = load_jax_step_transforms().vmap(mjx.step, in_axes=(None, 0))
        assert len(batched(model, state).qpos.tolist()) == WORLD_COUNT


class TestJitPreservesResults:
    """``jax.jit`` is declared as returning the same function, and does.

    The adapter measures the compiled path. If compiling changed the result,
    every verdict this instrument produced would describe a different
    computation from the one the caller believes was measured — so this is a
    claim about the vendor worth checking rather than assuming.
    """

    def test_compiled_and_eager_batched_steps_agree(self) -> None:
        """One batched step gives the same positions compiled or not."""
        mjx = load_mjx()
        transforms = load_jax_step_transforms()
        model = mjx.put_model(m=load_mujoco().MjModel.from_xml_string(xml=FALLING_BALL_XML))
        base = mjx.make_data(m=model)
        rows = [base.qpos.tolist()] * WORLD_COUNT
        build = load_jax_state_transforms().vmap(_state_builder(base))
        state = build(load_jax_numpy().asarray(a=rows))
        eager = transforms.vmap(mjx.step, in_axes=(None, 0))
        compiled = transforms.jit(eager)
        assert compiled(model, state).qpos.tolist() == eager(model, state).qpos.tolist()
