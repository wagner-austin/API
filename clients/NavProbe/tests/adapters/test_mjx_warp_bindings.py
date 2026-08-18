"""Drift tests: the MuJoCo-Warp Protocols against the installed vendor API.

Same discipline as the MJX bindings' drift tests: every declared name is
**called** by keyword using the Protocol's own parameter names and driven to a
result only the real function could produce. A vendor rename fails here rather
than mid-measurement.

The Warp surface needs this more than the JAX one, not less. ``step`` and
``render`` both return ``None`` and communicate entirely by mutating their
arguments, so a call that silently did nothing would look identical to a call
that worked — every test here therefore asserts on state that changed.
"""

from __future__ import annotations

from navprobe.adapters.mjx_warp_bindings import (
    POSITION_DTYPE,
    load_mjwarp,
    load_numpy,
)
from navprobe.adapters.mujoco_bindings import MjModelProtocol, load_mujoco
from tests.adapters.models import (
    FREE_JOINT_COORDINATE_COUNT,
    RENDER_PIXEL_COUNT,
    RENDER_RESOLUTION,
    RENDERABLE_BALL_XML,
)

#: Worlds the drift checks allocate.
WORLD_COUNT = 2


def _compiled_model() -> MjModelProtocol:
    """Compile the renderable model.

    Returns:
        The compiled MuJoCo model.
    """
    return load_mujoco().MjModel.from_xml_string(xml=RENDERABLE_BALL_XML)


class TestDeclaredKeywordNames:
    """Each vendor function accepts the parameter names its Protocol declares."""

    def test_put_model_takes_mjm(self) -> None:
        """``mujoco_warp.put_model(mjm=...)`` places the model.

        The placed model's coordinate count is asserted rather than its mere
        existence: it proves the compiled model reached the device rather than
        the call returning some other object.
        """
        placed = load_mjwarp().put_model(mjm=_compiled_model())
        assert placed.nq == FREE_JOINT_COORDINATE_COUNT

    def test_make_data_takes_mjm_and_nworld(self) -> None:
        """``mujoco_warp.make_data(mjm=..., nworld=...)`` allocates per world."""
        model = load_mujoco().MjModel.from_xml_string(xml=RENDERABLE_BALL_XML)
        data = load_mjwarp().make_data(mjm=model, nworld=WORLD_COUNT)
        assert len(data.qpos.numpy().tolist()) == WORLD_COUNT

    def test_make_data_takes_the_capacity_keywords(self) -> None:
        """The three capacity arguments are accepted by their declared names.

        Contact-rich scenes overflow the vendor's default sizing, and an
        overflow prints a warning and continues rather than raising — so if one
        of these names drifted, the probe would quietly go back to measuring a
        truncated solve instead of failing.
        """
        model = load_mujoco().MjModel.from_xml_string(xml=RENDERABLE_BALL_XML)
        data = load_mjwarp().make_data(
            mjm=model, nworld=WORLD_COUNT, njmax=512, nconmax=512, naconmax=512
        )
        assert len(data.qpos.numpy().tolist()) == WORLD_COUNT

    def test_allocated_state_has_one_row_of_coordinates_per_world(self) -> None:
        """The batch axis leads, which is what the adapter's flattening assumes."""
        model = load_mujoco().MjModel.from_xml_string(xml=RENDERABLE_BALL_XML)
        data = load_mjwarp().make_data(mjm=model, nworld=WORLD_COUNT)
        assert len(data.qpos.numpy().tolist()[0]) == FREE_JOINT_COORDINATE_COUNT

    def test_assign_writes_through_to_the_device(self) -> None:
        """``assign`` is a real write, not a no-op on a host copy.

        The adapter's whole seeding path depends on this: an ``assign`` that
        silently failed would leave every world at the model's default state,
        every world identical, and the batch carrying no information — while
        still reporting perfect determinism.
        """
        model = load_mujoco().MjModel.from_xml_string(xml=RENDERABLE_BALL_XML)
        data = load_mjwarp().make_data(mjm=model, nworld=WORLD_COUNT)
        rows = data.qpos.numpy().tolist()
        for index, row in enumerate(rows):
            row[0] = 0.25 + index
        data.qpos.assign(load_numpy().array(rows, dtype=POSITION_DTYPE))
        assert [row[0] for row in data.qpos.numpy().tolist()] == [0.25, 1.25]

    def test_step_mutates_the_state_in_place(self) -> None:
        """``mujoco_warp.step(m=..., d=...)`` advances by mutation.

        Asserted by the state having changed, because the call returns nothing
        and a step that did not run would be indistinguishable otherwise.
        """
        mjwarp = load_mjwarp()
        step = mjwarp.step
        model = _compiled_model()
        placed = mjwarp.put_model(mjm=model)
        data = mjwarp.make_data(mjm=model, nworld=WORLD_COUNT)
        before = data.qpos.numpy().tolist()
        step(m=placed, d=data)
        assert data.qpos.numpy().tolist() != before

    def test_create_render_context_takes_its_declared_keywords(self) -> None:
        """Every keyword the adapter passes is accepted, and buffers are sized."""
        mjwarp = load_mjwarp()
        model = _compiled_model()
        context = mjwarp.create_render_context(
            mjm=model,
            nworld=WORLD_COUNT,
            cam_res=RENDER_RESOLUTION,
            render_rgb=True,
            render_depth=True,
        )
        assert len(context.depth_data.numpy().tolist()) == WORLD_COUNT

    def test_render_writes_into_the_context(self) -> None:
        """``mujoco_warp.render(m=..., d=..., rc=...)`` fills the pixel buffers.

        Like ``step`` it returns nothing, so the assertion is that the buffer
        stopped being uniform: a render that never ran would leave the
        allocation at its initial constant.
        """
        mjwarp = load_mjwarp()
        step = mjwarp.step
        model = _compiled_model()
        placed = mjwarp.put_model(mjm=model)
        data = mjwarp.make_data(mjm=model, nworld=WORLD_COUNT)
        context = mjwarp.create_render_context(
            mjm=model,
            nworld=WORLD_COUNT,
            cam_res=RENDER_RESOLUTION,
            render_rgb=True,
            render_depth=True,
        )
        step(m=placed, d=data)
        mjwarp.render(m=placed, d=data, rc=context)
        assert len(set(context.depth_data.numpy().tolist()[0])) > 1

    def test_the_pixel_buffers_are_sized_by_the_camera_resolution(self) -> None:
        """Buffer width follows ``cam_res``, which pins the observation length."""
        mjwarp = load_mjwarp()
        model = _compiled_model()
        context = mjwarp.create_render_context(
            mjm=model,
            nworld=WORLD_COUNT,
            cam_res=RENDER_RESOLUTION,
            render_rgb=True,
            render_depth=True,
        )
        assert len(context.rgb_data.numpy().tolist()[0]) == RENDER_PIXEL_COUNT
