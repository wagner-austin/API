"""MJCF models the adapter tests drive.

A sphere with a free joint on a plane. Small enough that compiling and tracing
it costs little, and physical enough that the contact solver actually runs —
a model with no contacts would exercise the integrator and skip the code path
whose reproducibility is the open question.
"""

from __future__ import annotations

#: Generalised coordinates a free joint carries: three translational, four
#: quaternion.
FREE_JOINT_COORDINATE_COUNT = 7

#: A single sphere falling onto a plane.
FALLING_BALL_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 0.5">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.1" density="1000"/>
    </body>
  </worldbody>
</mujoco>
"""

#: Camera resolution the render tests use, as ``(width, height)``.
RENDER_RESOLUTION = (32, 32)

#: Pixels one camera produces per world at :data:`RENDER_RESOLUTION`.
RENDER_PIXEL_COUNT = RENDER_RESOLUTION[0] * RENDER_RESOLUTION[1]

#: The same scene with what rendering requires: a light to shade by, coloured
#: geoms to distinguish, and a camera to render from. Kept separate from
#: :data:`FALLING_BALL_XML` so the physics measurements are not silently taken
#: against a different model than the ones already recorded.
RENDERABLE_BALL_XML = """
<mujoco>
  <option timestep="0.005"/>
  <visual><global offwidth="32" offheight="32"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="ball" pos="0 0 0.5">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.1" density="1000" rgba="1 0 0 1"/>
    </body>
    <camera name="cam0" pos="0 -2 1" xyaxes="1 0 0 0 0.5 1"/>
  </worldbody>
</mujoco>
"""

__all__ = [
    "FALLING_BALL_XML",
    "FREE_JOINT_COORDINATE_COUNT",
    "RENDERABLE_BALL_XML",
    "RENDER_PIXEL_COUNT",
    "RENDER_RESOLUTION",
]
