"""Apply or revert the tactile alias patch that unblocks Warp deterministic mode.

MuJoCo-Warp 3.11.0's ``_sensor_tactile`` writes channel 0 of ``sensordata_out``
with ``wp.atomic_max`` and channels 1-2 with ``wp.atomic_add``. Warp's
deterministic execution mode allows one reduction family per target array, so
the module is rejected under ``RUN_TO_RUN`` and ``GPU_TO_GPU`` for every model
(the kernel compiles with the rest of the module, tactile sensors or not).

The fix is the alias-binding shape from the withdrawn upstream PR
google-deepmind/mujoco_warp#1591 (NY-WaKeUp, 2026-08-18, closed unmerged over
CLA): bind the same ``d.sensordata`` array to a second kernel parameter and
route the ``atomic_max`` through it. Each binding then carries one reduction
family; the channels write disjoint index ranges (channel k at
``adr + k*dim + vertid`` with ``vertid < dim``), so aliasing is semantically
inert. Measured 2026-08-18 on the NavProbe venv (warp-lang 1.16.0 /
mujoco-warp 3.11.0): with this patch BOTH deterministic modes compile every
module cold and step the touching-row scene on the Warp CPU device; without it
both fail at sensor.py:2307 with WarpCodegenError. See the wiki page
``tactile-alias-patch-clears-warp-deterministic-compile``.

Usage:
    python scripts/apply_tactile_alias_patch.py apply  [path-to-sensor.py]
    python scripts/apply_tactile_alias_patch.py revert [path-to-sensor.py]

Default path is this repo's venv:
``.venv/Lib/site-packages/mujoco_warp/_src/sensor.py``.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from navprobe import NavProbeError
from scripts import _test_hooks
from scripts.arguments import ScriptArgumentError

#: The two things this script can do to the installed vendor file.
ACTIONS = ("apply", "revert")

USAGE = "usage: apply_tactile_alias_patch (apply|revert) [path-to-sensor.py]"

_SIG_OLD = """  # Data out:
  sensordata_out: wp.array2d[float],
):
  worldid, taxelid = wp.tid()

  sensor_id = taxel_sensorid[taxelid]"""

_SIG_NEW = """  # Data out:
  sensordata_out: wp.array2d[float],
  # Aliases sensordata_out. Bound separately so the atomic_max on the normal
  # channel and the atomic_add on the shear channels are distinct reduction
  # targets: deterministic mode allows only one reduction family per target,
  # and the channels are disjoint (channel 0 vs channels 1 and 2).
  sensordata_max_out: wp.array2d[float],
):
  worldid, taxelid = wp.tid()

  sensor_id = taxel_sensorid[taxelid]"""

# Split across source lines by implicit concatenation only. The values must
# match the vendor file byte for byte, so the strings themselves are unwrapped.
_MAX_OLD = (
    "    wp.atomic_max(sensordata_out, worldid, "
    "sensor_adr[sensor_id] + 0 * dim + vertid, forceT[0])"
)
_MAX_NEW = (
    "    wp.atomic_max(sensordata_max_out, worldid, "
    "sensor_adr[sensor_id] + 0 * dim + vertid, forceT[0])"
)

_LAUNCH_OLD = """      weld_geom_count,
      weld_geom_list,
    ],
    outputs=[
      d.sensordata,
    ],
  )

  sensor_contact_nmatch"""

_LAUNCH_NEW = """      weld_geom_count,
      weld_geom_list,
    ],
    outputs=[
      d.sensordata,
      d.sensordata,
    ],
  )

  sensor_contact_nmatch"""

_DEFAULT = ".venv/Lib/site-packages/mujoco_warp/_src/sensor.py"


class PatchSiteError(NavProbeError):
    """A patch site did not appear exactly once in the target file.

    Anything other than one occurrence means the installed vendor file is not
    the revision this patch was written against, and swapping anyway would
    either miss the site or corrupt a second one.

    Args:
        code: Stable identifier in the ``NP-PATCH-<NNN>`` range.
        message: Which site failed, and how many times it was found.
    """


def _swap(text: str, pairs: list[tuple[str, str]]) -> str:
    """Apply each replacement exactly once.

    Args:
        text: The file's contents, newline-normalised.
        pairs: ``(old, new)`` site replacements, applied in order.

    Returns:
        The rewritten contents.

    Raises:
        PatchSiteError: When a site is absent or appears more than once.
    """
    for old, new in pairs:
        found = text.count(old)
        if found != 1:
            raise PatchSiteError(
                "NP-PATCH-001",
                f"expected exactly one occurrence, found {found}: {old[:80]!r}",
            )
        text = text.replace(old, new)
    return text


def main(argv: Sequence[str] | None = None) -> int:
    """Apply or revert the alias patch.

    Args:
        argv: Arguments excluding the program name. ``None`` reads the process
            arguments.

    Returns:
        ``0`` on success.

    Raises:
        ScriptArgumentError: When no action is given, or it is neither
            ``apply`` nor ``revert``.
        PatchSiteError: When the target file is not the revision this patch was
            written against.
    """
    args = list(sys.argv[1:]) if argv is None else list(argv)
    if not args or args[0] not in ACTIONS:
        raise ScriptArgumentError(
            "NP-ARGS-007", f"{USAGE} -- expected one of {ACTIONS}, got {args[:1]}"
        )
    action = args[0]
    path = pathlib.Path(args[1] if len(args) > 1 else _DEFAULT)
    raw = path.read_bytes()
    # Preserve the file's own line-ending convention: normalize to LF for the
    # swap, then restore CRLF iff the file used it. A silent EOL rewrite would
    # change Warp's module hash (cache key) for every kernel in the module.
    had_crlf = b"\r\n" in raw
    text = raw.replace(b"\r\n", b"\n").decode("utf-8")
    forward = [(_SIG_OLD, _SIG_NEW), (_MAX_OLD, _MAX_NEW), (_LAUNCH_OLD, _LAUNCH_NEW)]
    pairs = forward if action == "apply" else [(n, o) for (o, n) in forward]
    out = _swap(text, pairs).encode("utf-8")
    if had_crlf:
        out = out.replace(b"\n", b"\r\n")
    path.write_bytes(out)
    _test_hooks.write_out(f"{action}: OK ({path})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
