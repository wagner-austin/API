"""RNG state capture, encoding, and restoration for checkpoints."""

from __future__ import annotations

import torch
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    require_int,
    require_list,
)

from model_trainer.core import _test_hooks
from model_trainer.core.types import TorchStateValue


class RngStates:
    """Captured random-number-generator states for exact continuation.

    Attributes:
        torch_state: CPU generator state from ``torch.get_rng_state``.
        cuda_states: Per-device generator states from
            ``torch.cuda.get_rng_state_all``; empty on CPU-only hosts.
        python_state_json: ``random.getstate`` encoded as a JSON string.
    """

    __slots__ = ("cuda_states", "python_state_json", "torch_state")

    torch_state: torch.Tensor
    cuda_states: list[torch.Tensor]
    python_state_json: str

    def __init__(
        self: RngStates,
        *,
        torch_state: torch.Tensor,
        cuda_states: list[torch.Tensor],
        python_state_json: str,
    ) -> None:
        """Initialise captured RNG states.

        Args:
            torch_state: CPU generator state.
            cuda_states: Per-device CUDA generator states.
            python_state_json: Encoded ``random.getstate`` result.
        """
        self.torch_state = torch_state
        self.cuda_states = cuda_states
        self.python_state_json = python_state_json


def _encode_python_rng_state() -> str:
    """Encode the current ``random`` module state as a JSON string.

    The state tuple's shape is validated rather than assumed, because a
    checkpoint that persists a malformed state would corrupt the resumed
    run's sampling silently.

    Returns:
        JSON string carrying version, internal state and gauss carry.

    Raises:
        RuntimeError: If the state tuple does not have the documented
            CPython shape.
    """
    state = _test_hooks.random_getstate()
    if len(state) != 3:
        raise RuntimeError(f"random.getstate returned {len(state)} entries; expected 3")
    version_raw = state[0]
    if isinstance(version_raw, bool) or not isinstance(version_raw, int):
        raise RuntimeError(f"random.getstate version is {type(version_raw).__name__}, not int")
    internal_raw = state[1]
    if not isinstance(internal_raw, tuple):
        raise RuntimeError(
            f"random.getstate internal state is {type(internal_raw).__name__}, not tuple"
        )
    internal_values: list[JSONValue] = []
    for value in internal_raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"random.getstate internal state holds {type(value).__name__}, not int"
            )
        internal_values.append(value)
    gauss_raw = state[2]
    gauss_next: float | None
    if gauss_raw is None:
        gauss_next = None
    elif isinstance(gauss_raw, float):
        gauss_next = gauss_raw
    else:
        raise RuntimeError(
            f"random.getstate gauss value is {type(gauss_raw).__name__}, not float or None"
        )
    payload: JSONObject = {
        "version": version_raw,
        "internal_state": internal_values,
        "gauss_next": gauss_next,
    }
    return dump_json_str(payload)


def _decode_python_rng_state(raw: str) -> tuple[TorchStateValue, ...]:
    """Decode a JSON string back into a ``random.setstate`` tuple.

    Args:
        raw: JSON string produced by :func:`_encode_python_rng_state`.

    Returns:
        The state tuple ``random.setstate`` accepts.

    Raises:
        JSONTypeError: If the string does not carry a valid state.
    """
    loaded = load_json_str(raw)
    if not isinstance(loaded, dict):
        raise JSONTypeError(f"python RNG state must be an object, got {type(loaded).__name__}")
    version = require_int(loaded, "version")
    internal_raw = require_list(loaded, "internal_state")
    internal: list[int] = []
    for index, item in enumerate(internal_raw):
        if isinstance(item, bool) or not isinstance(item, int):
            raise JSONTypeError(
                f"Field 'internal_state[{index}]' must be an integer, got {type(item).__name__}"
            )
        internal.append(item)
    gauss_raw: JSONValue = loaded.get("gauss_next")
    gauss_next: float | None
    if gauss_raw is None:
        gauss_next = None
    elif isinstance(gauss_raw, bool) or not isinstance(gauss_raw, int | float):
        raise JSONTypeError(
            f"Field 'gauss_next' must be a number or null, got {type(gauss_raw).__name__}"
        )
    else:
        gauss_next = float(gauss_raw)
    return (version, tuple(internal), gauss_next)


def capture_rng_states() -> RngStates:
    """Capture torch CPU, CUDA and python RNG states.

    Returns:
        The captured states, ready for checkpoint persistence.
    """
    cuda_states: list[torch.Tensor] = (
        _test_hooks.torch_cuda_get_rng_state_all() if _test_hooks.cuda_is_available() else []
    )
    return RngStates(
        torch_state=torch.get_rng_state(),
        cuda_states=cuda_states,
        python_state_json=_encode_python_rng_state(),
    )


def restore_rng_states(states: RngStates) -> None:
    """Restore torch CPU, CUDA and python RNG states.

    CUDA states are restored only when the checkpoint carries them; a
    checkpoint written on a CPU-only host carries an empty list, and the
    config fingerprint guarantees the resumed run targets the same
    device kind as the original.

    Args:
        states: States captured by :func:`capture_rng_states`.

    Raises:
        JSONTypeError: If the python RNG state string is invalid.
    """
    torch.set_rng_state(states.torch_state)
    if states.cuda_states:
        _test_hooks.torch_cuda_set_rng_state_all(states.cuda_states)
    _test_hooks.random_setstate(_decode_python_rng_state(states.python_state_json))
