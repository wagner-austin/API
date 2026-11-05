"""The small control policy, its weights as a file, and its weights as text.

A two-input, three-action multilayer perceptron, ``2 -> hidden -> hidden ->
3`` with tanh between layers: the same shape of network as the post's
Experiment 4 policy, small enough that every weight fits in a reader's
context as text, which is the condition that experiment ran under.

TWO RENDERINGS OF THE SAME WEIGHTS, FOR TWO READERS. The file
(:func:`encode_policy_weights`) is for the program: every float32 value
written as the shortest decimal that reads back to the same bits, so a policy
saved by one job and loaded by another is the same policy, which the
weights digest then confirms. The text (:func:`render_weights_text`) is for
the language model: rows of three-decimal numbers under the parameter names,
the form the post's prompts show.
"""

from __future__ import annotations

import math

import torch
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    narrow_json_to_float,
    narrow_json_to_int,
    require_dict,
    require_int,
    require_list,
)
from typing_extensions import TypedDict

#: The policy's parameter names, in the order the text rendering lists them.
POLICY_PARAMETERS: tuple[str, ...] = (
    "hidden_in.weight",
    "hidden_in.bias",
    "hidden_mid.weight",
    "hidden_mid.bias",
    "action_out.weight",
    "action_out.bias",
)


def _uniform(shape: tuple[int, ...], bound: float) -> torch.nn.Parameter:
    """Draw a parameter uniformly from ``[-bound, bound)``, off the global generator.

    Args:
        shape: The parameter's shape.
        bound: The half-width.

    Returns:
        The parameter, trainable.
    """
    return torch.nn.Parameter((torch.rand(shape) * 2.0 - 1.0) * bound)


class ControlPolicy:
    """``2 -> hidden -> hidden -> 3`` with tanh, scoring the three actions.

    THE SIX PARAMETERS ARE HELD BY NAME rather than inside ``torch.nn.Linear``
    modules. A weight program addresses parameters by name, and the shared
    snapshot and digest checks read a sequence of named parameters; holding
    them here gives exactly that, and the layer arithmetic below is the one
    line ``tanh(x @ W.T + b)`` that ``Linear`` would have run. Each layer is
    initialised as ``Linear`` initialises it, every entry uniform in
    ``[-1/sqrt(fan_in), 1/sqrt(fan_in))``.

    Attributes:
        hidden: Units per hidden layer.
        parameters_by_name: The six parameters, keyed as
            :data:`POLICY_PARAMETERS` names them, torch's ``[out, in]``
            orientation for each weight.
    """

    hidden: int
    parameters_by_name: dict[str, torch.nn.Parameter]

    def __init__(self, hidden: int) -> None:
        """Build the parameters, drawing from torch's global generator.

        Args:
            hidden: Units per hidden layer.
        """
        self.hidden = hidden
        self.parameters_by_name = {}
        for layer, (fan_in, fan_out) in (
            ("hidden_in", (2, hidden)),
            ("hidden_mid", (hidden, hidden)),
            ("action_out", (hidden, 3)),
        ):
            bound = 1.0 / math.sqrt(float(fan_in))
            self.parameters_by_name[f"{layer}.weight"] = _uniform((fan_out, fan_in), bound)
            self.parameters_by_name[f"{layer}.bias"] = _uniform((fan_out,), bound)

    def named_parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        """Every parameter with its name, in :data:`POLICY_PARAMETERS` order.

        Returns:
            ``(name, parameter)`` pairs.
        """
        return tuple((name, self.parameters_by_name[name]) for name in POLICY_PARAMETERS)

    def get_parameter(self, target: str) -> torch.nn.Parameter:
        """Return one parameter by name.

        Args:
            target: A name from :data:`POLICY_PARAMETERS`.

        Returns:
            The parameter, writable.

        Raises:
            KeyError: If the name is not one of the six; the shared
                :func:`~model_trainer.core.services.model.editing.sites.require_parameter`
                checks the inventory first and refuses by error code.
        """
        return self.parameters_by_name[target]

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Score each action for each observation.

        Args:
            observation: ``(episodes, 2)`` scaled position and velocity.

        Returns:
            ``(episodes, 3)`` action scores.
        """
        weights = self.parameters_by_name
        first = torch.tanh(observation @ weights["hidden_in.weight"].T + weights["hidden_in.bias"])
        second = torch.tanh(first @ weights["hidden_mid.weight"].T + weights["hidden_mid.bias"])
        return second @ weights["action_out.weight"].T + weights["action_out.bias"]

    def greedy(self, observation: torch.Tensor) -> torch.Tensor:
        """Pick the highest-scoring action, which is how the policy is evaluated.

        Args:
            observation: ``(episodes, 2)`` scaled position and velocity.

        Returns:
            One action index per episode.
        """
        return torch.argmax(self.forward(observation), dim=1)


class ParameterValues(TypedDict):
    """One parameter's shape and its values in row-major order.

    Attributes:
        shape: The tensor's dimensions.
        values: Every element, row-major.
    """

    shape: tuple[int, ...]
    values: tuple[float, ...]


class PolicyWeights(TypedDict):
    """A whole policy's weights, as the file holds them.

    Attributes:
        hidden: Units per hidden layer.
        parameters: One entry per name in :data:`POLICY_PARAMETERS`.
    """

    hidden: int
    parameters: dict[str, ParameterValues]


def policy_weights(policy: ControlPolicy) -> PolicyWeights:
    """Read a policy's weights out into the file form.

    Args:
        policy: The policy.

    Returns:
        Its weights.
    """
    parameters: dict[str, ParameterValues] = {}
    for name, parameter in policy.named_parameters():
        values = parameter.detach().to("cpu").contiguous().flatten()
        parameters[name] = ParameterValues(
            shape=tuple(int(size) for size in parameter.shape),
            values=tuple(float(values[index].item()) for index in range(values.numel())),
        )
    return PolicyWeights(hidden=policy.hidden, parameters=parameters)


def load_policy(weights: PolicyWeights) -> ControlPolicy:
    """Build a policy holding exactly the given weights.

    Args:
        weights: The weights.

    Returns:
        The policy.

    Raises:
        JSONTypeError: If a parameter's shape is not the one a policy of this
            width has, or its values do not fill that shape.
    """
    policy = ControlPolicy(weights["hidden"])
    for name, parameter in policy.named_parameters():
        entry = weights["parameters"][name]
        expected = tuple(int(size) for size in parameter.shape)
        if entry["shape"] != expected or len(entry["values"]) != parameter.numel():
            raise JSONTypeError(
                f"parameter '{name}' of a {weights['hidden']}-unit policy has shape "
                f"{expected}; the file gives {entry['shape']} with {len(entry['values'])} values"
            )
        filled = torch.empty(parameter.numel(), dtype=parameter.dtype)
        for index, value in enumerate(entry["values"]):
            filled[index] = value
        with torch.no_grad():
            parameter.copy_(filled.reshape(parameter.shape))
    return policy


def encode_policy_weights(weights: PolicyWeights) -> JSONObject:
    """Encode a policy's weights.

    Args:
        weights: The weights.

    Returns:
        The JSON-serializable form.
    """
    parameters: JSONObject = {}
    for name in POLICY_PARAMETERS:
        entry = weights["parameters"][name]
        shape: list[JSONValue] = list(entry["shape"])
        values: list[JSONValue] = list(entry["values"])
        parameters[name] = {"shape": shape, "values": values}
    return {"hidden": weights["hidden"], "parameters": parameters}


def decode_policy_weights(obj: JSONObject) -> PolicyWeights:
    """Decode a policy's weights.

    Args:
        obj: The object to decode.

    Returns:
        The weights; whether they fit a policy is checked by
        :func:`load_policy`, which has the policy to fit them to.

    Raises:
        JSONTypeError: If a field is missing or mistyped, the width is not
            positive, or the parameter names are not exactly
            :data:`POLICY_PARAMETERS`.
    """
    hidden = require_int(obj, "hidden")
    if hidden < 1:
        raise JSONTypeError(f"Field 'hidden' must be at least 1, got {hidden}")
    raw = require_dict(obj, "parameters")
    if sorted(raw) != sorted(POLICY_PARAMETERS):
        raise JSONTypeError(
            f"Field 'parameters' must name exactly {', '.join(POLICY_PARAMETERS)}, "
            f"got {', '.join(sorted(raw))}"
        )
    parameters: dict[str, ParameterValues] = {}
    for name in POLICY_PARAMETERS:
        entry = narrow_json_to_dict(raw[name])
        parameters[name] = ParameterValues(
            shape=tuple(narrow_json_to_int(size) for size in require_list(entry, "shape")),
            values=tuple(narrow_json_to_float(value) for value in require_list(entry, "values")),
        )
    return PolicyWeights(hidden=hidden, parameters=parameters)


def render_weights_text(weights: PolicyWeights) -> str:
    """Write the weights as the text a language model reads.

    A matrix is written one row per line, ``<name> row <i>`` then its
    entries; a vector on one line under its name. Three decimals, as in the
    post's prompts. The convention is torch's: ``hidden_in.weight`` row ``i``
    holds the weights INTO hidden unit ``i``, one per input.

    Args:
        weights: The weights.

    Returns:
        The text, one line per row or vector.
    """
    lines: list[str] = []
    for name in POLICY_PARAMETERS:
        entry = weights["parameters"][name]
        numbers = [f"{value:.3f}" for value in entry["values"]]
        if len(entry["shape"]) == 1:
            lines.append(f"{name} {' '.join(numbers)}")
            continue
        width = entry["shape"][1]
        for row in range(entry["shape"][0]):
            lines.append(f"{name} row {row} {' '.join(numbers[row * width : (row + 1) * width])}")
    return "\n".join(lines)


__all__ = [
    "POLICY_PARAMETERS",
    "ControlPolicy",
    "ParameterValues",
    "PolicyWeights",
    "decode_policy_weights",
    "encode_policy_weights",
    "load_policy",
    "policy_weights",
    "render_weights_text",
]
