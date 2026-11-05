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


class ControlPolicy(torch.nn.Module):
    """``2 -> hidden -> hidden -> 3`` with tanh, scoring the three actions.

    Attributes:
        hidden_in: Observation to the first hidden layer.
        hidden_mid: First hidden layer to the second.
        action_out: Second hidden layer to one score per action.
    """

    hidden_in: torch.nn.Linear
    hidden_mid: torch.nn.Linear
    action_out: torch.nn.Linear

    def __init__(self, hidden: int) -> None:
        """Build the layers with torch's default initialisation.

        Args:
            hidden: Units per hidden layer.
        """
        super().__init__()
        self.hidden_in = torch.nn.Linear(2, hidden)
        self.hidden_mid = torch.nn.Linear(hidden, hidden)
        self.action_out = torch.nn.Linear(hidden, 3)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Score each action for each observation.

        Args:
            observation: ``(episodes, 2)`` scaled position and velocity.

        Returns:
            ``(episodes, 3)`` action scores.
        """
        first = torch.tanh(self.hidden_in(observation))
        second = torch.tanh(self.hidden_mid(first))
        scores: torch.Tensor = self.action_out(second)
        return scores

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
    return PolicyWeights(hidden=int(policy.hidden_in.out_features), parameters=parameters)


def load_policy(weights: PolicyWeights) -> ControlPolicy:
    """Build a policy holding exactly the given weights.

    Args:
        weights: The weights.

    Returns:
        The policy, in evaluation mode.

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
    policy.eval()
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
