"""The control policy, its weights file, its text, and its training.

The policy is a real ``torch.nn.Module`` throughout. Where an expected value
is a number, it is worked by hand: a policy whose weights are written by the
test so its output is computable on paper, and a three-step return computed
from its definition.
"""

from __future__ import annotations

import math

import pytest
import torch
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from model_trainer.core.services.model.editing.verify import parameter_digests, weights_digest
from model_trainer.core.services.model.weight_space.control_policy import (
    POLICY_PARAMETERS,
    ControlPolicy,
    ParameterValues,
    PolicyWeights,
    decode_policy_weights,
    encode_policy_weights,
    load_policy,
    policy_weights,
    render_weights_text,
)
from model_trainer.core.services.model.weight_space.policy_training import (
    TrainingConfig,
    discounted_returns,
    train_policy,
)


def _hand_policy() -> ControlPolicy:
    """A one-unit policy whose action scores are computable on paper.

    hidden_in: unit 0 reads velocity with weight 1; hidden_mid copies it;
    action_out scores push-right by +2 times the unit and push-left by -2
    times it, no-push 0. So the policy pushes the way the car moves.

    Returns:
        The policy.
    """
    policy = ControlPolicy(1)
    with torch.no_grad():
        for _, parameter in policy.named_parameters():
            parameter.zero_()
        policy.hidden_in.weight[0, 1] = 1.0
        policy.hidden_mid.weight[0, 0] = 1.0
        policy.action_out.weight[0, 0] = -2.0
        policy.action_out.weight[2, 0] = 2.0
    return policy


def test_the_hand_policy_scores_what_paper_says() -> None:
    """Velocity 0.5: tanh(tanh(0.5)) = 0.4329; scores (-0.8658, 0, 0.8658)."""
    policy = _hand_policy()
    scores = policy.forward(torch.tensor([[0.3, 0.5]]))
    inner = math.tanh(math.tanh(0.5))
    assert torch.allclose(scores, torch.tensor([[-2.0 * inner, 0.0, 2.0 * inner]]), atol=1e-6)
    assert torch.equal(policy.greedy(torch.tensor([[0.3, 0.5], [0.3, -0.5]])), torch.tensor([2, 0]))


def test_weights_survive_the_file_bit_for_bit() -> None:
    torch.manual_seed(5)
    policy = ControlPolicy(4)
    text = dump_json_str(encode_policy_weights(policy_weights(policy)))
    loaded = load_policy(decode_policy_weights(narrow_json_to_dict(load_json_str(text))))
    assert weights_digest(parameter_digests(loaded)) == weights_digest(parameter_digests(policy))


def test_generated_weights_round_trip_through_the_encoding() -> None:
    """Property: decode(encode(w)) == w, over policies of several widths."""
    for hidden in (1, 2, 3, 8):
        torch.manual_seed(hidden)
        weights = policy_weights(ControlPolicy(hidden))
        assert decode_policy_weights(encode_policy_weights(weights)) == weights


def test_the_text_lists_rows_under_their_parameter_names() -> None:
    text = render_weights_text(policy_weights(_hand_policy()))
    assert text.splitlines() == [
        "hidden_in.weight row 0 0.000 1.000",
        "hidden_in.bias 0.000",
        "hidden_mid.weight row 0 1.000",
        "hidden_mid.bias 0.000",
        "action_out.weight row 0 -2.000",
        "action_out.weight row 1 0.000",
        "action_out.weight row 2 2.000",
        "action_out.bias 0.000 0.000 0.000",
    ]


def _weights(hidden: int, shape_of_first: tuple[int, ...]) -> PolicyWeights:
    """Build weights for a policy, with the first parameter's shape overridden.

    Args:
        hidden: The declared width.
        shape_of_first: The shape to give ``hidden_in.weight``.

    Returns:
        The weights.
    """
    real = policy_weights(ControlPolicy(hidden))
    first = real["parameters"]["hidden_in.weight"]
    parameters = dict(real["parameters"])
    parameters["hidden_in.weight"] = ParameterValues(shape=shape_of_first, values=first["values"])
    return PolicyWeights(hidden=hidden, parameters=parameters)


def test_loading_weights_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(JSONTypeError, match=r"'hidden_in\.weight' of a 2-unit policy has shape"):
        load_policy(_weights(2, (4, 1)))


def test_decoding_refuses_a_width_below_one_and_a_wrong_inventory() -> None:
    encoded = encode_policy_weights(policy_weights(ControlPolicy(2)))
    with pytest.raises(JSONTypeError, match="'hidden' must be at least 1, got 0"):
        decode_policy_weights({**encoded, "hidden": 0})
    with pytest.raises(JSONTypeError, match="must name exactly"):
        decode_policy_weights({**encoded, "parameters": {"hidden_in.weight": {}}})
    assert len(POLICY_PARAMETERS) == 6


def test_a_discounted_return_sums_the_future_by_hand() -> None:
    """Rewards (1, 2, 4) at discount 0.5: G = (1 + 1 + 1, 2 + 2, 4) = (3, 4, 4)."""
    rewards = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float64)
    assert torch.equal(
        discounted_returns(rewards, 0.5), torch.tensor([[3.0], [4.0], [4.0]], dtype=torch.float64)
    )


_TINY: TrainingConfig = {
    "seed": 3,
    "hidden": 4,
    "iterations": 2,
    "batch": 4,
    "discount": 0.9,
    "learning_rate": 0.01,
}


def test_training_moves_the_weights_and_is_repeatable() -> None:
    first = train_policy(_TINY)
    second = train_policy(_TINY)
    torch.manual_seed(_TINY["seed"])
    untrained = ControlPolicy(_TINY["hidden"])
    trained_digest = weights_digest(parameter_digests(first["policy"]))
    assert trained_digest == weights_digest(parameter_digests(second["policy"]))
    assert trained_digest != weights_digest(parameter_digests(untrained))
    assert not first["policy"].training
    assert 0 <= first["final_batch_successes"] <= _TINY["batch"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"batch": 0}, "must each be at least 1"),
        ({"discount": 1.5}, "must lie in \\[0, 1\\], got 1.5"),
        ({"learning_rate": 0.0}, "must be positive, got 0.0"),
    ],
)
def test_training_settings_outside_their_range_are_refused(
    override: dict[str, int | float], message: str
) -> None:
    config: TrainingConfig = {
        "seed": _TINY["seed"],
        "hidden": _TINY["hidden"],
        "iterations": _TINY["iterations"],
        "batch": int(override.get("batch", _TINY["batch"])),
        "discount": float(override.get("discount", _TINY["discount"])),
        "learning_rate": float(override.get("learning_rate", _TINY["learning_rate"])),
    }
    with pytest.raises(ValueError, match=message):
        train_policy(config)
