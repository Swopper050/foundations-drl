import numpy as np
import torch
import torch.nn as nn


def calculate_returns(rewards, *, gamma):
    """
    Efficiently calculates the returns for every timestep in the episode.

    :param rewards: list with rewards of the episode
    :param gamma: float, the discount factor
    :returns: torch.tensor with returns for every timestep
    """

    T = len(rewards)
    returns = np.empty(T, dtype=np.float32)

    future_return = 0.0
    for t in reversed(range(T)):
        future_return = rewards[t] + gamma * future_return
        returns[t] = future_return
    return torch.from_numpy(returns)


def estimate_generalized_advantages(
    rewards, dones, value_preds, gamma, exp_weight
):
    """
    Estimates the generalized Advantage (GAE) for the episodes in the batch.
    This makes the important assumption that the rewards are in order, i.e.
    reward[1] followed reward[0].

    :param rewards: torch.tensor with rewards for one or more episodes
    :param dones: torch.tensor with dones (0 or 1) for on or more episodes
    :param value_preds: torch.tensor with predicted values for all steps
    :param gamma: float, discount value
    :param exp_weight: lambda in the GAE expression
    """
    gaes = torch.zeros_like(rewards)
    # Last step is terminal, hence the following values are 0 at the last step
    future_gae = torch.tensor(0.0, dtype=torch.float32)
    future_value = torch.tensor(0.0, dtype=torch.float32)

    for i in reversed(range(len(rewards))):
        delta = (
            rewards[i] + gamma * future_value * (1 - dones[i]) - value_preds[i]
        )
        gaes[i] = delta + gamma * exp_weight * (1 - dones[i]) * future_gae
        future_gae = gaes[i]
        future_value = value_preds[i]
    return gaes


def create_mlp(sizes, hidden_activation, output_activation=nn.Identity):
    """
    Creates a simple Multi-Layer Perceptron (MLP) with the given architecture.
    Every layer except the last layer will have the given hidden activation
    function. The last layer can have a custom activation function, where the
    default activation is the identity function.

    :param hidden_sizes: list of integers, specifying the hidden layers sizes
    :param hidden_activation: activation function to use for hidden layers
    :param output_activation: activation function to use for the output layer
    :returns: nn.Sequential network
    """

    layers = []
    for j in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), hidden_activation()]
    layers += [nn.Linear(sizes[-2], sizes[-1]), output_activation()]
    return nn.Sequential(*layers)
