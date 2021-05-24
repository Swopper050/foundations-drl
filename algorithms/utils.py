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
