import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.base_agent import BaseAgent
from algorithms.utils import calculate_returns, create_mlp
from torch.distributions import Categorical


class DiscreteReinforceAgent(nn.Module, BaseAgent):
    """
    Discrete REINFORCE agent, to be trained by the ReinforceTrainer on an
    environment with discrete actions.
    """

    def __init__(self, *, obs_dim, act_dim, hidden_sizes=[128, 64]):
        """
        :param obs_dim: int, number of dimensions in the observation space
        :param act_dim: int, number of dimensions in the action space
        :param hidden_sizes: list with ints, dimensions of the hidden layers
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.net = create_mlp(
            sizes=[obs_dim] + hidden_sizes + [act_dim], hidden_activation=nn.ReLU
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)

        self.episode_reset()

    def forward(self, X):
        """
        Performs a forward pass through the network. Here, only the observation is
        passed through the network and the probabilities are not yet calculated from
        them.

        :param X: torch.tensor with a batch of observations
        :returns: torch.tensor with the output of the network
        """
        return self.net(X)

    def act(self, observation):
        """
        Determines an action based on the current policy.

        :param observation: np.ndarray with the current observation
        :returns: action, and possibly the logprobability of that action
        """

        logits = self.forward(torch.from_numpy(observation.astype(np.float32)))
        action_distribution = Categorical(logits=logits)
        action = action_distribution.sample()
        self.log_probs.append(action_distribution.log_prob(action))
        return action.item()

    def episode_reset(self):
        """
        Called at the start of an episode, which resets the memory of the
        agent. All information gathered up until now will be discarded.
        """
        self.rewards = []
        self.log_probs = []

    def store_step(self, reward):
        """
        Stores the information about a step. For REINFORCE, only the reward
        needs to be remembered (the logprob is already remembered earlier).

        :param reward: float, current reward
        """
        self.rewards.append(reward)

    def train(self, *, gamma, center_returns=True):
        """
        Performs a training step according to the REINFORCE algorithm.

        :param gamma: float, the discount factor to use
        :param center_returns: bool, if True, center the returns (apply mena baseline)
        """

        returns = calculate_returns(self.rewards, gamma=gamma)
        if center_returns:
            returns -= returns.mean()

        log_probs = torch.stack(self.log_probs)
        loss = torch.sum(-log_probs * returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
