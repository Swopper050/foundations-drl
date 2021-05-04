import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.base_agent import BaseAgent
from algorithms.utils import calculate_returns, create_mlp
from torch.distributions import Normal

LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0


class ContinuousReinforceAgent(nn.Module, BaseAgent):
    """
    Continuous REINFORCE agent, to be trained by the ReinforceTrainer on an
    environment with continuous actions.
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
            sizes=[obs_dim] + hidden_sizes,
            hidden_activation=nn.ReLU,
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=0.002)
        self.episode_reset()

    def forward(self, X):
        """
        Performs a forward pass through the network. Here, only the observation is
        passed through the network and the probability distribution parmeters are
        returned.

        :param X: torch.tensor with a batch of observations
        :returns: torch.tensor with the means for every action dimension
        """
        net_out = self.net(X)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        return mu, log_std

    def act(self, observation, deterministic=False):
        """
        Determines an action based on the current policy.

        :param observation: np.ndarray with the current observation
        :param deterministic: bool, whether or not to determine action deterministically
        :returns: action, and possibly the logprobability of that action
        """

        mu, log_std = self.forward(torch.from_numpy(observation.astype(np.float32)))
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        action_distribution = Normal(mu, std)
        action = mu if deterministic else action_distribution.rsample()
        self.log_probs.append(action_distribution.log_prob(action))
        return action.detach().numpy()

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
        loss = torch.sum(-log_probs * returns.view(-1, 1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
