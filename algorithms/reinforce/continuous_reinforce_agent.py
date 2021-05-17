import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.base_agent import BaseAgent
from algorithms.reinforce.reinforce_replay_memory import ReinforceReplayMemory
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
        self.memory = ReinforceReplayMemory(
            experience_keys=[
                "states",
                "actions",
                "log_probs",
                "rewards",
                "next_states",
                "dones",
            ]
        )

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

    def act(self, observation, deterministic=True):
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
        self.prev_log_prob = action_distribution.log_prob(action)
        return action.detach().numpy()

    def store_step(self, obs, action, reward, next_obs, done):
        """
        Stores the information about a step. For REINFORCE, only the reward
        needs to be remembered (the logprob is already remembered earlier).

        :param state: np.ndarray with the state at time t
        :param action: np.ndarray with the action at time t
        :param reward: float with reward for this action
        :param next_state: np.ndarray with state at time t+1
        :param done: bool, whether or not the episode is done
        """
        self.memory.store_experience(
            {
                "states": obs,
                "actions": action,
                "log_probs": self.prev_log_prob,
                "rewards": reward,
                "next_states": next_obs,
                "dones": done,
            }
        )

    def perform_training(self, *, gamma, center_returns=True):
        """
        Performs a training step according to the REINFORCE algorithm.

        :param gamma: float, the discount factor to use
        :param center_returns: bool, if True, center the returns (apply mena baseline)
        """

        batch = self.memory.sample()
        batch_size = len(batch["rewards"])

        loss = torch.tensor(0, dtype=torch.float32)
        for i in range(batch_size):
            returns = calculate_returns(batch["rewards"][i], gamma=gamma)
            if center_returns:
                returns -= returns.mean()

            log_probs = torch.stack(batch["log_probs"][i])
            loss += torch.sum(-log_probs * returns.view(-1, 1)) / batch_size

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.reset()
