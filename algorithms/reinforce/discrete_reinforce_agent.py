import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.base_agent import BaseAgent
from algorithms.reinforce.reinforce_replay_memory import ReinforceReplayMemory
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
        passed through the network and the probabilities are not yet calculated from
        them.

        :param X: torch.tensor with a batch of observations
        :returns: torch.tensor with the output of the network
        """
        return self.net(X)

    def act(self, observation, deterministic=True):
        """
        Determines an action based on the current policy.

        :param observation: np.ndarray with the current observation
        :param deterministic: bool, whether or not to determine action deterministically
        :returns: action, and possibly the logprobability of that action
        """

        logits = self.forward(torch.from_numpy(observation.astype(np.float32)))
        action_distribution = Categorical(logits=logits)
        action = torch.argmax(logits) if deterministic else action_distribution.sample()
        self.prev_log_prob = action_distribution.log_prob(action)
        return action.item()

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
            loss += torch.sum(-log_probs * returns) / batch_size

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.reset()
