import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.base_agent import BaseAgent
from algorithms.sarsa.sarsa_replay_memory import SarsaReplayMemory
from algorithms.utils import create_mlp


class SarsaAgent(nn.Module, BaseAgent):
    """
    Discrete REINFORCE agent, to be trained by the ReinforceTrainer on an
    environment with discrete actions.
    """

    def __init__(self, *, obs_dim, act_dim, hidden_sizes=[64], grad_clip=0.5):
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
        self.optimizer = optim.Adam(self.parameters(), lr=0.05)
        self.criterion = nn.MSELoss()

        self.memory = SarsaReplayMemory(
            experience_keys=[
                "states",
                "actions",
                "rewards",
                "next_states",
                "next_actions",
                "dones",
            ]
        )

        for param in self.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -grad_clip, grad_clip))

    def forward(self, X):
        """
        Performs a forward pass through the network. Here, only the observation is
        passed through the network and the probabilities are not yet calculated from
        them.

        :param X: torch.tensor with a batch of observations
        :returns: torch.tensor with the output of the network
        """
        return self.net(X)

    def act(self, observation, *, epsilon=0.0):
        """
        Determines an action based on the current q function.

        :param observation: np.ndarray with the current observation
        :param epsilon: float, chance on a random action
        :returns: action, and possibly the logprobability of that action
        """

        # Perform a random action
        if epsilon > np.random.rand():
            return np.random.randint(self.act_dim)
        # Perform an action based on policy
        else:
            q_vals = self.forward(torch.from_numpy(observation.astype(np.float32)))
            return torch.argmax(q_vals).item()

    def store_step(self, obs, action, reward, next_obs, next_action, done):
        """
        Stores the information about a step. As implied by its name, SARSA remembers:
            (state, action, reward, next_state, next_action)

        :param state: np.ndarray with the state at time t
        :param action: np.ndarray with the action at time t
        :param reward: float with reward for action at time t
        :param next_state: np.ndarray with state at time t+1
        :param next_action: np.ndarray with the action at time t+1
        :param done: bool, whether or not the episode is done
        """
        self.memory.store_experience(
            {
                "states": obs,
                "actions": action,
                "rewards": reward,
                "next_states": next_obs,
                "next_actions": next_action,
                "dones": done,
            }
        )

    def train(self, *, gamma):
        """
        Performs a training step according to the SARSA algorithm.

        :param gamma: float, the discount factor to use
        """

        batch = self.memory.sample()

        states = torch.as_tensor(batch["states"], dtype=torch.float32)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"])
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.int8)

        q_preds = self.forward(states)
        with torch.no_grad():
            next_q_preds = self.forward(next_states)

        action_q_preds = q_preds.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        next_action_q_preds = next_q_preds.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_targets = rewards + (1 - dones) * gamma * next_action_q_preds

        loss = self.criterion(action_q_preds, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.reset()
