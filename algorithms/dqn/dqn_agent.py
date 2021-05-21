import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.base_agent import BaseAgent
from algorithms.dqn.prioritized_replay_memory import PrioritizedReplayMemory
from algorithms.utils import create_mlp
from torch.distributions import Categorical


class DQNAgent(nn.Module, BaseAgent):
    """
    DQN agent, to be trained by the DQNTrainer on an environment with discrete
    actions. Selects actions according to a Boltzmann policy.
    """

    def __init__(
        self,
        *,
        obs_dim,
        act_dim,
        max_memory_size=10000,
        hidden_sizes=[64],
        grad_clip=0.5,
        polyak_coef=0.995,
    ):
        """
        :param obs_dim: int, number of dimensions in the observation space
        :param act_dim: int, number of dimensions in the action space
        :param max_memory_size: int, max number of experience to store in the memory
        :param hidden_sizes: list with ints, dimensions of the hidden layers
        :param grad_clip: maximum value of the gradients during an update
        :param polyak_coef: float, polyak coefficient used to update the target networks
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.polyak_coef = polyak_coef

        self.net = create_mlp(
            sizes=[obs_dim] + hidden_sizes + [act_dim], hidden_activation=nn.SELU
        )

        self.target_net = copy.deepcopy(self.net)
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

        self.memory = PrioritizedReplayMemory(
            max_size=max_memory_size,
            experience_keys=[
                "states",
                "actions",
                "rewards",
                "next_states",
                "dones",
            ],
        )

        for param in self.net.parameters():
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

    def act(self, observation, *, tau=0.001):
        """
        Determines an action based on the current q function.

        :param observation: np.ndarray with the current observation
        :param tau: float, value that influences the sampling distribution.
        :returns: action, and possibly the logprobability of that action
        """
        logits = self.forward(torch.from_numpy(observation.astype(np.float32)))
        logits /= tau
        action_distribution = Categorical(logits=logits)
        return action_distribution.sample().item()

    def store_step(self, obs, action, reward, next_obs, done):
        """
        Stores the information about a step. DQN remembers:
            (state, action, reward, next_state)

        :param state: np.ndarray with the state at time t
        :param action: np.ndarray with the action at time t
        :param reward: float with reward for action at time t
        :param next_state: np.ndarray with state at time t+1
        :param done: bool, whether or not the episode is done
        """
        self.memory.store_experience(
            {
                "states": obs,
                "actions": action,
                "rewards": reward,
                "next_states": next_obs,
                "dones": done,
            }
        )

    def perform_training(self, *, gamma, batch_size, n_updates):
        """
        Performs a training step according to the DQN algorithm.

        :param gamma: float, the discount factor to use
        :param batch_size: int, batch size to use
        :param n_updates: int, number of times to perform an update with the batch
        """

        batch = self.memory.sample(batch_size=batch_size)
        states = batch["states"]
        actions = batch["actions"].type(torch.int64)
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        for _ in range(n_updates):
            q_preds = self.net(states)
            with torch.no_grad():
                next_q_preds = self.net(next_states)
                target_next_q_preds = self.target_net(next_states)

            action_q_preds = q_preds.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            next_actions = next_q_preds.argmax(dim=-1, keepdims=True)
            next_act_q_preds = target_next_q_preds.gather(-1, next_actions).squeeze(-1)
            action_q_targets = rewards + (1 - dones) * gamma * next_act_q_preds

            loss = self.criterion(action_q_preds, action_q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_errors = (action_q_targets - action_q_preds.detach()).abs().numpy()
            self.memory.update_priorities(batch_errors)

        self.update_target_network()

    def update_target_network(self):
        with torch.no_grad():
            for param, target_param in zip(
                self.net.parameters(), self.target_net.parameters()
            ):
                target_param.data.mul_(self.polyak_coef)
                target_param.data.add_((1 - self.polyak_coef) * param.data)
