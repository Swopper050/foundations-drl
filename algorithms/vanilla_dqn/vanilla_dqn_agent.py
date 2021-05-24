import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algorithms.base_agent import BaseAgent
from algorithms.utils import create_mlp
from algorithms.vanilla_dqn.vanilla_dqn_replay_memory import (
    VanillaDQNReplayMemory,
)


class VanillaDQNAgent(nn.Module, BaseAgent):
    """
    Vanilla DQN agent, to be trained by the VanillaDQNTrainer on an
    environment with discrete actions.

    Vanilla DQN is closely related to SARSA, but is off-policy. This is
    because instead of using the actions taken during experience gathering
    during the calculation of the TD error, the optimal action is taken.
    The optimal action is determined using the current Q-network.

    Vanilla DQN learns a Q-function that approximates state-action values.
    It learns this function with Neural Network where the number of inputs
    equal the size of the observation and the number of outputs equal the
    number of possible actions. It outputs Q-values for all actions.

    It remembers the following tuple:
        (state, action, reward, next_state)

    And then samples from memory and minimizing the following error:
        Q(s,a) - (r + gamma * max(Q(s',a')))

    Hence, using its current approximation, it uses the optimal action in
    the TD error. This leads to the network learning the Q-function for an
    optimal policy.

    Selects actions according to a Boltzmann policy. This can be used instead
    of an epsilon-greedy policy for exploration. From the Q-values a
    probability distribution is calculated using the softmax function. The
    calculation of the softmax is dependent on tau. If tau is large (i.e. 5),
    the distribution becomes more uniform. If tau is small, the distribution
    becomes more skewed. Hence, tau can be used to explore.
    """

    def __init__(
        self,
        *,
        obs_dim,
        act_dim,
        max_memory_size=10000,
        hidden_sizes=[64],
        grad_clip=0.5
    ):
        """
        :param obs_dim: int, number of dimensions in the observation space
        :param act_dim: int, number of dimensions in the action space
        :param max_memory_size: int, max experiences to store in memory
        :param hidden_sizes: list with ints, dimensions of the hidden layers
        :param grad_clip: maximum value of the gradients during an update
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.net = create_mlp(
            sizes=[obs_dim] + hidden_sizes + [act_dim],
            hidden_activation=nn.SELU,
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

        self.memory = VanillaDQNReplayMemory(
            max_size=max_memory_size,
            experience_keys=[
                "states",
                "actions",
                "rewards",
                "next_states",
                "dones",
            ],
        )

        for param in self.parameters():
            param.register_hook(
                lambda grad: torch.clamp(grad, -grad_clip, grad_clip)
            )

    def forward(self, X):
        """
        Performs a forward pass through the network. Q-values for all actions
        are returned.

        :param X: torch.tensor with a batch of observations
        :returns: torch.tensor with the output of the network
        """
        return self.net(X)

    def act(self, observation, *, tau=0.001):
        """
        Determines an action based on the current Q-function.

        :param observation: np.ndarray with the current observation
        :param tau: float, value that influences the sampling distribution.
        :returns: action, and possibly the logprobability of that action
        """
        q_vals = self.forward(torch.from_numpy(observation.astype(np.float32)))
        q_vals /= tau
        action_distribution = Categorical(logits=q_vals)
        return action_distribution.sample().item()

    def store_step(self, obs, action, reward, next_obs, done):
        """
        Stores the information about a step. Vanilla DQN remembers:
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
        Performs a training step according to the Vanilla DQN algorithm.
        Samples a batch of experiences from memory and uses them to calculate
        the TD error as specified by the DQN algorithm (using optimal actions).
        Then the network is updated.

        :param gamma: float, the discount factor to use
        :param batch_size: int, batch size to use
        :param n_updates: int, number of updates to perform with the batch
        """

        # Sample a batch of experiences from memory
        batch = self.memory.sample(batch_size=batch_size)
        states = batch["states"]
        actions = batch["actions"].type(torch.int64)
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Perform a number of updates with this batch
        for _ in range(n_updates):
            # Calculate Q-values for all actions for the 'current states'
            q_preds = self.forward(states)

            # Calculate Q-values for all actions for the 'next states'
            with torch.no_grad():
                next_q_preds = self.forward(next_states)

            # Extract Q-values for the actions that were actually taken
            action_q_preds = q_preds.gather(-1, actions.unsqueeze(-1)).squeeze(
                -1
            )

            # Select optimal actions with the Q-estimates of the 'next states'
            optimal_action_q_preds, _ = next_q_preds.max(axis=1)

            # Calculate the targets by adding the observed rewards
            # Note: (1 - dones) handles terminal states
            q_targets = rewards + (1 - dones) * gamma * optimal_action_q_preds

            # Calculate the loss, which is:
            #       0.5 * ( Q(s,a) - (r + gamma * max(Q(s', a'))) ) ** 2
            loss = self.criterion(action_q_preds, q_targets)

            # Perform a network update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
