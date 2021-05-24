import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algorithms.base_agent import BaseAgent
from algorithms.reinforce.reinforce_replay_memory import ReinforceReplayMemory
from algorithms.utils import calculate_returns, create_mlp


class DiscreteReinforceAgent(nn.Module, BaseAgent):
    """
    REINFORCE is an policy-based algorithm. This means it tries to learn
    a policy directly from experience, without learning a value function.
    It does so by estimating the policy gradient. The policy gradient is
    the derivative of the objective function (maximizing the expected
    return of an episode) with respect to the policy's parameters:

        grad J(pi) = E[sum(Rt(tau) * grad log(pi(at | st)))]  summed over t

    Intuitively: the policy-gradient, given an episode, is the sum over all
    timesteps of the returns at that timestep multiplied by the derivative
    of the log probability of the action taken that timestep. Hence, actions
    with a high return become more likely, whereas actions with a low return
    become less likely.

    Introducing a baseline means replacing 'Rt(tau)' with '(Rt(tau) - b(st))'.
    This centers the returns, meaning that the updates will have a lower
    variance (not only negative updates, if the environment only produces
    positive returns for example).

    A discrete REINFORCE agent maps states to actions. Discrete means the
    number of actions are a set of actions such as:
        A = {'up', 'down', 'left', 'right'}

    The agent is parameterized by a Neural Network where the input size
    equals the size of the observation and the output size equals the
    number of actions.

    It has a memory which stores experiences of full episodes, which are
    used during training to estimate the policy gradient. For this, the log
    probability of the actions taken in these episodes need to be remembered
    as well.
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
            sizes=[obs_dim] + hidden_sizes + [act_dim],
            hidden_activation=nn.ReLU,
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
        Performs a forward pass through the network. Here, only the observation
        is passed through the network and the probabilities are not yet
        calculated from them.

        :param X: torch.tensor with a batch of observations
        :returns: torch.tensor with the output of the network
        """
        return self.net(X)

    def act(self, observation, deterministic=True):
        """
        Determines an action based on the current policy. Stores the log
        probability of the action as attribute.

        :param observation: np.ndarray with the current observation
        :param deterministic: bool, if True, takes action with highest prob
        :returns: int, action to take
        """

        # Get the activations from the network
        logits = self.forward(torch.from_numpy(observation.astype(np.float32)))

        # Use the logits to determine probabilities for all actions
        action_distribution = Categorical(logits=logits)

        if deterministic:
            # Choose the action with the highest activation
            action = torch.argmax(logits)
        else:
            # Sample randomly using the probabilities
            action = action_distribution.sample()

        # Determine the log probability of the chosen action
        self.prev_log_prob = action_distribution.log_prob(action)

        return action.item()

    def store_step(self, obs, action, reward, next_obs, done):
        """
        Stores the information about a step. Adds the log probability of the
        action as well.

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
        Estimates the policy gradient using the episodes stored in
        memory and updates the policy (neural network) accordingly.

        :param gamma: float, the discount factor to use
        :param center_returns: bool, if True, center the returns
        """

        # Sample a batch of episodes from the memory
        batch = self.memory.sample()
        batch_size = len(batch["rewards"])

        # Calculate the sum of the loss for all episodes
        loss = torch.tensor(0, dtype=torch.float32)
        for i in range(batch_size):
            # Calculate the returns for every timestep in the current episode
            returns = calculate_returns(batch["rewards"][i], gamma=gamma)

            # Apply baseline if specified
            if center_returns:
                returns -= returns.mean()

            log_probs = torch.stack(batch["log_probs"][i])
            # Add estimate of gradient for this episode to the loss
            # Note: it is *-1 as pytorch by default minimizes loss
            loss += torch.sum(-log_probs * returns) / batch_size

        # Perform a network update based on the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Discard all memory, as we can not use these experiences again
        self.memory.reset()
