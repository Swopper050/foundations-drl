import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from algorithms.base_agent import BaseAgent
from algorithms.reinforce.reinforce_replay_memory import ReinforceReplayMemory
from algorithms.utils import calculate_returns, create_mlp

LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0


class ContinuousReinforceAgent(nn.Module, BaseAgent):
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

    A continuous REINFORCE agent maps states to parameters that describe
    the probability functions of actions. Continuous means that every action
    is a number in a certain range. This class assumes Gaussian distributions
    for every action.
    Hence, an state yields Gaussian distributions describing the probability
    of an action. The mean of the distribution describes the most likely
    action, and hence the means will be used when acting deterministically.
    Otherwise, we want to explore by sampling from this distribution.

    The agent is parameterized by a Neural Network where the input size
    equals the size of the observation and there are two output layers.
    One output layer outputs the mean actions, the other layer outputs the
    standard deviations of the actions.

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
        Performs a forward pass through the network. Here, only the observation
        is passed through the network and the probability distribution
        parmeters are returned.

        :param X: torch.tensor with a batch of observations
        :returns: torch.tensor with the means and log std devs for every action
        """
        net_out = self.net(X)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        return mu, log_std

    def act(self, observation, deterministic=True):
        """
        Determines an action based on the current policy.

        :param observation: np.ndarray with the current observation
        :param deterministic: bool, if True, acts deterministically
        :returns: np.ndarray with the action
        """

        # Determine the mean vector and the log standard deviation vector
        mu, log_std = self.forward(
            torch.from_numpy(observation.astype(np.float32))
        )
        # Convert to standard deviation domain
        # Note: the clamp is because standard devations >= 0
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))

        # Construct a Normal distribution using these parameters
        action_distribution = Normal(mu, std)

        # Retrieve the action, if deterministc, use the means. If not,
        # sample from the distribution.
        action = mu if deterministic else action_distribution.rsample()

        # Remember the log probability of the chosen action
        self.prev_log_prob = action_distribution.log_prob(action)

        return action.detach().numpy()

    def store_step(self, obs, action, reward, next_obs, done):
        """
        Stores the information about a step. Includes the log probability
        of the action.

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
        Estimates the policy-gradient using the episodes stored in
        memory and performs an update accordingly.

        :param gamma: float, the discount factor to use
        :param center_returns: bool, if True, center the returns
        """

        # Sample a batch of episodes from memory
        batch = self.memory.sample()
        batch_size = len(batch["rewards"])

        # Calculate the mean loss for all these episodes
        loss = torch.tensor(0, dtype=torch.float32)
        for i in range(batch_size):
            # Calculate the returns for all timesteps in the current episode
            returns = calculate_returns(batch["rewards"][i], gamma=gamma)

            # If specified, center returns (apply baseline) to reduce variance
            if center_returns:
                returns -= returns.mean()

            # Add the estimated gradient of this episode to the total loss
            # Note: it is - because pytorch by default minimizes loss
            log_probs = torch.stack(batch["log_probs"][i])
            loss += torch.sum(-log_probs * returns.view(-1, 1)) / batch_size

        # Apply the update to the neural network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset the memory, as the experiences can not be reused
        self.memory.reset()
