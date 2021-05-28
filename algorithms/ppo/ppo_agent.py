import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from algorithms.base_agent import BaseAgent
from algorithms.ppo.ppo_replay_memory import PPOReplayMemory
from algorithms.utils import create_mlp, estimate_generalized_advantages

MIN_LOG_STD = -20.0
""" Min value for the log standard deviation of the Continuous Actor. """

MAX_LOG_STD = 2.0
""" Max value for the log standard deviation of the Continuous Actor. """


class ContinuousActor(nn.Module):
    """
    Continuous Actor class which maps observations to parameterized action
    distributions. It is an MLP with two output layers. One layer outputs the
    means for all actions, the other layer outputs the log standard deviation
    for all layers. It assumes the actions can be described by Gaussians.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, grad_clip):
        """
        :param obs_dim: int, number of dimensions in the observation space
        :param act_dim: int, number of dimensions in the action space
        :param hidden_sizes: list with ints, sizes of the hidden layers
        :param grad_clip: float, max value of a gradient
        """
        super().__init__()
        sizes = [obs_dim] + hidden_sizes
        self.net = create_mlp(
            sizes=sizes, hidden_activation=nn.ReLU, output_activation=nn.ReLU
        )
        self.mu_layer = nn.Linear(sizes[-1], act_dim)
        self.std_layer = nn.Linear(sizes[-1], act_dim)

        for param in self.parameters():
            param.register_hook(
                lambda grad: torch.clamp(grad, -grad_clip, grad_clip)
            )

    def forward(self, obs):
        """
        Performs a forward pass through the network. Returns means and
        standard deviations for all action dimensions.

        :param obs: torch.tensor with a batch of observations
        :returns: torch.tensor with the means and one with std deviations
        """
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        std_dev = torch.exp(
            torch.clamp(self.std_layer(net_out), MIN_LOG_STD, MAX_LOG_STD)
        )
        return mu, std_dev


class DiscreteActor(nn.Module):
    """
    Discrete actor class which maps observations to action logits.
    It is a simple MLP network. It is specific for discrete actions,
    and the returned logits can be turned into probabilities afterwards.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, grad_clip):
        """
        :param obs_dim: int, number of dimensions in the observation space
        :param act_dim: int, number of dimensions in the action space
        :param hidden_sizes: list with ints, sizes of the hidden layers
        :param grad_clip: float, max value of a gradient
        """
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.net = create_mlp(sizes=sizes, hidden_activation=nn.ReLU)
        for param in self.parameters():
            param.register_hook(
                lambda grad: torch.clamp(grad, -grad_clip, grad_clip)
            )

    def forward(self, obs):
        """
        Performs a forward pass through the network. Returns logits for
        all actions given the current observations

        :param obs: torch.tensor with a batch of observations
        :returns: torch.tensor with the output of the network
        """
        return self.net(obs)


class ValueNetwork(nn.Module):
    """
    Class that estimates a value function, i.e. V(s). It estimates this value
    based on the observation. It is a simple MLP.
    """

    def __init__(self, obs_dim, hidden_sizes, grad_clip):
        """
        :param obs_dim: int, number of dimensions in the observation space
        :param hidden_sizes: list with ints, sizes of the hidden layers
        :param grad_clip: float, max value of a gradient
        """

        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [1]
        self.net = create_mlp(sizes=sizes, hidden_activation=nn.ReLU)
        for param in self.parameters():
            param.register_hook(
                lambda grad: torch.clamp(grad, -grad_clip, grad_clip)
            )

    def forward(self, obs):
        """
        Performs a forward pass through the network. Returns V-values for
        the current observation.

        :param obs: torch.tensor with a batch of observations
        :returns: torch.tensor with the output of the network
        """
        return self.net(obs)


class PPOAgent(BaseAgent):
    """
    An PPO agent that trains according to Proximal Policy Optimization (PPO).
    It learns both a policy and a value function. The policy is learned in the
    same manner as the A2C extension on the REINFORCE algorithm, i.e. with
    action log probabilities reinforced by some sort of signal.
    However, here the objective is modified in order to constraint the policy
    to 'move' too far away from the current policy.

    Here, the policy is the Actor, and is represented by a network.
    The policy is reinforced by the advantage function:
        A(s,a) = Q(s,a) - V(s)

    The advantage function states the preference of a certain action over
    the average action in the current state. This advantage function can be
    determined solely from a value function, i.e. the Critic network, which
    learns the value function through TD learning, as is done with DQN.

    The main difference with A2C lies in the objective function. Instead
    of the log probability of the action, it extends this with the following
    probability ratio r:
        r = pi(a|s) / pi_old(a|s)

    Hence, it shows how much the ratio diverges from the old policy.
    Then, the objective clips this ratio as follows:
        J(pi) = E[min(r * A(s,a), clip(r, 1-e, 1+e) * A(s,a)]

    Hence, when the ratio exceeds the trust region in order to improve on the
    objective, it is ignored.

    Where the Advantage function is estimated with Generalized Advantage
    Estimation (GAE), which means:
        A(s,a) = sum((gamma * lambda)^l * (r + gamma * V(s') - V(s)))

    Where we sum over t, the time in the episode.

    The Critic simply learns the value function V(s), and its targets
    are generated by:
        Vtar(s) = A(s,a) + V(s)

    Which holds as action a is selected by the current policy.
    """

    def __init__(
        self,
        *,
        obs_dim,
        act_dim,
        discrete,
        hidden_sizes=[64],
        grad_clip=0.5,
    ):
        """
        :param obs_dim: int, number of dimensions in the observation space
        :param act_dim: int, number of dimensions in the action space
        :param discrete: bool, whether to actor should be discrete or not
        :param hidden_sizes: list with ints, dimensions of the hidden layers
        :param grad_clip: maximum value of the gradients during an update
        """

        self.entropy_weight = 0.01
        self.gamma = 0.99
        self.exp_weight = 0.95
        self.clip_eps = 0.2

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.discrete = discrete

        actor_class = DiscreteActor if discrete else ContinuousActor
        self.actor = actor_class(obs_dim, act_dim, hidden_sizes, grad_clip)
        self.old_actor = actor_class(obs_dim, act_dim, hidden_sizes, grad_clip)
        self.critic = ValueNetwork(obs_dim, hidden_sizes, grad_clip)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0005)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0005)
        self.critic_criterion = nn.MSELoss()

        self.memory = PPOReplayMemory(
            experience_keys=[
                "states",
                "actions",
                "rewards",
                "next_states",
                "dones",
            ],
        )

    def act(self, obs, *, deterministic=True):
        """
        Determines an action using the Actor network

        :param obs: np.ndarray with the current observation
        :param deterministic: whether or not to act deterministically
        :param with_logprob: bool, if True returns the logprob as well
        :returns: action
        """

        # In the case of an environment with discrete actions
        if self.discrete:
            with torch.no_grad():
                logits = self.actor.forward(
                    torch.from_numpy(obs.astype(np.float32))
                )

            if deterministic:
                # Choose the action with the highest activation
                action = torch.argmax(logits).item()
            else:
                # Sample randomly using the probabilities
                action_distribution = Categorical(logits=logits)
                action = action_distribution.sample().item()

        # In the case of an environment with continuous actions
        else:
            with torch.no_grad():
                mu, std_dev = self.actor.forward(
                    torch.from_numpy(obs.astype(np.float32))
                )

            if deterministic:
                # Use the means as action (most likely actions)
                action = mu.numpy()
            else:
                # Sample randomly using the distribution that follows
                # from the parameters.
                action_distribution = Normal(mu, std_dev)
                action = action_distribution.sample().numpy()

        return action

    def store_step(self, obs, action, reward, next_obs, done):
        """
        Stores the information about a step. PPO remembers:
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

    def perform_training(self, *, n_updates):
        """
        Performs a training step according to the A2C algorithm. Roughly
        performs the following steps:
            1) Samples a batch from memory
            2) Predict values for all states in the batch
            3) Use these predictions to calculate advantages and value targets
            4) Update actor and critic networks

        :param n_updates: int, number of updates to perform
        """

        self.old_actor.load_state_dict(self.actor.state_dict())
        batch = self.memory.sample()

        value_preds = self.critic.forward(batch["states"]).squeeze()
        # We need this as we are still bootstrapping, for which we do not
        # want to use the gradient
        detached_value_preds = value_preds.detach().clone()

        # This implementation uses GAE (instead of n-step estimation)
        advantages = estimate_generalized_advantages(
            batch["rewards"],
            batch["dones"],
            detached_value_preds,
            self.gamma,
            self.exp_weight,
        )
        # Calculate the value targets and standardize the advantages
        value_targets = advantages + detached_value_preds
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(n_updates):
            self.update_actor(batch, advantages)
            self.update_critic(batch, value_targets)

        self.memory.reset()

    def update_actor(self, batch, advantages):
        """
        Updates the actor network. First it calculates the loss of the actor.
        This is basically the same as it is for REINFORCE. However, instead of
        a Monte Carlo estimate of the return, the advantage values are used as
        reinforce signals. Furthermore, the probability ratio between the old
        and the new policy is used, and the clipped PPO objective is used.
        The actor loss then becomes:
            min(r*A, clip(r, 1-e, 1+e)At)

        When provided, extra entropy loss is subtracted to keep
        the policy more uniform.

        :param batch: dict with torch.tensors with values for the batch
        :param advantages: torch.tensor with advantage values for the batch
        """
        if self.discrete:
            act_dists = Categorical(logits=self.actor.forward(batch["states"]))
            log_probs = act_dists.log_prob(batch["actions"])

            with torch.no_grad():
                old_log_probs = Categorical(
                    logits=self.old_actor.forward(batch["states"])
                ).log_prob(batch["actions"])

        else:
            act_dists = Normal(*self.actor.forward(batch["states"]))
            log_probs = act_dists.log_prob(batch["actions"]).sum(axis=-1)

            with torch.no_grad():
                old_log_probs = (
                    Normal(*self.old_actor.forward(batch["states"]))
                    .log_prob(batch["actions"])
                    .sum(axis=-1)
                )

        ratios = torch.exp(log_probs - old_log_probs)
        clipped_ratios = torch.clamp(
            ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps
        )

        clip_loss = -1 * torch.min(
            ratios * advantages, clipped_ratios * advantages
        )

        actor_loss = clip_loss.mean()

        # Calculate the loss for the actor, and possibly add entropy loss
        if self.entropy_weight > 0.0:
            actor_loss -= self.entropy_weight * act_dists.entropy().mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_critic(self, batch, value_targets):
        """
        Updates the critic network. Calculates the current loss of the
        value estimates and updates the network accordingly.

        :param value_preds: torch.tensor (with grads) with value predictions
        :param value_targets: torch.tensor with target values
        """

        value_preds = self.critic.forward(batch["states"]).squeeze()
        critic_loss = self.critic_criterion(value_preds, value_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
