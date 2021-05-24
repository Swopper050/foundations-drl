import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.base_agent import BaseAgent
from algorithms.sarsa.sarsa_replay_memory import SarsaReplayMemory
from algorithms.utils import create_mlp


class SarsaAgent(nn.Module, BaseAgent):
    """
    SARSA agent. The SARSA algorithm is an on-policy algorithm. This is because
    the updates are based on the SARSA tuple:
        (state, action, reward, next_state, next_action)
    where the next action should be the action taken by the current policy.

    It is a value-based algorithm, approximating a Q-function. The Q-function
    able to determine state-action values.

    It uses Temporal Difference (TD) learning to estimate errors with which
    it can update its Q-function. It uses bootstrapping to do this, and makes
    use of the fact that:
        Q(s, a) = E[r + gamma * Q(s', a')]
    Intuitively, the state-action value is the same as the reward received
    added to the discounted state-action value of the next timestep. This is
    true by definition (Bellman equations).

    This implies that SARSA does not have to wait for the end of an episode in
    order to perform episodes, because updates do not depend on the whole
    episode anymore (this was the case with REINFORCE).
    As training continues, automatically values get 'backed up', because
    experiences of later timesteps will be reflected in the Q-function.

    The Q-function is approximated using a Neural Network. It has as inputs
    the size of the observation and will output state-action values for all
    actions simultaneously, hence the number of outputs will be equal to the
    number of actions. This makes it easy to derive a policy from this, as
    we can simply take the action with the largest Q-value, i.e. the output
    neuron with the largest value.

    The network will be updated using gathered experiences as follows. It will
    try to minimize the following error:
        0.5 * (Q(s,a) - (r + gamma * Q(s', a')))^2
    Intuitively, the error will be its current estimate minus the observed
    estimate, which is dependent on the observed reward and the estimate of
    the state-action value at the next timestep.

    In order to maintain exploration, the policy will be made epsilon greedy,
    meaning (1 - epsilon) of the time the action with the largest Q-value
    will be selected, and an epsilon fraction of the time a random action is
    selected.
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
            sizes=[obs_dim] + hidden_sizes + [act_dim],
            hidden_activation=nn.ReLU,
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
            param.register_hook(
                lambda grad: torch.clamp(grad, -grad_clip, grad_clip)
            )

    def forward(self, X):
        """
        Performs a forward pass through the network. Returns the Q-values
        for every action.

        :param X: torch.tensor with a batch of observations
        :returns: torch.tensor with the output of the network
        """
        return self.net(X)

    def act(self, observation, *, epsilon=0.0):
        """
        Determines an action based on an e-greedy policy.

        :param observation: np.ndarray with the current observation
        :param epsilon: float, chance on a random action
        :returns: int, action to take
        """

        # Perform a random action
        if epsilon > np.random.rand():
            return np.random.randint(self.act_dim)
        # Perform an action based on policy
        else:
            q_vals = self.forward(
                torch.from_numpy(observation.astype(np.float32))
            )
            return torch.argmax(q_vals).item()

    def store_step(self, obs, action, reward, next_obs, next_action, done):
        """
        Stores the information about a step. As implied by its name:
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

    def perform_training(self, *, gamma):
        """
        Performs a training step according to the SARSA algorithm.
        Samples a batch from memory, consisting of experiences, and
        calculates the TD errors for all experiences in the batch. The
        Q-network is updated based on these errors.

        :param gamma: float, the discount factor to use
        """

        # Sample batch and convert all to torch.tensors
        batch = self.memory.sample()
        states = torch.as_tensor(batch["states"], dtype=torch.float32)
        next_states = torch.as_tensor(
            batch["next_states"], dtype=torch.float32
        )
        actions = torch.as_tensor(batch["actions"])
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.int8)

        # Estimate Q-values for all actions for all 'current states'
        # Note: this is the current estimate, which we will refine based
        #       on the TD error
        q_preds = self.forward(states)

        # Estimate Q-values for all actions for all 'next states'
        # Note: this is the bootstrap part, where the TD error is based
        #       on the current Q-network
        with torch.no_grad():
            next_q_preds = self.forward(next_states)

        # We only use the Q-values of the actions we actually took. Hence
        # we select the Q-values of the actions we have taken which we can
        # get from our sampled batch as well.
        action_q_preds = q_preds.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # We also do this for the 'next states'.
        next_action_q_preds = next_q_preds.gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        # Now we can calulcate the targets by adding the observed reward and
        # discounting the estimated Q-values of the next timestep
        # Note: (1 - done) handles terminal states
        q_targets = rewards + (1 - dones) * gamma * next_action_q_preds

        # We can calculate the loss which is the difference between the
        # predicted Q-values at 'current states' and the calculated targets
        loss = self.criterion(action_q_preds, q_targets)

        # Perform an network update based on these errors
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset the memory as we can not use the experiences again
        self.memory.reset()
