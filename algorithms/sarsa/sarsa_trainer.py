import gym
import seaborn as sns

from algorithms.base_trainer import BaseTrainer
from algorithms.sarsa.sarsa_agent import SarsaAgent

sns.set_style("darkgrid")


class SarsaTrainer(BaseTrainer):
    """
    Helper class for training an agent using the SARSA algorithm. Implements
    the main training loop for SARSA.
    """

    def __init__(self, *, gamma=0.9):
        self.gamma = gamma

    def train_agent(
        self,
        *,
        env,
        test_env,
        train_every=32,
        eval_every=1000,
        max_steps=100000,
        start_epsilon=0.9,
        end_epsilon=0.001,
        epsilon_decay_steps=1000,
        render=True,
    ):
        """
        Trains an agent on the given environment following the SARSA algorithm.
        Creates an agent and then loops as follows:
            1) Gather a number of episodes, storing the experience
            2) If time, train the agent with these experiences
            3) If time, evaluate the agent, using a low epsilon

        :param env: gym.env to train an agent on
        :param test_env: gym.env to test an agent on
        :param train_every: int, specifies to train after x steps
        :param eval_every: int, evaluates every x steps
        :param max_stpes: int, maximum number of steps to gather/train
        :param start_epsilon: float, epsilon to start with
        :param end_epsilon: float, epsilon to end with
        :param epsilon_decay_steps: int, number of steps over which to decay
        :param render: bool, if True, renders the environment during training
        :returns: trained agent of type BaseAgent
        """

        agent = self.create_agent(env)
        curr_epsilon = start_epsilon
        epsilon_decay = self.get_decay_value(
            start_epsilon, end_epsilon, epsilon_decay_steps
        )

        obs = env.reset()
        action = agent.act(obs, epsilon=curr_epsilon)

        for step in range(1, max_steps + 1):
            next_obs, reward, done, _ = env.step(action)
            next_action = agent.act(next_obs, epsilon=curr_epsilon)
            agent.store_step(obs, action, reward, next_obs, next_action, done)
            obs = next_obs

            if render:
                env.render()

            if self.time_to(train_every, step):
                agent.perform_training(gamma=self.gamma)
            curr_epsilon = max(end_epsilon, curr_epsilon - epsilon_decay)

            if self.time_to(eval_every, step):
                self.evaluate_agent(agent, test_env, end_epsilon)

            if done:
                obs = env.reset()
                action = agent.act(obs, epsilon=curr_epsilon)

            print("At step {}".format(step), end="\r")
        print("\nDone!")

        return agent

    def create_agent(self, env):
        """
        Given a specific environment, creates an SARSA agent specific for this
        environment. Can only handle discrete environments.

        :param env: gym.env to create an agent for
        :returns: SarsaAgent
        """

        if isinstance(env.action_space, gym.spaces.Discrete):
            return SarsaAgent(
                obs_dim=env.observation_space.shape[0],
                act_dim=env.action_space.n,
                hidden_sizes=[64],
            )

        raise ValueError("SARSA can only be used for discrete action spaces.")
