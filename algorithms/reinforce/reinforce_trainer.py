import gym
import seaborn as sns

from algorithms.base_trainer import BaseTrainer
from algorithms.reinforce import (
    ContinuousReinforceAgent,
    DiscreteReinforceAgent,
)

sns.set_style("darkgrid")


class ReinforceTrainer(BaseTrainer):
    """
    Helper class for training an agent using the REINFORCE algorithm.
    Implements the main training loop for training a REINFORCE agent.
    """

    def __init__(self, *, gamma=0.99):
        self.gamma = gamma

    def train_agent(
        self,
        *,
        env,
        test_env,
        train_every=1,
        max_episodes=1000,
        center_returns=True,
        render=True,
    ):
        """
        Trains an agent on the given environment according to REINFORCE.
        The steps consist of the following:
            1) Create an agent given the environment
            2) Gather 'train_every' episodes of experience
            3) Train the agent with these episodes

        :param env: gym.env to train an agent on
        :param test_env: gym.env to test an agent on
        :param train_every: int, specifies to train after x episodes
        :param max_episodes: int, maximum number of episodes to gather/train on
        :param center_returns: bool, if True, centers the returns for training
        :param render: bool, if True, renders the environment while training
        :returns: trained agent of type BaseAgent
        """

        agent = self.create_agent(env)

        for episode in range(1, max_episodes + 1):
            obs = env.reset()
            done = False

            episode_return = 0.0
            while not done:
                action = agent.act(obs, deterministic=False)
                next_obs, reward, done, _ = env.step(action)
                episode_return += reward
                agent.store_step(obs, action, reward, next_obs, done)
                obs = next_obs

                if render:
                    env.render()

            if episode % train_every == 0:
                agent.perform_training(
                    gamma=self.gamma, center_returns=center_returns
                )

            print("Episode {} -- return={}".format(episode, episode_return))
        return agent

    def create_agent(self, env):
        """
        Given a specific environment, creates an agent specific for this
        environment. It checks whether the agent requires continuous or
        discrete actions, and then creates an agent accordingly.

        :param env: gym.env to create an agent for
        :returns: ContinuousReinforceAgent or DiscreteReinforceAgent
        """

        if isinstance(env.action_space, gym.spaces.Box):
            return ContinuousReinforceAgent(
                obs_dim=env.observation_space.shape[0],
                act_dim=env.action_space.shape[0],
                hidden_sizes=[64],
            )

        if isinstance(env.action_space, gym.spaces.Discrete):
            return DiscreteReinforceAgent(
                obs_dim=env.observation_space.shape[0],
                act_dim=env.action_space.n,
                hidden_sizes=[64],
            )

        raise ValueError("No known action space for this environment")
