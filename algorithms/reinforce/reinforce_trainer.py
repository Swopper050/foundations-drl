import gym
import matplotlib.pyplot as plt
import seaborn as sns
from algorithms.base_trainer import BaseTrainer
from algorithms.reinforce import (ContinuousReinforceAgent,
                                  DiscreteReinforceAgent)

sns.set_style("darkgrid")


class ReinforceTrainer(BaseTrainer):
    """
    Helper class for training an agent using the REINFORCE algorithm.
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
        show_results=False
    ):
        """
        Trains an agent on the given environment following the REINFORCE algorithm.

        :param env: gym.env to train an agent on
        :param test_env: gym.env to test an agent on
        :param train_every: int, specifies to train after x episodes
        :param max_episodes: int, maximum number of episodes to gather/train on
        :param center_returns: bool, whether or not to apply mean baseline during training
        :param render: bool, whether or not to render the environment during training
        :param show_results: bool, whether or not to show the results after training
        :returns: trained agent of type BaseAgent
        """

        agent = self.create_agent(env)

        episode_returns = []
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
                agent.perform_training(gamma=self.gamma, center_returns=center_returns)

            episode_returns.append(episode_return)
            print("Episode {} -- return={}".format(episode, episode_return))

        if show_results:
            sns.lineplot(x=list(range(max_episodes)), y=episode_returns)
            plt.show()

        return agent

    def create_agent(self, env):
        """
        Given a specific environment, creates an agent specific for this environment.
        It checks whether the agent requires continuous or discrete actions, and then
        creates an agent accordingly

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
