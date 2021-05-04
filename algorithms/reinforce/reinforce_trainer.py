import matplotlib.pyplot as plt
import seaborn as sns
from algorithms.base_trainer import BaseTrainer
from algorithms.reinforce.discrete_reinforce_agent import \
    DiscreteReinforceAgent

sns.set_style("darkgrid")


class ReinforceTrainer(BaseTrainer):
    """
    Helper class for training an agent using the REINFORCE algorithm.
    """

    def __init__(self, *, gamma=0.99):
        self.gamma = gamma

    def train_agent(self, *, env, max_episodes=1000, center_returns=True, render=False):
        """
        Trains an agent on the given environment following the REINFORCE algorithm.

        :param env: gym.env to train an agent on
        :param max_episodes: int, maximum number of episodes to gather/train on
        :param center_returns: bool, whether or not to apply mean baseline during training
        :param render: bool, whether or not to render the environment during training
        :returns: trained agent of type BaseAgent
        """

        agent = self.create_agent(env)

        episode_returns = []
        for episode in range(max_episodes):
            obs = env.reset()
            agent.episode_reset()
            done = False

            episode_return = 0.0
            while not done:
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                episode_return += reward
                agent.store_step(reward)

                if render:
                    env.render()

            agent.train(gamma=self.gamma, center_returns=center_returns)
            print("Episode {} -- return={}".format(episode, episode_return))
            episode_returns.append(episode_return)

        sns.lineplot(x=list(range(max_episodes)), y=episode_returns)
        plt.show()

    def create_agent(self, env):
        """
        Given a specific environment, creates an agent specific for this environment.

        :param env: gym.env to create an agent for
        :returns: DiscreteReinforceAgent
        """

        return DiscreteReinforceAgent(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.n,
            hidden_sizes=[64],
        )
