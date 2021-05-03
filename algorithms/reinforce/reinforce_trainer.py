from algorithms.base_trainer import BaseTrainer
from algorithms.reinforce.discrete_reinforce_agent import \
    DiscreteReinforceAgent


class ReinforceTrainer(BaseTrainer):
    """
    Helper class for training an agent using the REINFORCE algorithm.
    """

    def __init__(self, *, gamma=0.99):
        self.gamma = gamma

    def train_agent(self, *, env, max_episodes=1000):
        """
        Trains an agent on the given environment following the REINFORCE algorithm.

        :param env: gym.env to train an agent on
        :returns: trained agent of type BaseAgent
        """

        agent = self.create_agent(env)

        for episode in range(max_episodes):
            obs = env.reset()
            agent.episode_reset()
            done = False

            episode_return = 0.0
            while not done:
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                episode_return += reward
                env.render()

                agent.store_step(reward)

            loss = agent.train(gamma=self.gamma)
            print(
                "Episode {} -- loss={:.2f} -- return={}".format(
                    episode, loss, episode_return
                )
            )

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
