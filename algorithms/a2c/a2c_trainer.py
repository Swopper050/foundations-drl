import gym

from algorithms.a2c import A2CAgent
from algorithms.base_trainer import BaseTrainer


class A2CTrainer(BaseTrainer):
    """
    Helper class for training an A2CAgent. Implements the main
    training loop of the A2C algorithm.
    """

    def train_agent(
        self,
        *,
        env,
        test_env,
        max_episodes=int(1e3),
        train_every=5,
        render=True,
    ):
        """
        Trains an agent on the given environment according to A2C.
        The steps consist of the following:
            1) Create an agent given the environment
            2) Gather steps from a number of episodes
            3) Train the agent with these episodes

        :param env: gym.env to train an agent on
        :param test_env: gym.env to test an agent on
        :param max_episodes: int, maximum number of episodes to gather/train on
        :param train_every: int, specifies to train after x episodes
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
                agent.perform_training()
                self.evaluate_agent(agent, test_env, n_eval_episodes=5)
            print(
                "Episode {}/{} -- return={}".format(
                    episode, max_episodes, episode_return
                ),
                end="\r",
            )

        return agent

    def create_agent(self, env):
        """
        Given a specific environment, creates an agent specific for this
        environment. It checks whether the agent requires continuous or
        discrete actions, and then creates an agent accordingly.

        :param env: gym.env to create an agent for
        :returns: A2CAgent
        """

        if isinstance(env.action_space, gym.spaces.Box):
            discrete = False
            act_dim = env.action_space.shape[0]
        elif isinstance(env.action_space, gym.spaces.Discrete):
            discrete = True
            act_dim = env.action_space.n
        else:
            raise ValueError("No known action space for this environment")

        return A2CAgent(
            obs_dim=env.observation_space.shape[0],
            act_dim=act_dim,
            hidden_sizes=[64],
            discrete=discrete,
        )
