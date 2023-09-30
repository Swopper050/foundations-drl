import gymnasium as gym
import torch

from algorithms.base_trainer import BaseTrainer
from algorithms.ppo import PPOAgent


class PPOTrainer(BaseTrainer):
    """
    Helper class for training an PPOAgent. Implements the main
    training loop of the PPO algorithm.
    """

    def train_agent(
        self,
        *,
        env,
        test_env,
        save_name,
        max_episodes=int(1e4),
        train_every=5,
        n_updates=80,
        render=True,
    ):
        """
        Trains an agent on the given environment according to PPO.
        The steps consist of the following:
            1) Create an agent given the environment
            2) Gather steps from a number of episodes
            3) Train the agent with these episodes

        :param env: gym.env to train an agent on
        :param test_env: gym.env to test an agent on
        :param save_name: str, name to save the agent under
        :param max_episodes: int, maximum number of episodes to gather/train on
        :param train_every: int, specifies to train after x episodes
        :param n_updates: int, number of updates per training moment
        :param render: bool, if True, renders the environment while training
        :returns: trained agent of type BaseAgent
        """

        agent = self.create_agent(env)

        n_steps_seen = 0
        for episode in range(1, max_episodes + 1):
            obs, _ = env.reset()
            done = False

            episode_return = 0.0
            while not done:
                action = agent.act(obs, deterministic=False)
                next_obs, reward, done, _, _ = env.step(action)
                n_steps_seen += 1
                episode_return += reward
                agent.store_step(obs, action, reward, next_obs, done)
                obs = next_obs

                if render:
                    env.render()

            if episode % train_every == 0:
                agent.perform_training(n_updates=n_updates)
                self.evaluate_agent(agent, test_env, n_eval_episodes=5)
                torch.save(agent, f"saved_agents/{save_name}")

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
        :returns: PPOAgent
        """

        if isinstance(env.action_space, gym.spaces.Box):
            discrete = False
            act_dim = env.action_space.shape[0]
        elif isinstance(env.action_space, gym.spaces.Discrete):
            discrete = True
            act_dim = env.action_space.n
        else:
            raise ValueError("No known action space for this environment")

        return PPOAgent(
            obs_dim=env.observation_space.shape[0],
            act_dim=act_dim,
            hidden_sizes=[128, 64],
            discrete=discrete,
        )
