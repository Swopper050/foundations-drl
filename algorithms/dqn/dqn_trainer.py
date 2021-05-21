import gym
import numpy as np
import seaborn as sns
from algorithms.base_trainer import BaseTrainer
from algorithms.dqn.dqn_agent import DQNAgent

sns.set_style("darkgrid")


class DQNTrainer(BaseTrainer):
    """
    Helper class for training an agent using the DQN algorithm.
    """

    def __init__(self, *, gamma=0.99):
        self.gamma = gamma
        self.obs = None

    def train_agent(
        self,
        *,
        env,
        test_env,
        max_steps=10000,
        n_init_steps=32,
        train_every=4,
        eval_every=500,
        n_batches_per_train_step=4,
        n_updates_per_batch=8,
        batch_size=32,
        start_tau=5.0,
        end_tau=0.5,
        tau_decay_steps=4000,
        render=True,
    ):
        """
        Trains an agent on the given environment following the DQN algorithm.

        :param env: gym.env to train an agent on
        :param test_env: gym.env to test an agent on
        :param train_every: int, specifies to train after x steps
        :param eval_every: int, evaluates every x steps
        :param max_stpes: int, maximum number of steps to gather/train
        :param render: bool, whether or not to render the environment during training
        :param show_results: bool, whether or not to show the results after training
        :returns: trained agent of type BaseAgent
        """

        agent = self.create_agent(env)
        curr_tau = start_tau
        tau_decay = self.get_decay_value(start_tau, end_tau, tau_decay_steps)

        self.gather_experience(
            env, agent, curr_tau, n_steps_to_gather=n_init_steps, render=render
        )
        loops = int((max_steps - n_init_steps) / train_every)
        for loop in range(1, loops):
            self.gather_experience(
                env, agent, curr_tau, n_steps_to_gather=train_every, render=render
            )
            for _ in range(n_batches_per_train_step):
                agent.perform_training(
                    gamma=self.gamma,
                    batch_size=batch_size,
                    n_updates=n_updates_per_batch,
                )
            curr_tau = max(end_tau, curr_tau - (tau_decay * train_every))

            if self.time_to(int(eval_every / train_every), loop):
                self.evaluate_agent(agent, test_env)

            print(
                "N frames seen {}".format(loop * train_every + n_init_steps), end="\r"
            )
        print("\nDone!")

        return agent

    def gather_experience(self, env, agent, curr_tau, *, n_steps_to_gather, render):

        if self.obs is None:
            self.obs = env.reset()

        n_steps_seen = 0
        while n_steps_seen < n_steps_to_gather:
            action = agent.act(self.obs, tau=curr_tau)
            next_obs, reward, done, _ = env.step(action)
            agent.store_step(self.obs, action, reward, next_obs, done)
            self.obs = next_obs
            if render:
                env.render()

            if done:
                self.obs = env.reset()

            n_steps_seen += 1

    @staticmethod
    def time_to(every, step):
        return (step % every) == 0

    @staticmethod
    def get_decay_value(start, end, steps):
        return (start - end) / steps

    def evaluate_agent(self, agent, env, n_eval_episodes=10):
        """
        Evaluates the performance of the agent.
        """
        returns = []
        for _ in range(n_eval_episodes):
            ret = 0.0
            obs = env.reset()
            done = False
            while not done:
                obs, reward, done, _ = env.step(agent.act(obs, tau=0.001))
                ret += reward
            returns.append(ret)
        print("Evaluated agent. Mean return: {}".format(np.mean(returns)))

    def create_agent(self, env):
        """
        Given a specific environment, creates an vanilla DQN agent specific
        for this environment. Can only handle discrete environments.

        :param env: gym.env to create an agent for
        :returns: VanillaDQNAgent
        """

        if isinstance(env.action_space, gym.spaces.Discrete):
            return DQNAgent(
                obs_dim=env.observation_space.shape[0],
                act_dim=env.action_space.n,
                hidden_sizes=[64],
            )

        raise ValueError("DQN can only be used for discrete action spaces.")
