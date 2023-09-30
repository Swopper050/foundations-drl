import gymnasium as gym
import torch

from algorithms.base_trainer import BaseTrainer
from algorithms.dqn.dqn_agent import DQNAgent


class DQNTrainer(BaseTrainer):
    """
    Helper class for training an agent using the DQN algorithm.
    Basically same as the Vanilla DQN trainer, implements the
    same training loop.
    """

    def __init__(self, *, gamma=0.99):
        self.gamma = gamma
        self.obs = None

    def train_agent(
        self,
        *,
        env,
        test_env,
        save_name,
        max_steps=10000,
        n_init_steps=32,
        train_every=4,
        eval_every=500,
        n_batches_per_train_step=8,
        n_updates_per_batch=4,
        batch_size=32,
        start_tau=5.0,
        end_tau=0.5,
        tau_decay_steps=4000,
        render=True,
    ):
        """
        Trains an agent on the given environment following the Vanilla DQN
        algorithm. First creates an agent, then follows the loop:
            1) Gather a number of experiences
            2) Sample a number of batches
            3) Perform a number of updates for every batch
            4) Decreases tau
            5) When time, evaluate the agent

        :param env: gym.env to train an agent on
        :param test_env: gym.env to test an agent on
        :param save_name: str, name to save the agent under
        :param max_steps: int, maximum number of steps to gather for training
        :param n_init_steps: int, initial number of experiences to gather
        :param train_every: int, specifies to train after x steps
        :param eval_every: int, evaluates every x steps
        :param n_batches_per_train_step: int, number of batches per train step
        :param n_updates_per_batch: int, number of updates per batch
        :param batch_size: int, size of batches to sample
        :param start_tau: float, start value of tau (Boltzmann policy)
        :param end_tau: float, end value of tau
        :param tau_decay_steps: int, number of steps to take to decrease tau
        :param render: bool, if True, renders the environment during training
        :returns: trained agent of type BaseAgent
        """

        # Create agent and initialize tau
        agent = self.create_agent(env)
        curr_tau = start_tau
        tau_decay = self.get_decay_value(start_tau, end_tau, tau_decay_steps)

        # Gather initial experience
        self.gather_experience(
            env, agent, curr_tau, n_steps_to_gather=n_init_steps, render=render
        )

        loops = int((max_steps - n_init_steps) / train_every)
        for loop in range(1, loops):
            self.gather_experience(
                env,
                agent,
                curr_tau,
                n_steps_to_gather=train_every,
                render=render,
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
                torch.save(agent, f"saved_agents/{save_name}")

            print(
                "N frames seen {}".format(loop * train_every + n_init_steps),
                end="\r",
            )
        print("\nDone!")

        return agent

    def gather_experience(
        self, env, agent, curr_tau, *, n_steps_to_gather, render
    ):
        """
        Gathers a number of experiences in the given environment.

        :param env: gym.env to gather experiences from
        :param agent: agent to use to act in the env
        :param curr_tau: float, current tau for action selection
        :param n_steps_to_gather: int, number of steps to gather
        :param render: bool, if True, renders the environment
        """

        if self.obs is None:
            self.obs, _ = env.reset()

        n_steps_seen = 0
        while n_steps_seen < n_steps_to_gather:
            action = agent.act(self.obs, tau=curr_tau)
            next_obs, reward, done, _, _ = env.step(action)
            agent.store_step(self.obs, action, reward, next_obs, done)
            self.obs = next_obs
            if render:
                env.render()

            if done:
                self.obs, _ = env.reset()

            n_steps_seen += 1

    def create_agent(self, env):
        """
        Given a specific environment, creates an DQN agent specific
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
