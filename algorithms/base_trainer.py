from abc import ABC, abstractmethod

import numpy as np


class BaseTrainer(ABC):
    """
    Base class for a trainer. Should implement the main training loop of
    the algorithm.
    """

    @abstractmethod
    def train_agent(self, *, env):
        """
        Method that creates and trains an agent according to the given
        algorithm. Should return an trained agent of type BaseAgent.

        :param env: environment to train on
        """
        raise NotImplementedError

    @staticmethod
    def time_to(every, step):
        """
        Can be used to check if it is 'time to' do something.

        :param every: int, something that should be done every x somethings
        :param step: int, current x
        :returns: True it is time to do the thing, else False
        (best docstrings ever)
        """
        return (step % every) == 0

    @staticmethod
    def get_decay_value(start, end, steps):
        """
        Returns the decay value needed to reach the end value in a specific
        number of steps.

        :param start: float, start value
        :param end: float, end value
        :param steps: int, n steps to take for the decay
        :returns: float, value to subtract every step
        """
        return (start - end) / steps

    def evaluate_agent(self, agent, env, n_eval_episodes=10):
        """
        Evaluates the performance of the agent on the given environment.
        Performs a number of episodes and prints the mean reward.

        :param agent: agent to use for acting
        :param env: gym.env to test on
        :param n_eval_episodes: int, number of episodes for evaluating
        """
        returns = []
        for _ in range(n_eval_episodes):
            episode_return = 0.0
            obs = env.reset()
            done = False
            while not done:
                obs, reward, done, _ = env.step(agent.act(obs))
                episode_return += reward
            returns.append(episode_return)
        print("Evaluated agent. Mean return: {}".format(np.mean(returns)))
