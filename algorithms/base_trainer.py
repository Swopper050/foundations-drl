from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def train_agent(self, *, env):
        """
        Method that creates and trains an agent according to the given algorithm. Should
        return an trained agent of type BaseAgent.

        :param env: environment to train on
        """
        raise NotImplementedError
