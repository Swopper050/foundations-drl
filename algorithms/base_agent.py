from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Base class for an agent which should be able to act in an environment.
    It describes the API an agent should have.
    """

    @abstractmethod
    def act(self, observation, *args, **kwargs):
        """
        Should determine an action from the current observation according to
        the way the agent is defined. By default, it should act
        deterministically, i.e. exploiting its knowledge.

        :param observation: np.ndarray with the observation of the environment
        """
        raise NotImplementedError

    @abstractmethod
    def store_step(self, *args, **kwargs):
        """
        Should store the experience of a step in the memory of the agent, such
        that it can train on it later.
        """
        raise NotImplementedError

    @abstractmethod
    def perform_training(self, *args, **kwargs):
        """
        Should perform the training of the agent. Called by a trainer at
        a specific moment during the training loop.
        """

    def episode_reset(self):
        """
        Called before the start of an episode, can be overridden to do
        meaningful stuff agent-specific if needed.
        """
        pass
