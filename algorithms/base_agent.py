from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Base class for an agent which should be able to act in an environment.
    It describes the API an agent should have.
    """

    @abstractmethod
    def act(self, observation, *args, **kwargs):
        """
        Should determine an action from the current observation according to the way
        the agent is defined.

        :param observation: np.ndarray with the observation of the environment
        """
        raise NotImplementedError
