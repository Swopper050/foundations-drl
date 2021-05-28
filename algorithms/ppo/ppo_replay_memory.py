import torch


class PPOReplayMemory:
    """
    Replay memory for the PPO algorithm. This algorithm remembers tuples:
        (state, action, reward, next_state, done)

    These are collected for a specific number of episodes, which are all
    returned when sampled. PPO is an on-policy algorithm, so it needs to
    be reset every time it is sampled.

    It is the same as for the A2C algorithm.
    """

    def __init__(self, *, experience_keys):
        """
        :param experience_keys: list with strings, specifying the keys
                                of the experiences to be stored
        """
        self.size = 0
        self.experience_keys = experience_keys
        self.reset()

    def reset(self):
        """
        Resets the memory, should be called at the end of a trainig update.
        All experiences for all episodes are discarded.
        """
        self.memory = {key: [] for key in self.experience_keys}

    def store_experience(self, experience):
        """
        Stores the experience of one step in the memory.

        :param experience: dict with key-value pairs for all experience_keys
        """

        for key, value in experience.items():
            self.memory[key].append(value)

    def sample(self):
        """
        Samples the experience from memory, consisting of all fully collected
        episodes up until now.

        :returns: dict with for every key their values for all episodes
        """
        return {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in self.memory.items()
        }
