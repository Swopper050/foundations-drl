class SarsaReplayMemory:
    """
    Replay Memory used by the SARSA algorithm. As SARSA is an on-policy
    algorithm, it can not use experiences from before the previous update.

    It stores experiences individually from their episode, i.e. simply a
    dictionary with experiences for every step in the memory.
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
        Samples the experience from memory, consisting of all collected
        episodes up until now.

        :returns: dict with for every key their values currently in memory
        """
        return self.memory
