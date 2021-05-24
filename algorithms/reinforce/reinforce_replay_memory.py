class ReinforceReplayMemory:
    """
    Replay Memory for when training an agent using the REINFORCE algorithm.
    As REINFORCE is an on-policy algorithm, only experiences gathered
    after the last update can be remembered.
    Furthermore, REINFORCE requires full episodes to make an update, as it
    uses Monte Carlo Sampling to estime the policy gradient.

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
        self.stored_episodes = {key: [] for key in self.experience_keys}
        self.current_episode = {key: [] for key in self.experience_keys}

    def store_experience(self, experience):
        """
        Stores the experience of one step in the memory. Checks if this step
        ended the episode, and if this is the case, stores the episode and
        reinitializes the current episode buffer.

        :param experience: dict with key-value pairs for all experience_keys
        """

        for key, value in experience.items():
            self.current_episode[key].append(value)

        if experience["dones"]:
            for key in self.experience_keys:
                self.stored_episodes[key].append(self.current_episode[key])
            self.current_episode = {key: [] for key in self.experience_keys}

    def sample(self):
        """
        Samples the experience from memory, consisting of all fully collected
        episodes up until now.

        :returns: dict with for every key their values for all episodes
        """
        return self.stored_episodes
