import random
from collections import deque

import numpy as np
import torch


class VanillaDQNReplayMemory:
    """
    Replay Memory used by the Vanilla DQN algorithm. Stores experiences
    individually, and when sampled returns a batch of torch.tensors with
    experiences. Uses a FIFO deque structure to store experiences, which
    deletes the oldest experiences for new experiences when memory is full.

    Sampling is randomly.
    """

    def __init__(self, *, max_size, experience_keys):
        """
        :param max_size: int, max number of experience to store in the memory
        :param experience_keys: list with strings, specifying the keys
                                of the experiences to be stored
        """
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)
        self.experience_keys = experience_keys

    def store_experience(self, experience):
        """
        Store the experience of this specific step into memory.

        :param experience: dict with information about the step
        """
        self.memory.append(experience)

    def sample(self, *, batch_size):
        """
        Retrieve a batch of experiences from memory.

        :param batch_size: int, number of experiences to sample from memory
        """
        batch = random.sample(self.memory, batch_size)

        return {
            key: torch.as_tensor(
                np.stack([experience[key] for experience in batch]),
                dtype=torch.float32,
            )
            for key in batch[0].keys()
        }
