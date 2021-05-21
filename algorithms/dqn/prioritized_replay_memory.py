import random

import numpy as np
import torch

INIT_ERROR = 100000
""" Initial error an experience starts with. It is high such that every
experience is sampled at least once. """


class PrioritizedReplayMemory:
    def __init__(self, *, max_size, experience_keys):
        """
        :param max_size: int, max number of experience to store in the memory
        :param experience_keys: list with strings, specifying the keys
                                of the experiences to be stored
        """
        self.max_size = max_size
        self.memory = np.array([None for _ in range(max_size)])
        self.head = 0
        self.experience_keys = experience_keys + ["priority"]
        self.sum_tree = SumTree(size=max_size)

    def store_experience(self, experience):
        """
        Store the experience of this specific step into memory.

        :param experience: dict with information about the step
        """
        self.sum_tree.set_priority(i=self.head, error=INIT_ERROR)
        self.memory[self.head] = experience
        self.head += 1
        if self.head == self.max_size:
            self.head = 0

    def sample(self, *, batch_size):
        """
        Retrieve a batch of experiences from memory.

        :param batch_size: int, number of experiences to sample from memory
        """
        self.batch_indices = self.get_batch_indices(batch_size)
        batch = self.memory[self.batch_indices]

        return {
            key: torch.as_tensor(
                np.stack([experience[key] for experience in batch]),
                dtype=torch.float32,
            )
            for key in batch[0].keys()
        }

    def get_batch_indices(self, batch_size):
        indices = []
        for _ in range(batch_size):
            x = random.uniform(0, self.sum_tree.root.value)
            (idx, _) = self.sum_tree.get_index_from_value(x)
            indices.append(idx)
        return indices

    def update_priorities(self, batch_errors):
        for idx, error in zip(self.batch_indices, batch_errors):
            self.sum_tree.set_priority(i=idx, error=error)


class SumTree:
    def __init__(self, *, size, alpha=0.6):
        self.size = size
        self.alpha = alpha
        self.construct_empty_tree()

    def set_priority(self, *, i, error):
        priority = np.power(error + 0.0001, self.alpha)
        self.leaf_nodes[i].set_and_update_parent(priority)

    def get_index_from_value(self, x):
        node = self.root
        while not node.is_leaf_node():
            if x <= node.child1.value:
                node = node.child1
            else:
                x -= node.child1.value
                node = node.child2
        return node.index, node.value

    def construct_empty_tree(self):
        self.leaf_nodes = [
            Node(value=0, index=i, parent=None, child1=None, child2=None)
            for i in range(self.size)
        ]
        layer_nodes = self.leaf_nodes
        while len(layer_nodes) != 1:
            layer_nodes = self.construct_parents(layer_nodes)
        self.root = layer_nodes[0]

    def construct_parents(self, layer_nodes):
        parent_nodes = []
        n_nodes = len(layer_nodes)
        for i in range(0, n_nodes, 2):
            if i + 1 == n_nodes:
                parent_nodes.append(layer_nodes[i])
            else:
                child1 = layer_nodes[i]
                child2 = layer_nodes[i + 1]
                parent_node = Node(
                    value=0, index=None, parent=None, child1=child1, child2=child2
                )
                child1.parent = child2.parent = parent_node
                parent_nodes.append(parent_node)

        return parent_nodes


class Node:
    def __init__(self, *, value, index, parent, child1, child2):
        self.value = value
        self.index = index
        self.parent = parent
        self.child1 = child1
        self.child2 = child2

    def is_leaf_node(self):
        return self.child1 is None and self.child2 is None

    def set_and_update_parent(self, new_value):
        if not self.is_leaf_node():
            raise ValueError("Can only update values of leaf nodes")
        self.value = new_value
        self.parent.recalculate_value()

    def recalculate_value(self):
        self.value = self.child1.value + self.child2.value
        if self.parent:
            self.parent.recalculate_value()
