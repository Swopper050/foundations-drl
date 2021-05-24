import random

import numpy as np
import torch

INIT_ERROR = 100000
""" Initial error an experience starts with. It is high such that every
experience is sampled at least once. """


class PrioritizedReplayMemory:
    """
    Implements Prioritized Experience Replay (PER). Based on the idea that
    some experiences are more informative than others, and thus should be
    sampled more often. Therefore, next to all the usual values, it remembers
    the 'priority' of the experience. This is based on the current error of
    the estimate as follows:
        priority = pow(error + constant, alpha)

    The constant handles 0 errors, the alpha determines the weight, i.e. how
    much prioritizing is done. The probability of sampling an experience is
    as follows:
        prob = priority / sum(priorities)

    Initially, all samples get the same priority, being very high, such that
    all experiences are sampled at least once.

    Furthermore, priorities change during training, as (hopefully) their errors
    decrease during training. Thus priorities need to be updated.

    Sampling is done using a SumTree by picking a random number between 0 and
    the sum of priorities. Then the tree is traversed until a leaf node is
    reached, corresponding to an experience.
    """

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
        self.sum_tree = SumTree(n_leaf_nodes=max_size)

    def store_experience(self, experience):
        """
        Store the experience of this specific step into memory.

        :param experience: dict with information about the step
        """
        # Update the corresponding leaf node and give it the initial error
        self.sum_tree.set_priority(i=self.head, error=INIT_ERROR)
        # Insert the experience into memory
        self.memory[self.head] = experience

        # Move the head. If the max size is reached, reset the head
        self.head += 1
        if self.head == self.max_size:
            self.head = 0

    def sample(self, *, batch_size):
        """
        Retrieve a batch of experiences from memory. Samples indices of
        experiences in memory using the SumTree.
        The batch indices are remembered such that the priorities of the
        corresponding experiences can be updated later on.

        :param batch_size: int, number of experiences to sample from memory
        :returns: dict with a batch of experiences
        """
        self.batch_indices = self.sample_batch_indices(batch_size)
        batch = self.memory[self.batch_indices]

        return {
            key: torch.as_tensor(
                np.stack([experience[key] for experience in batch]),
                dtype=torch.float32,
            )
            for key in batch[0].keys()
        }

    def sample_batch_indices(self, batch_size):
        """
        Samples a number of indices from the sum tree.

        :param batch_size: int, number of indices to sample
        :returns: list of indices
        """
        indices = []
        for _ in range(batch_size):
            x = random.uniform(0, self.sum_tree.root.value)
            (idx, _) = self.sum_tree.get_index_from_value(x)
            indices.append(idx)
        return indices

    def update_priorities(self, batch_errors):
        """
        Given the predictions errors of the sampled batch, updates their
        errors accordingly by updating their values in the SumTree.

        :param batch_errors: np.ndarray with errors for every experience
        """

        for idx, error in zip(self.batch_indices, batch_errors):
            self.sum_tree.set_priority(i=idx, error=error)


class SumTree:
    """
    Basic implementation of a SumTree. It has a structure where the parent
    is the sum of its children. Hence, something like:
            11
           /  \
          6    5
         / \
        4  2

    Sampling can be done by picking a number between 0 and 11 and traversing it
    down until a leaf node is reached. The number of leaf nodes should equal
    the maximum number of experiences that can be stored in memory. Then, every
    leaf node has an index corresponding to a specific experience in replay
    memory.
    """

    def __init__(self, *, n_leaf_nodes, alpha=0.6):
        """
        :param n_leaf_nodes: int, number of leaf nodes in the tree
        :param alpha: float, how much to prioritize
        """
        self.alpha = alpha
        self.construct_empty_tree(n_leaf_nodes)

    def set_priority(self, *, i, error):
        """
        Updates the priority of the ith leaf node, based on the given error.

        :param i: int, index of the leaf node to update
        :param error: float, error of the ith experience
        """
        priority = np.power(error + 0.0001, self.alpha)
        self.leaf_nodes[i].set_value_and_update_parent(priority)

    def get_index_from_value(self, x):
        """
        Performs the traversing given a value x. The algorithm works by
        repeatedly performing the following:
            1) Start at the root node
            2) If x is smaller than the value of child 1, move to child 1,
               else, subtract the value of child 1 and move to child 2.
            3) Check if this is a leaf node. If so return its index and value.
            4) Repeat until done.

        :param x: float, value to use while traversing
        :returns: index of the leaf node and the corresponding value
        """
        node = self.root
        while not node.is_leaf_node():
            if x <= node.child1.value:
                node = node.child1
            else:
                x -= node.child1.value
                node = node.child2
        return node.index, node.value

    def construct_empty_tree(self, n_leaf_nodes):
        """
        Constructs an empty SumTree with a specified number of leaf
        nodes. All values will be initialized with 0. Only the leaf
        nodes are remembered as list, and the root node. They
        are set as attributes.

        :param n_leaf_nodes: int, number of leaf nodes
        """
        self.leaf_nodes = [
            Node(value=0, index=i, parent=None, child1=None, child2=None)
            for i in range(n_leaf_nodes)
        ]
        layer_nodes = self.leaf_nodes
        while len(layer_nodes) != 1:
            layer_nodes = self._construct_parents(layer_nodes)
        self.root = layer_nodes[0]

    def _construct_parents(self, layer_nodes):
        """
        Given a 'layer' with nodes, constructs parents for these nodes.
        A parent will be a new node with as value the sum of two children.
        If the layer has an uneven number of nodes, the last node is moved
        up to the parent layer.

        :param layer_nodes: list with Nodes
        :returns: list with parents for the Nodes
        """

        parent_nodes = []
        n_nodes = len(layer_nodes)
        for i in range(0, n_nodes, 2):
            # If uneven, the last node will be added to the parent node layer
            if i + 1 == n_nodes:
                parent_nodes.append(layer_nodes[i])
            else:
                # Determine the childeren
                child1 = layer_nodes[i]
                child2 = layer_nodes[i + 1]
                # Construct a new parent
                parent_node = Node(
                    value=child1.value + child2.value,
                    index=None,
                    parent=None,
                    child1=child1,
                    child2=child2,
                )
                # Add the parent to the children and to the list
                child1.parent = child2.parent = parent_node
                parent_nodes.append(parent_node)

        return parent_nodes


class Node:
    """
    Simple node for a SumTree. Holds:
     - value
     - index (only for leaf nodes)
     - parent
     - child1
     - child2

    The index, only present for leaf nodes, corresponds to a specific
    experience in replay memory.
    """

    def __init__(self, *, value, index, parent, child1, child2):
        """
        :param value: float, value of the node
        :param index: int, index of the corresponding experience, else None
        :param parent: parent Node, only None for root node
        :param child1: left child Node, only None for leaf nodes
        :param child2: right child Node, only None for leaf nodes
        """
        self.value = value
        self.index = index
        self.parent = parent
        self.child1 = child1
        self.child2 = child2

    def is_leaf_node(self):
        """
        :returns: True if node is a leaf node, i.e. has no children
        """
        return self.child1 is None and self.child2 is None

    def set_value_and_update_parent(self, new_value):
        """
        Gives the node a new value (can only be done for leaf nodes) and then
        updates the value of all related nodes, i.e. its parent. This is called
        recursively up until the root node.

        :param new_value: float, new value for the leaf node
        :raises ValueError: if the node is not a leaf node
        """
        if not self.is_leaf_node():
            raise ValueError("Can only update values of leaf nodes")
        self.value = new_value
        self.parent.recalculate_value()

    def recalculate_value(self):
        """
        Recalculates the value based on the value of its children. Then,
        tells its parent to also recalculate its value.
        """
        self.value = self.child1.value + self.child2.value
        if self.parent:
            self.parent.recalculate_value()
