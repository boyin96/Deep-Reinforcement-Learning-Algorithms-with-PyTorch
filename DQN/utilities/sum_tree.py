import numpy as np
import torch


class SumTree:

    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity
        self.tree_capacity = 2 * buffer_capacity - 1
        self.tree = np.zeros(self.tree_capacity)

    def update(self, data_index, priority):
        """"Update priority of tree"""
        # data_index represent the index of transition in buffer
        # tree_index represent the index of data in sumtree
        tree_index = data_index + self.buffer_capacity - 1
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # then propagate the change through the tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_index(self, v):
        """"Return the index of data in buffer corresponding to the specific leaf node and priority"""
        parent_idx = 0
        while True:
            child_left_idx = 2 * parent_idx + 1
            child_right_idx = child_left_idx + 1
            if child_left_idx >= self.tree_capacity:
                tree_index = parent_idx
                break
            else:
                if v <= self.tree[child_left_idx]:
                    parent_idx = child_left_idx
                else:
                    v -= self.tree[child_left_idx]
                    parent_idx = child_right_idx

        data_index = tree_index - self.buffer_capacity + 1

        return data_index, self.tree[tree_index]

    def get_batch_index(self, current_size, batch_size, beta):
        """Sample batch transitions and return data index and its learning rate factor"""
        batch_index = np.zeros(batch_size, dtype=np.long)
        IS_weight = torch.zeros(batch_size, dtype=torch.float32)
        segment = self.priority_sum / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            index, priority = self.get_index(v)
            batch_index[i] = index
            prob = priority / self.priority_sum
            IS_weight[i] = (current_size * prob) ** (-beta)

        IS_weight /= IS_weight.max()

        return batch_index, IS_weight

    @property
    def priority_sum(self):
        return self.tree[0]

    @property
    def priority_max(self):
        return self.tree[self.buffer_capacity - 1:].max()
