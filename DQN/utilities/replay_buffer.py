import random
import numpy as np

from collections import deque, namedtuple
from sum_tree import SumTree

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "terminal"])


class ReplayBuffer:
    def __init__(self, args):
        self.memory = deque([], maxlen=args.buffer_capacity)
        self.batch_size = args.batch_size

    def store_transition(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        # return a list including k elements
        return random.sample(self.memory, k=self.batch_size)

    @property
    def current_size(self):
        return len(self.memory)


class N_Steps_ReplayBuffer:
    def __init__(self, args):
        self.memory = deque([], maxlen=args.buffer_capacity)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.n_steps = args.n_steps
        self.n_steps_deque = deque([], maxlen=self.n_steps)

    def store_transition(self, *args):
        """
        store n_steps transitions in n_steps_deque and store the first and last transition in memory
        the reward stored in memory is cumulative n_steps reward
        """
        self.n_steps_deque.append(Transition(*args))
        if len(self.n_steps_deque) == self.n_steps:
            state, action, n_steps_reward, next_state, terminal = self.get_n_steps_transition()
            self.memory.append(Transition(state, action, n_steps_reward, next_state, terminal))

    def get_n_steps_transition(self):
        state, action = self.n_steps_deque[0].state, self.n_steps_deque[0].action
        n_steps_reward = 0
        # state, action = self.n_steps_deque[0][:2]
        next_state, terminal = self.n_steps_deque[-1].next_state, self.n_steps_deque[-1].terminal
        # n_steps_reward = 0
        for i in reversed(range(self.n_steps)):
            r, s_, ter, d = self.n_steps_deque[i][2:]
            n_steps_reward = r + self.gamma * (1 - d) * n_steps_reward
            if d:
                next_state, terminal = s_, ter

        return state, action, n_steps_reward, next_state, terminal

    def sample(self):
        # return a list including k elements
        return random.sample(self.memory, k=self.batch_size)

    @property
    def current_size(self):
        return len(self.memory)


class Prioritized_ReplayBuffer:
    def __init__(self, args):
        self.memory = deque([], maxlen=args.buffer_capacity)
        self.sum_tree = SumTree(args.buffer_capacity)
        self.batch_size = args.batch_size
        self.max_train_steps = args.max_train_steps
        self.alpha = args.alpha
        self.beta_init = args.beta_init
        self.beta = args.beta_init
        self.count = 0
        self.all_count = 0
        self.buffer_capacity = args.buffer_capacity

    def store_transition(self, *args):
        if self.all_count < self.buffer_capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.count] = Transition(*args)
        # Avoid TD error being zero
        priority = 1.0 if self.current_size == 0 else self.sum_tree.priority_max
        # Update sum tree node priority
        self.sum_tree.update(data_index=self.count, priority=priority)
        self.count = (self.count + 1) % self.buffer_capacity
        self.all_count += 1

    def sample(self, total_steps):
        batch_index, IS_weight = self.sum_tree.get_batch_index(current_size=self.current_size,
                                                               batch_size=self.batch_size, beta=self.beta)
        self.beta = self.beta_init + (1 - self.beta_init) * (total_steps / self.max_train_steps)
        batch = self.memory[batch_index]

        return batch, batch_index, IS_weight

    def update_batch_priorities(self, batch_index, td_errors):
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update(data_index=index, priority=priority)

    @property
    def current_size(self):
        return len(self.memory)


class N_Steps_Prioritized_ReplayBuffer(object):
    def __init__(self, args):
        self.memory = deque([], maxlen=args.buffer_capacity)
        self.max_train_steps = args.max_train_steps
        self.alpha = args.alpha
        self.beta_init = args.beta_init
        self.beta = args.beta_init
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.sum_tree = SumTree(self.buffer_capacity)
        self.n_steps = args.n_steps
        self.n_steps_deque = deque(maxlen=self.n_steps)
        self.all_count = 0
        self.count = 0

    def store_transition(self, *args):
        self.n_steps_deque.append(Transition(*args))
        if len(self.n_steps_deque) == self.n_steps:
            if self.all_count < self.buffer_capacity:
                state, action, n_steps_reward, next_state, terminal = self.get_n_steps_transition()
                self.memory.append(Transition(state, action, n_steps_reward, next_state, terminal))
            else:
                state, action, n_steps_reward, next_state, terminal = self.get_n_steps_transition()
                self.memory[self.count] = Transition(state, action, n_steps_reward, next_state, terminal)

            priority = 1.0 if self.current_size == 0 else self.sum_tree.priority_max
            self.sum_tree.update(data_index=self.count, priority=priority)
            self.count = (self.count + 1) % self.buffer_capacity
            self.all_count += 1

    def sample(self, total_steps):
        batch_index, IS_weight = self.sum_tree.get_batch_index(current_size=self.current_size,
                                                               batch_size=self.batch_size, beta=self.beta)
        self.beta = self.beta_init + (1 - self.beta_init) * (total_steps / self.max_train_steps)  # beta：beta_init->1.0
        batch = self.memory[batch_index]
        return batch, batch_index, IS_weight

    def get_n_steps_transition(self):
        state, action = self.n_steps_deque[0][:2]
        next_state, terminal = self.n_steps_deque[-1][3:5]
        n_steps_reward = 0
        for i in reversed(range(self.n_steps)):
            r, s_, ter, d = self.n_steps_deque[i][2:]
            n_steps_reward = r + self.gamma * (1 - d) * n_steps_reward
            if d:
                next_state, terminal = s_, ter

        return state, action, n_steps_reward, next_state, terminal

    def update_batch_priorities(self, batch_index, td_errors):
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update(data_index=index, priority=priority)

    @property
    def current_size(self):
        return len(self.memory)
