import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as n
import gym
from collections import deque
# import argparse
#
# parser = argparse.ArgumentParser(description="Hyperparameter Setting for DQN")
# parser.add_argument('-integers', metavar='N', type=int,
#                     help='an integer for the accumulator')
# args = parser.parse_args()
# class A:
#     def __init__(self, args):
#         self.x = args
#         self.x.y = 2
# a = A(args)
# print(a.x.integers)

# env = gym.make("CartPole-v1", render_mode="rgb_array")
# print(env._max_episode_steps)
# a = torch.nn.Parameter(torch.FloatTensor(10, 10))
a = deque([], maxlen=5)
a.append(3)
a.append(3)
print(len(a))