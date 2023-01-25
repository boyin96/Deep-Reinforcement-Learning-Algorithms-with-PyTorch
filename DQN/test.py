import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as n
import gym
from itertools import count
# from collections import deque, namedtuple
# Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "terminal"])
# import argparse
# #
# parser = argparse.ArgumentParser(description="Hyperparameter Setting for DQN")
# parser.add_argument('-integers', metavar='N', type=int,
#                     help='an integer for the accumulator')
# args = parser.parse_args()
# args.e = 1
# print(args.e)
# class A:
#     def __init__(self, args):
#         self.x = args
#         self.x.y = 2
# a = A(args)
# print(a.x.integers)
#
env = gym.make("LunarLander-v2", render_mode="rgb_array")
env.reset()
action = env.action_space.sample()  # agent policy that uses the observation and info
observation, reward, terminated, truncated, info = env.step(action)
print(action)
print(observation, reward, terminated, truncated, info)

# print(env._max_episode_steps)
# a = torch.nn.Parameter(torch.FloatTensor(10, 10))
# t = Transition((212,12),3,4,5,7)
# x = deque([], maxlen=4)
# x[0]=1
# x[1]=2
# x[2]=3
# x[3]=4
# print(x)
# x = np.array([1,2,3])
# print(2**2)
# i = 0
# for _ in count():
#     print(1)
#     i += 1
#     if i > 6:
#         break