import random
import gym
import torch
import numpy as np

from agent import A2CAgent

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
env_id = "Pendulum-v1"
env = gym.make(env_id)
# env.seed(seed): this is no longer supported by gym.

num_frames = 100000
gamma = 0.9
entropy_weight = 1e-2

agent = A2CAgent(env, gamma, entropy_weight)
agent.train(num_frames)
