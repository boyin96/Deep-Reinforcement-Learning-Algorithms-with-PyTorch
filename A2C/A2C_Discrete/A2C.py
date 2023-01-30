import random
import gym
import torch
import numpy as np
import argparse

from agent import A2CAgent

# environment
# continuous action
# env_id = "Pendulum-v1"
# discrete action
env_id = "CartPole-v1"
env = gym.make(env_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(tr_seed):
    torch.manual_seed(tr_seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser("Hyperparameters setting for Continuous A2C")
# About system setting
parser.add_argument("--num_frames", type=int, default=int(1e5), help="Maximum number of training steps")
parser.add_argument("--seed", type=int, default=42, help="The seed of entire system")
parser.add_argument("--env", type=gym.Env, default=env, help="The environment in system")
parser.add_argument("--device", type=torch.device, default=device, help="The cpu or gpu being used")
parser.add_argument("--plotting_interval", type=int, default=1000, help="How many steps to plot results")

# About sample size
parser.add_argument("--hidden_dim", type=int, default=128, help="The number of neurons in hidden layers")

# About agent learning
parser.add_argument("--lr_actor", type=float, default=float(1e-4), help="Learning rate of actor")
parser.add_argument("--lr_critic", type=float, default=float(1e-3), help="Learning rate of critic")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--entropy_weight", type=float, default=float(1e-2), help="Entropy Weight")

# About updating target network
parser.add_argument("--use_target_net", type=bool, default=False, help="Whether to use target network")
parser.add_argument("--tau", type=float, default=0.005, help="Soft update the target network")
parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
parser.add_argument("--target_update", type=int, default=300, help="Period for target model's hard update")

# About processing results
parser.add_argument("--is_test", type=bool, default=False, help="Test or Train")
parser.add_argument("--save_results", type=bool, default=True, help="Whether to save results")

args = parser.parse_args()

if __name__ == '__main__':
    np.random.seed(args.seed)
    random.seed(args.seed)
    seed_torch(args.seed)
    # env.seed(seed): this is no longer supported by gym.

    # train
    agent = A2CAgent(args)
    agent.train()
