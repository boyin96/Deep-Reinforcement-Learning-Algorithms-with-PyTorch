import random
import gym
import torch
import numpy as np
import argparse

from agent import DQNAgent

# environment
env_id = "CartPole-v1"
env = gym.make(env_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(tr_seed):
    torch.manual_seed(tr_seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser("Hyperparameters setting for Rainbow DQN")
# About system setting
parser.add_argument("--num_frames", type=int, default=int(2e4), help="Maximum number of training steps")
parser.add_argument("--seed", type=int, default=42, help="The seed of entire system")
parser.add_argument("--env", type=gym.Env, default=env, help="The environment in system")
parser.add_argument("--device", type=torch.device, default=device, help="The cpu or gpu being used")

# About sample size
parser.add_argument("--memory_size", type=int, default=int(1e3), help="The maximum replay buffer capacity")
parser.add_argument("--batch_size", type=int, default=256, help="The number of batch size")
parser.add_argument("--hidden_dim", type=int, default=128, help="The number of neurons in hidden layers")

# About agent learning
parser.add_argument("--lr", type=float, default=float(1e-4), help="Learning rate of actor")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

# About updating target network
parser.add_argument("--tau", type=float, default=0.005, help="Soft update the target network")
parser.add_argument("--use_soft_update", type=bool, default=False, help="Whether to use soft update")
parser.add_argument("--target_update", type=int, default=100, help="Period for target model's hard update")

# About distributed value DQN
parser.add_argument("--v_min", type=float, default=0.0, help="value of support")
parser.add_argument("--v_max", type=float, default=200.0, help="max value of support")
parser.add_argument("--atom_size", type=int, default=51, help="the unit number of support")

# About multi-steps TD algorithm
parser.add_argument("--ues_n_step", type=bool, default=True, help="Whether to use n steps")
parser.add_argument("--n_step", type=int, default=5, help="Using n_steps TD algorithm")
parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
parser.add_argument("--beta", type=float, default=0.4, help="Important sampling parameter in PER")
parser.add_argument("--prior_eps", type=float, default=float(1e-6), help="Guarantees every transition can be sampled")

# About learning
parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

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
    agent = DQNAgent(args)
    agent.train()
