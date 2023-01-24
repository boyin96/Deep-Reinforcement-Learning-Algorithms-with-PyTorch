import sys
import argparse

from os.path import dirname, abspath
from utilities.trainer import Runner

sys.path.append(dirname(abspath(__file__)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters for DQN")

    parser.add_argument("--max_train_steps", type=int, default=int(500), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--buffer_capacity", type=int, default=int(1e4), help="The maximum replay-buffer capacity ")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5),
                        help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200,
                        help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="Using n_steps TD algorithm")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    # Choose DQN training strategies
    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=True, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=bool, default=True, help="Whether to use prioritized experience replay (PER)")
    parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")

    args = parser.parse_args()

    env_names = ["CartPole-v1", "LunarLander-v2"]
    model = ["human", "rgb_array"]
    env_index = 1
    seed = 42
    runner = Runner(args=args, env_name=env_names[env_index], number=1, model=model[0], seed=seed)
    runner.run()
