import torch
import gym
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dqn_agent import DQN
from replay_buffer import ReplayBuffer, Prioritized_ReplayBuffer, N_Steps_ReplayBuffer, \
    N_Steps_Prioritized_ReplayBuffer


class Runner:
    def __init__(self, args, env_name, number, model, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        self.env = gym.make(env_name, render_model=model)
        self.env.action_space.seed(seed)

        self.env_evaluate = gym.make(env_name, render_model=model)  # Rebuild an environment to evaluate model
        self.env_evaluate.action_space.seed(seed)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.n
        self.args.episode_limit = self.env._max_episode_steps  # Maximum steps in each episode

        print("env={}".format(self.env_name))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        if args.use_per and args.use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)

        self.agent = DQN(args)

        self.algorithm = "DQN"
        if args.use_double and args.use_dueling and args.use_noisy and args.use_per and args.use_n_steps:
            self.algorithm = 'Rainbow_' + self.algorithm
        else:
            if args.use_double:
                self.algorithm += "_Double"
            if args.use_dueling:
                self.algorithm += "_Dueling"
            if args.use_noisy:
                self.algorithm += "_Noisy"
            if args.use_per:
                self.algorithm += "_PER"
            if args.use_n_steps:
                self.algorithm += "_N_Steps"

        self.writer = SummaryWriter(
            log_dir="runs/DQN/{}_env_{}_number_{}_seed_{}".format(self.algorithm, env_name, number, seed))

        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training

        if args.use_noisy:
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def run(self):
        self.evaluate_policy()
        while self.total_steps < self.args.max_train_steps:
            state = self.env.reset()
            done = False
            episode_steps = 0

            while not done:
                action = self.agent.choose_action(state, epsilon=self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                episode_steps += 1
                self.total_steps += 1

                if not self.args.use_noisy:  # Decay epsilon
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # terminal means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != self.args.episode_limit:
                    if self.env_name == 'LunarLander-v2':
                        if reward <= -100: reward = -1  # good for LunarLander
                    terminal = True
                else:
                    terminal = False

                self.replay_buffer.store_transition(state, action, reward, next_state, terminal,
                                                    done)  # Store the transition
                state = next_state

                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)

                if self.total_steps % self.args.evaluate_freq == 0:
                    self.evaluate_policy()

        # Save reward
        np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.algorithm, self.env_name, self.number,
                                                                      self.seed), np.array(self.evaluate_rewards))

    def evaluate_policy(self):
        evaluate_reward = 0
        self.agent.predict_net.eval()
        for _ in range(self.args.evaluate_times):
            state = self.env_evaluate.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=0)
                next_state, reward, done, _ = self.env_evaluate.step(action)
                episode_reward += reward
                state = next_state
            evaluate_reward += episode_reward
        self.agent.predict_net.train()
        evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps, evaluate_reward,
                                                                          self.epsilon))
        self.writer.add_scalar('step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
