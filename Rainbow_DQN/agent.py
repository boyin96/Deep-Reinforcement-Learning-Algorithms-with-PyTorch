import gym
import torch
import argparse
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.utils import clip_grad_norm_
from typing import Dict, List, Tuple, Any

from utilities import ReplayBuffer, PrioritizedReplayBuffer, Network


class DQNAgent:
    """DQN Agent interacting with environment."""

    def __init__(self, args: argparse.Namespace):

        self.args = args
        self.use_soft_update = args.use_soft_update

        self.num_frames = args.num_frames
        self.memory_size = args.memory_size
        obs_dim = args.env.observation_space.shape[0]
        action_dim = args.env.action_space.n
        self.args.obs_dim = obs_dim
        self.args.action_dim = action_dim

        self.env = args.env
        self.batch_size = args.batch_size
        self.target_update = args.target_update
        self.gamma = args.gamma

        # Device: cpu / gpu
        self.device = args.device

        # PER
        # Memory for 1-step Learning
        self.beta = args.beta
        self.prior_eps = args.prior_eps
        self.memory = PrioritizedReplayBuffer(obs_dim, self.memory_size, self.batch_size, alpha=args.alpha)

        # Memory for N-step Learning
        self.use_n_step = args.ues_n_step
        if self.use_n_step:
            self.n_step = args.n_step
            self.memory_n = ReplayBuffer(obs_dim, self.memory_size, self.batch_size, n_step=args.n_step,
                                         gamma=self.gamma)

        # Categorical DQN parameters
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.atom_size = args.atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)
        self.args.support = self.support

        # Networks: dqn and dqn_target
        self.dqn = Network(args).to(self.device)
        self.dqn_target = Network(args).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=args.lr)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = args.is_test

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
        selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are going combine 1-step loss and n-step loss to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), self.args.grad_clip)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def train(self, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        state, _ = self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        score = 0
        plt.ion()

        for frame_idx in range(1, self.num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # PER: increase beta
            fraction = min(frame_idx / self.num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if self.use_soft_update:
                    self._target_soft_update()
                else:
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot_dynamic(frame_idx, scores, losses)

        plt.ioff()
        plt.show()
        self.env.close()

    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state = self.env.reset()
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        # reset
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _target_soft_update(self):
        """Soft update: target <- local."""
        for param, target_param in zip(self.dqn.parameters(), self.dqn_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    @staticmethod
    def _plot_dynamic(
            frame_idx: int,
            scores: List[float],
            losses: List[torch.Tensor],
    ):
        """Plot the training progresses."""
        # clear_output(True)
        plt.clf()
        plt.figure(1)
        plt.subplot(121)
        plt.title("frame {}. score: {}".format(frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(122)
        plt.title('loss')
        plt.plot(losses)
        plt.pause(0.001)
