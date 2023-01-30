import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from typing import List, Tuple, Union

from utilities import Actor, Critic


class A2CAgent:
    """A2CAgent interacting with environment."""

    def __init__(self, args: argparse.Namespace):
        """Initialize."""
        self.use_soft_update = args.use_soft_update
        self.args = args
        self.env = args.env
        self.gamma = args.gamma
        self.entropy_weight = args.entropy_weight
        self.num_frames = args.num_frames
        self.plotting_interval = args.plotting_interval
        self.use_target_net = args.use_target_net

        # device: cpu / gpu
        self.device = args.device

        self.target_update = args.target_update

        # networks
        obs_dim = args.env.observation_space.shape[0]
        action_dim = args.env.action_space.n
        self.args.obs_dim = obs_dim
        self.args.action_dim = action_dim
        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)

        self.critic_target = Critic(args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), args.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), args.lr_critic)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = args.is_test

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        prob_weights = self.actor(state)
        action = np.random.choice(range(self.args.action_dim), p=prob_weights.detach().numpy())

        if not self.is_test:
            log_prob_pi = torch.log(prob_weights[action])
            log_prob_sum = prob_weights * torch.log(prob_weights)
            self.transition = [state, [log_prob_pi, log_prob_sum.sum()]]

        return action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        state, log_prob, next_state, reward, done = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        next_state = torch.FloatTensor(next_state).to(self.device)
        pred_value = self.critic(state)
        if self.use_target_net:
            targ_value = reward + self.gamma * self.critic_target(next_state) * mask
        else:
            targ_value = reward + self.gamma * self.critic(next_state) * mask
        value_loss = F.smooth_l1_loss(pred_value, targ_value.detach())

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        advantage = (targ_value - pred_value).detach()  # not backpropagated
        policy_loss = -advantage * log_prob[0]
        policy_loss += self.entropy_weight * -log_prob[1]  # entropy maximization

        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False

        update_cnt = 0
        actor_losses, critic_losses, scores = [], [], []
        state, _ = self.env.reset()
        score = 0
        plt.ion()

        for self.total_step in range(1, self.num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            actor_loss, critic_loss = self.update_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0

            # if using target network
            if self.use_target_net:
                update_cnt += 1
                # if hard update is needed
                if self.use_soft_update:
                    self._target_soft_update()
                else:
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                # plot
            if self.total_step % self.plotting_interval == 0:
                self._plot_dynamic(self.total_step, scores, actor_losses, critic_losses)

        plt.ioff()
        plt.show()
        self.env.close()

    def test(self):
        """Test the agent."""
        self.is_test = True

        state = self.env.reset()
        done = False
        score = 0

        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        return frames

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.critic_target.load_state_dict(self.critic.state_dict())

    def _target_soft_update(self):
        """Soft update: target <- local."""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    @staticmethod
    def _plot_dynamic(
            frame_idx: int,
            scores: List[float],
            actor_losses: Union[List[float], List[torch.Tensor]],
            critic_losses: Union[List[float], List[torch.Tensor]],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        plt.clf()
        plt.figure(1)
        for loc_i, title_i, values_i in subplot_params:
            subplot(loc_i, title_i, values_i)
        plt.pause(0.001)
