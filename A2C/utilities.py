import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Union, Any


def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, args: argparse.Namespace):
        """Initialize."""
        super(Actor, self).__init__()
        self.in_dim = args.obs_dim
        self.out_dim = args.action_dim

        self.hidden1 = nn.Linear(self.in_dim, args.hidden_dim)
        self.mu_layer = nn.Linear(args.hidden_dim, self.out_dim)
        self.log_std_layer = nn.Linear(args.hidden_dim, self.out_dim)

        initialize_uniformly(self.mu_layer)
        initialize_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor) -> Union[torch.Tensor, Any]:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))

        mu = torch.tanh(self.mu_layer(x)) * 2
        log_std = F.softplus(self.log_std_layer(x))
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class Critic(nn.Module):
    def __init__(self, args: argparse.Namespace):
        """Initialize."""
        super(Critic, self).__init__()

        self.in_dim = args.obs_dim

        self.hidden1 = nn.Linear(self.in_dim, args.hidden_dim)
        self.out = nn.Linear(args.hidden_dim, 1)

        initialize_uniformly(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        value = self.out(x)

        return value
