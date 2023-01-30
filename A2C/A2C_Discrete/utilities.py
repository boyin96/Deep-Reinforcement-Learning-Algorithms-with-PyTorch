import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Any


class Actor(nn.Module):
    def __init__(self, args: argparse.Namespace):
        """Initialize."""
        super(Actor, self).__init__()
        self.in_dim = args.obs_dim
        self.out_dim = args.action_dim

        self.hidden1 = nn.Linear(self.in_dim, args.hidden_dim)
        self.mu_layer = nn.Linear(args.hidden_dim, self.out_dim)

    def forward(self, state: torch.Tensor) -> Union[torch.Tensor, Any]:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        action_prob = F.softmax(self.mu_layer(x), dim=0)

        return action_prob


class Critic(nn.Module):
    def __init__(self, args: argparse.Namespace):
        """Initialize."""
        super(Critic, self).__init__()

        self.in_dim = args.obs_dim

        self.hidden1 = nn.Linear(self.in_dim, args.hidden_dim)
        self.out = nn.Linear(args.hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        value = self.out(x)

        return value
