import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dueling_Net(nn.Module):
    def __init__(self, args):
        super(Dueling_Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.use_noisy:
            self.V = NoisyLinear(args.hidden_dim, 1)
            self.D = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.V = nn.Linear(args.hidden_dim, 1)
            self.D = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        V = self.V(x)  # batch_size X 1
        D = self.D(x)  # batch_size X action_dim
        Q = V + (D - torch.mean(D, dim=-1, keepdim=True))  # Q(s,a)=V(s)+D(s,a)-mean(D(s,a))
        return Q


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.use_noisy:
            self.fc3 = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.fc3 = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
