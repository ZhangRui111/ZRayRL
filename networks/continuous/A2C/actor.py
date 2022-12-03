import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Actor, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )
        self.mu_layer = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.Tanh(),
        )
        self.log_std_layer = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.Softplus(),
        )

    def forward(
            self, state: torch.Tensor
    ) -> Tuple[torch.Tensor,
               torch.distributions.distribution.Distribution]:
        x = self.hidden(state)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)

        std = torch.exp(log_std)
        dist = Normal(mu, std)

        action = dist.sample()

        return action, dist
