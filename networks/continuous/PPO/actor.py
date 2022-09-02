import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            log_std_min: int = -20,
            log_std_max: int = 0,
    ):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hidden = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
        )
        self.mu_layer = nn.Sequential(
            nn.Linear(32, out_dim),
            nn.Tanh(),
        )
        self.log_std_layer = nn.Sequential(
            nn.Linear(32, out_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.hidden(state)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)

        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist
