import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple


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

    def forward(
            self, state: torch.Tensor
    ) -> Tuple[torch.Tensor,
               torch.distributions.distribution.Distribution]:
        x = self.hidden(state)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)

        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        # rsample: sampling using re-parameterization trick.
        # Thus, we can do back-propagation.
        z = dist.rsample()

        # normalize action and log_prob
        # see appendix C of [2]
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob
