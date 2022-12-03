import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value = self.layers(state)
        return value
