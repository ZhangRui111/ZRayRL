import torch
import torch.nn as nn


class CriticQ(nn.Module):
    def __init__(self, in_dim: int):
        super(CriticQ, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat((state, action), dim=-1)
        q = self.layers(x)
        return q


class CriticV(nn.Module):
    def __init__(self, in_dim: int):
        super(CriticV, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        v = self.layers(state)
        return v
