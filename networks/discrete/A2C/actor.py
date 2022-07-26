import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Actor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(
            self, state: torch.Tensor
    ) -> Tuple[torch.Tensor,
               torch.distributions.distribution.Distribution]:
        logits = self.layers(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist


# class Actor(nn.Module):
#     """ The network outputs probs for the Categorical, which is OK. """
#     def __init__(self, in_dim: int, out_dim: int):
#         super(Actor, self).__init__()
#
#         self.layers = nn.Sequential(
#             nn.Linear(in_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, out_dim),
#             nn.Softmax(dim=-1),
#         )
#
#     def forward(
#             self, state: torch.Tensor
#     ) -> Tuple[torch.Tensor,
#                torch.distributions.distribution.Distribution]:
#         prob = self.layers(state)
#         dist = Categorical(prob)
#         action = dist.sample()
#         return action, dist
