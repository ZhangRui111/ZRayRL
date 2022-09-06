from collections import deque
from typing import List, Deque


def compute_gae(
    next_value: list,
    rewards: list,
    masks: list,
    values: list,
    gamma: float,
    tau: float
) -> List:
    """
    Compute the GAE (Generalized Advantage Estimation).
    GAE help to reduce variance while maintaining a proper level of bias.
    https://arxiv.org/pdf/1506.02438.pdf
    """
    gae = 0
    values = values + [next_value]
    returns: Deque[float] = deque()

    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step]
            + gamma * values[step + 1] * masks[step]
            - values[step]
        )
        gae = delta + gamma * tau * masks[step] * gae
        returns.appendleft(gae + values[step])

    return list(returns)
