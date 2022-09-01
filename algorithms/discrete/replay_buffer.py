import numpy as np
from typing import Dict


class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, batch_size: int):
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        ids = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[ids],
                    next_obs=self.next_obs_buf[ids],
                    acts=self.acts_buf[ids],
                    rews=self.rews_buf[ids],
                    done=self.done_buf[ids])

    def __len__(self) -> int:
        return self.size
