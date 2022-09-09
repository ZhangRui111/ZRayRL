import numpy as np


class Env:

    def __init__(self):
        # MDP: S, A, P, R
        self.n_states = 4  # S = {s0, s1, s2, s3}
        self.n_actions = 4  # A = {a0, a1, a2, a3}
        self.R = np.asarray(  # R
            [[0, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0]],
            dtype=np.float,
        )
        self.P = np.asarray(  # P
            [[0, 2, 0, 1],
             [1, 3, 0, 1],
             [0, 2, 2, 3],
             [3, 3, 3, 3]],
            dtype=np.int,
        )

        self.terminal_s = 3  # terminal state s3

        self.s = 0  # initial state
        self.reset()

    def reset(self):
        # self.s = 0  # fixed initial state
        self.s = np.random.randint(0, self.n_states)  # random initial state
        return self.s

    def step(self, a):
        next_s = self.P[self.s, a]
        r = self.R[self.s, a]
        done = True if next_s == self.terminal_s else False
        self.s = next_s
        return next_s, r, done, {}
