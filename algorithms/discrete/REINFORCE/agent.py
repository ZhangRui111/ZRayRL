import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from torch.nn.utils import clip_grad_norm_  # gradient clipping
from typing import Tuple

from networks.discrete.REINFORCE import *


class REINFORCEAgent:
    """
    REINFORCE Agent interacting with environment.

    Attributes:
        env:
        policy (nn.Module): actor model to select actions
        optimizer (Optimizer): optimizer for training policy
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
            self,
            env,
            obs_dim: int,
            action_dim: int,
            lr: float,
            gamma: float,
            entropy_weight: float,
    ):
        """
        Initialization.
        :param env:
        :param obs_dim: observation dimension
        :param action_dim: action dimension
        :param lr (float): learning rate
        :param gamma (float): discount factor
        :param entropy_weight (float): rate of weighting entropy into the loss function
        """
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.entropy_weight = entropy_weight

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks
        self.policy = Policy(obs_dim, action_dim).to(self.device)

        # optimizer and loss
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # store necessary info
        self.saved_log_probs = []
        self.rewards = []
        self.eps = np.finfo(np.float32).eps.item()

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """ Select an action. """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action, dist = self.policy(state)
        selected_action = torch.argmax(dist.probs).unsqueeze(0) if self.is_test else action

        if not self.is_test:
            self.saved_log_probs.append(dist.log_prob(selected_action))

        return selected_action.detach().cpu().numpy()[0]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """ Take an action and return the response of the env. """
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated  # for the CartPole

        if not self.is_test:
            self.rewards.append(reward)

        return next_state, reward, done

    def update_model(self) -> float:
        """ Update the model by gradient descent. """
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for log_prob, R in zip(self.saved_log_probs, returns):
            loss = -R * log_prob
            loss += self.entropy_weight * -log_prob
            policy_loss.append(loss)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        # clip_grad_norm_(self.policy.parameters(), 10.0)  # gradient clipping
        self.optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]

        return policy_loss.item()

    def train(self, num_frames: int):
        """ Train the agent. """
        self.is_test = False

        policy_losses = []
        scores = []  # episodic cumulated reward

        state = self.env.reset()
        score = 0
        while self.total_step <= num_frames + 1:
            self.total_step += 1

            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

            # if episode ends
            if done:
                policy_loss = self.update_model()
                policy_losses.append(policy_loss)

                state = self.env.reset()
                scores.append(score)
                score = 0

            if self.total_step % 1000 == 0:
                # print("{}: {}".format(self.total_step, sum(scores) / len(scores)))
                print("{}: {}".format(self.total_step, sum(scores[-100:]) / 100))

        # termination
        self.env.close()

    def test(self) -> None:
        """ Test the agent. """
        self.is_test = True

        avg_score = []
        for _ in range(10):
            done = False
            state = self.env.reset()
            score = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward
            avg_score.append(score)

        print("Average score: {}".format(sum(avg_score) / len(avg_score)))
        self.env.close()
