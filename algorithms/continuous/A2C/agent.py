import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from torch.nn.utils import clip_grad_norm_  # gradient clipping
from typing import Tuple

from algorithms.base_agent import BaseAgent
from algorithms.continuous.action_normalizer import ActionNormalizer
from networks.continuous.A2C import *


class A2CAgent(BaseAgent):
    """
    A2C Agent interacting with environment.

    Attributes:
        env:
        actor (nn.Module): actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        transition (list): temporary storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
            self,
            env,
            obs_dim: int,
            action_dim: int,
            action_low: float,
            action_high: float,
            lr_actor: float,
            lr_critic: float,
            gamma: float,
            entropy_weight: float,
    ):
        """
        Initialization.
        :param env:
        :param obs_dim: observation dimension
        :param action_dim: action dimension
        :param action_low:
        :param action_high:
        :param lr_actor (float): learning rate for the actor
        :param lr_critic (float): learning rate for the critic
        :param gamma (float): discount factor
        :param entropy_weight (float): rate of weighting entropy into the
                                       loss function
        """
        super(A2CAgent, self).__init__()

        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_normalizer = ActionNormalizer(action_low, action_high)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.entropy_weight = entropy_weight

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Training device: {}".format(self.device))

        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer and loss
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=self.lr_critic)
        self.loss_criterion = nn.SmoothL1Loss()

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """ Select an action. """
        state = torch.from_numpy(state).float().to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            entropy_term = dist.entropy().mean()
            self.transition = [state, log_prob, entropy_term]

        return selected_action.clamp(-1.0, 1.0).detach().cpu().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """ Take an action and return the response of the env. """
        action = self.action_normalizer.reverse_action(action)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = truncated  # for the Pendulum

        if not self.is_test:
            self.transition.extend([next_state, reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[float, float]:
        """ Update the model by gradient descent. """
        state, log_prob, entropy_term, next_state, reward, done \
            = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        next_state = torch.from_numpy(next_state).float().to(self.device)
        pred_value = self.critic(state)
        targ_value = reward + self.gamma * self.critic(next_state) * mask
        value_loss = self.loss_criterion(pred_value, targ_value.detach())

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        # clip_grad_norm_(self.critic.parameters(), 10.0)  # gradient clipping
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        advantage = (targ_value - pred_value).detach()  # not back-propagated
        actor_loss = -advantage * log_prob
        actor_loss -= self.entropy_weight * entropy_term  # entropy maximization

        # update policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # clip_grad_norm_(self.actor.parameters(), 10.0)  # gradient clipping
        self.actor_optimizer.step()

        return actor_loss.item(), value_loss.item()

    def train(self, num_frames: int):
        """ Train the agent. """
        self.is_test = False

        actor_losses = []
        critic_losses = []
        scores = []  # episodic cumulated reward

        state = self.env.reset()
        score = 0
        while self.total_step <= num_frames + 1:
            self.total_step += 1

            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

            actor_loss, critic_loss = self.update_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            if self.total_step % 1000 == 0:
                # print("{}: {}".format(self.total_step,
                #                       sum(scores) / len(scores)))
                print("{}: {}".format(self.total_step,
                                      sum(scores[-100:]) / 100))

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
