import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from torch.nn.utils import clip_grad_norm_  # gradient clipping
from typing import Tuple

from algorithms.base_agent import BaseAgent
from algorithms.continuous.replay_buffer import ReplayBuffer
from algorithms.continuous.DDPG.noise import OUNoise
from algorithms.continuous.action_normalizer import ActionNormalizer
from networks.continuous.DDPG import *


class DDPGAgent(BaseAgent):
    """
    DDPG (Deep Deterministic Policy Gradient) Agent interacting with environment.

    Attribute:
        env:
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        noise (OUNoise): noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
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
            memory_size: int,
            batch_size: int,
            ou_noise_theta: float,
            ou_noise_sigma: float,
            gamma: float = 0.99,
            tau: float = 5e-3,
            initial_random_steps: int = 1e4,
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
        :param memory_size (int): length of memory
        :param batch_size (int): batch size for sampling
        :param ou_noise_theta: theta for Ornstein-Uhlenbeck noise
        :param ou_noise_sigma: sigma for Ornstein-Uhlenbeck noise
        :param gamma: discount factor
        :param tau: parameter for soft target update
        :param initial_random_steps: initial random action steps
        """
        super(DDPGAgent, self).__init__()

        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_normalizer = ActionNormalizer(action_low, action_high)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Training device: {}".format(self.device))

        # noise
        self.noise = OUNoise(
            action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
        )

        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer and loss
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=self.lr_critic)
        self.loss_criterion = nn.MSELoss()

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """ Select an action. """
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.random_action()
        else:
            selected_action = self.actor(
                torch.from_numpy(state).float().to(self.device)
            ).detach().cpu().numpy()

        # add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        self.transition = [state, selected_action]

        return selected_action

    def random_action(self):
        return self.env.action_space.sample()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """ Take an action and return the response of the env. """
        action = self.action_normalizer.reverse_action(action)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = truncated  # for the Pendulum

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> Tuple[float, float]:
        """ Update the model by gradient descent. """
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        state = torch.from_numpy(samples["obs"]).float().to(device)
        next_state = torch.from_numpy(samples["next_obs"]).float().to(device)
        action = torch.from_numpy(
            samples["acts"].reshape(-1, 1)).float().to(device)
        reward = torch.from_numpy(
            samples["rews"].reshape(-1, 1)).float().to(device)
        done = torch.from_numpy(
            samples["done"].reshape(-1, 1)).float().to(device)

        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.gamma * next_value * masks
        curr_return = curr_return.detach()

        # train critic
        values = self.critic(state, action)
        critic_loss = self.loss_criterion(values, curr_return)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clip_grad_norm_(self.critic.parameters(), 10.0)  # gradient clipping
        self.critic_optimizer.step()

        # train actor
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # clip_grad_norm_(self.actor.parameters(), 10.0)  # gradient clipping
        self.actor_optimizer.step()

        # target update
        self._target_soft_update()

        return actor_loss.item(), critic_loss.item()

    def train(self, num_frames: int):
        """ Train the agent. """
        self.is_test = False

        actor_losses = []
        critic_losses = []
        scores = []  # episodic cumulated reward

        state = self.env.reset()
        score = 0
        for self.total_step in range(1, num_frames + 1):

            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

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

            # if training is ready
            if (
                    len(self.memory) >= self.batch_size
                    and self.total_step > self.initial_random_steps
            ):
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        self.env.close()

    def _target_soft_update(self):
        """ Apply soft update to the target model. """
        tau = self.tau

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

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
