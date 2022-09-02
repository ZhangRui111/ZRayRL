import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from torch.nn.utils import clip_grad_norm_  # gradient clipping
from typing import Tuple

from algorithms.continuous.replay_buffer import ReplayBuffer
from algorithms.continuous.TD3.noise import GaussianNoise
from algorithms.continuous.action_normalizer import ActionNormalizer
from networks.continuous.TD3 import *


class TD3Agent:
    """
    TD3 (Twin Delayed Deep Deterministic Policy Gradient) Agent interacting with environment.

    Attribute:
        env:
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic1 (nn.Module): critic model to predict state values
        critic2 (nn.Module): critic model to predict state values
        critic_target1 (nn.Module): target critic model to predict state values
        critic_target2 (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        exploration_noise (GaussianNoise): gaussian noise for policy
        target_policy_noise (GaussianNoise): gaussian noise for target policy
        target_policy_noise_clip (float): clip target gaussian noise
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        policy_update_freq (int): update actor every time critic updates this times
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
            gamma: float = 0.99,
            tau: float = 5e-3,
            exploration_noise: float = 0.1,
            target_policy_noise: float = 0.2,
            target_policy_noise_clip: float = 0.5,
            initial_random_steps: int = int(1e4),
            policy_update_freq: int = 2,
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
        :param gamma: discount factor
        :param tau: parameter for soft target update
        :param exploration_noise: gaussian noise for policy
        :param target_policy_noise: gaussian noise for target policy
        :param target_policy_noise_clip: clip target gaussian noise
        :param initial_random_steps: initial random action steps
        :param policy_update_freq: update actor every time critic updates this times
        """
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
        self.policy_update_freq = policy_update_freq

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # noise
        self.exploration_noise = GaussianNoise(
            action_dim, exploration_noise, exploration_noise
        )
        self.target_policy_noise = GaussianNoise(
            action_dim, target_policy_noise, target_policy_noise
        )
        self.target_policy_noise_clip = target_policy_noise_clip

        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target1 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target1.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target2 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # concat critic parameters to use one optim
        self.critic_parameters = list(self.critic1.parameters()) + list(
            self.critic2.parameters()
        )

        # optimizer and loss
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_parameters, lr=self.lr_critic)
        self.loss_criterion = nn.MSELoss()

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # update step for actor
        self.update_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """ Select an action. """
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.random_action()
        else:
            selected_action = (
                self.actor(torch.FloatTensor(state).to(self.device))[0]
                    .detach()
                    .cpu()
                    .numpy()
            )

        # add noise for exploration during training
        if not self.is_test:
            noise = self.exploration_noise.sample()
            selected_action = np.clip(
                selected_action + noise, -1.0, 1.0
            )

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

    def update_model(self) -> torch.Tensor:
        """ Update the model by gradient descent. """
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        states = torch.from_numpy(samples["obs"]).float().to(device)
        next_states = torch.from_numpy(samples["next_obs"]).float().to(device)
        actions = torch.from_numpy(samples["acts"].reshape(-1, 1)).float().to(device)
        rewards = torch.from_numpy(samples["rews"].reshape(-1, 1)).float().to(device)
        dones = torch.from_numpy(samples["done"].reshape(-1, 1)).float().to(device)
        masks = 1 - dones

        # get actions with noise
        noise = torch.from_numpy(self.target_policy_noise.sample()).float().to(device)
        clipped_noise = torch.clamp(
            noise, -self.target_policy_noise_clip, self.target_policy_noise_clip
        )
        next_actions = (self.actor_target(next_states) + clipped_noise).clamp(
            -1.0, 1.0
        )

        # min (Q_1', Q_2')
        next_values1 = self.critic_target1(next_states, next_actions)
        next_values2 = self.critic_target2(next_states, next_actions)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_returns = rewards + self.gamma * next_values * masks
        curr_returns = curr_returns.detach()

        # critic loss
        values1 = self.critic1(states, actions)
        values2 = self.critic2(states, actions)
        critic1_loss = self.loss_criterion(values1, curr_returns)
        critic2_loss = self.loss_criterion(values2, curr_returns)

        # train critic
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clip_grad_norm_(self.critic_parameters, 10.0)  # gradient clipping
        self.critic_optimizer.step()

        if self.total_step % self.policy_update_freq == 0:
            # train actor
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # clip_grad_norm_(self.actor.parameters(), 10.0)  # gradient clipping
            self.actor_optimizer.step()

            # target update
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

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
                # print("{}: {}".format(self.total_step, sum(scores) / len(scores)))
                print("{}: {}".format(self.total_step, sum(scores[-100:]) / 100))

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
                self.critic_target1.parameters(), self.critic1.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target2.parameters(), self.critic2.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def test(self) -> None:
        """ Test the agent. """
        self.is_test = True

        done = False
        state = self.env.reset()
        score = 0
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

        print("score: {}".format(score))
        self.env.close()
