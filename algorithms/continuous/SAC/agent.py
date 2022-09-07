import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from torch.nn.utils import clip_grad_norm_  # gradient clipping
from typing import Tuple

from algorithms.continuous.replay_buffer import ReplayBuffer
from algorithms.continuous.action_normalizer import ActionNormalizer
from networks.continuous.SAC import *


class SACAgent:
    """
    SAC (Soft Actor-Critic) agent interacting with environment.

    Attributes:
        env:
        actor (nn.Module): actor model to select actions
        actor_optimizer (Optimizer): optimizer for training actor
        vf (nn.Module): critic model to predict state values
        vf_target (nn.Module): target critic model to predict state values
        vf_optimizer (Optimizer): optimizer for training vf
        qf_1 (nn.Module): critic model to predict state-action values
        qf_2 (nn.Module): critic model to predict state-action values
        qf_1_optimizer (Optimizer): optimizer for training qf_1
        qf_2_optimizer (Optimizer): optimizer for training qf_2
        memory (ReplayBuffer): replay memory
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        policy_update_freq (int): policy update frequency
        device (torch.device): cpu / gpu
        target_entropy (int): desired entropy used for the inequality constraint
        log_alpha (torch.Tensor): weight for entropy
        alpha_optimizer (Optimizer): optimizer for alpha
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
            lr_critic_q: float,
            lr_critic_v: float,
            memory_size: int,
            batch_size: int,
            gamma: float = 0.99,
            tau: float = 5e-3,
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
        :param lr_critic_q (float): learning rate for the critic Q
        :param lr_critic_v (float): learning rate for the critic V
        :param memory_size (int): length of memory
        :param batch_size (int): batch size for sampling
        :param gamma (float): discount factor
        :param tau (float): parameter for soft target update
        :param initial_random_steps (int): initial random action steps
        :param policy_update_freq (int): policy update frequency
        """
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_normalizer = ActionNormalizer(action_low, action_high)
        self.lr_actor = lr_actor
        self.lr_critic_q = lr_critic_q
        self.lr_critic_v = lr_critic_v
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # automatic entropy tuning
        self.target_entropy = -np.prod((action_dim,)).item()  # heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # actor
        self.actor = Actor(obs_dim, action_dim).to(self.device)

        # v function
        self.vf = CriticV(obs_dim).to(self.device)
        self.vf_target = CriticV(obs_dim).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        # q function
        self.qf_1 = CriticQ(obs_dim + action_dim).to(self.device)
        self.qf_2 = CriticQ(obs_dim + action_dim).to(self.device)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.lr_critic_v)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=self.lr_critic_q)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=self.lr_critic_q)
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
            )[0].detach().cpu().numpy()

        self.transition = [state, selected_action]

        return selected_action

    def random_action(self):
        return self.env.action_space.sample()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """ Take an action and return the response of the env. """
        action = self.action_normalizer.reverse_action(action)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = truncated  # for the Pendulum

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """ Update the model by gradient descent. """
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        state = torch.from_numpy(samples["obs"]).float().to(device)
        next_state = torch.from_numpy(samples["next_obs"]).float().to(device)
        action = torch.from_numpy(samples["acts"].reshape(-1, 1)).float().to(device)
        reward = torch.from_numpy(samples["rews"].reshape(-1, 1)).float().to(device)
        done = torch.from_numpy(samples["done"].reshape(-1, 1)).float().to(device)
        new_action, log_prob = self.actor(state)

        # train alpha (dual problem)
        alpha_loss = (
                -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()  # used for the actor loss calculation

        # q function loss
        mask = 1 - done
        q_1_pred = self.qf_1(state, action)
        q_2_pred = self.qf_2(state, action)
        v_target = self.vf_target(next_state)
        q_target = reward + self.gamma * v_target * mask
        qf_1_loss = self.loss_criterion(q_1_pred, q_target.detach())
        qf_2_loss = self.loss_criterion(q_2_pred, q_target.detach())

        # v function loss
        v_pred = self.vf(state)
        q_pred = torch.min(
            self.qf_1(state, new_action), self.qf_2(state, new_action)
        )
        v_target = q_pred - alpha * log_prob
        vf_loss = self.loss_criterion(v_pred, v_target.detach())

        if self.total_step % self.policy_update_freq == 0:
            # actor loss
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target update (vf)
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

        # train Q functions
        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()

        qf_loss = qf_1_loss + qf_2_loss

        # train V function
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        return actor_loss.item(), qf_loss.item(), vf_loss.item(), alpha_loss.item()

    def train(self, num_frames: int):
        """ Train the agent. """
        self.is_test = False

        actor_losses, qf_losses, vf_losses, alpha_losses = [], [], [], []
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
                losses = self.update_model()
                actor_losses.append(losses[0])
                qf_losses.append(losses[1])
                vf_losses.append(losses[2])
                alpha_losses.append(losses[3])

        self.env.close()

    def _target_soft_update(self):
        """ Apply soft update to the target model. """
        tau = self.tau

        for t_param, l_param in zip(
                self.vf_target.parameters(), self.vf.parameters()
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
