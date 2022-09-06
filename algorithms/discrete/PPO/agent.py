import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from torch.nn.utils import clip_grad_norm_  # gradient clipping
from typing import Tuple

from networks.discrete.PPO import *
from algorithms.discrete.PPO.gae import compute_gae
from algorithms.discrete.PPO.memory import Memory


class PPOAgent:
    """
    PPO Agent interacting with environment.

    Attributes:
        env:
        actor (nn.Module): actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        epsilon (float): amount of clipping surrogate objective
        epoch (int): the number of update
        rollout_len (int): the number of rollout
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
            lr_actor: float,
            lr_critic: float,
            batch_size: int,
            gamma: float,
            tau: float,
            epsilon: float,
            epoch: int,
            rollout_len: int,
            entropy_weight: float,
    ):
        """
        Initialization.
        :param env:
        :param obs_dim: observation dimension
        :param action_dim: action dimension
        :param lr_actor (float): learning rate for the actor
        :param lr_critic (float): learning rate for the critic
        :param batch_size (int): batch size for sampling
        :param gamma (float): discount factor
        :param tau (float): lambda of generalized advantage estimation (GAE)
        :param epsilon (float): amount of clipping surrogate objective
        :param epoch (int): the number of update
        :param rollout_len (int): the number of rollout
        :param entropy_weight (float): rate of weighting entropy into the loss function
        """
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.rollout_len = rollout_len
        self.entropy_weight = entropy_weight

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer and loss
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # memory for training
        self.memory = Memory()

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """ Select an action. """
        state = torch.from_numpy(state).float().to(self.device)
        action, dist = self.actor(state)
        selected_action = torch.argmax(dist.probs).unsqueeze(0) if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.memory.states.append(state)
            self.memory.actions.append(selected_action)
            self.memory.values.append(value)
            self.memory.log_probs.append(dist.log_prob(selected_action))

        return selected_action.detach().cpu().numpy()[0]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Take an action and return the response of the env. """
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated  # for the CartPole

        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.memory.rewards.append(torch.from_numpy(reward).float().to(self.device))
            self.memory.masks.append(torch.from_numpy(1 - done).float().to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """ Update the model by gradient descent. """
        device = self.device  # for shortening the following lines

        next_state = torch.from_numpy(next_state).float().to(device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.memory.rewards,
            self.memory.masks,
            self.memory.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.memory.states).view(-1, self.obs_dim)
        actions = torch.cat(self.memory.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.memory.values).detach()
        log_probs = torch.cat(self.memory.log_probs).detach()
        advantages = returns - values

        full_size = states.size(0)
        n_update = full_size // self.batch_size
        for _ in range(self.epoch):
            for _ in range(n_update):
                rand_ids = np.random.choice(full_size, self.batch_size)
                state, action, old_value = states[rand_ids, :], actions[rand_ids], values[rand_ids]
                old_log_prob, return_, adv = log_probs[rand_ids], returns[rand_ids], advantages[rand_ids]

                # calculate ratios
                _, dist = self.actor(state)
                log_prob = dist.log_prob(action)
                ratio = (log_prob - old_log_prob).exp()

                # actor_loss
                surr_loss = ratio * adv
                clipped_surr_loss = (
                        torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
                )

                # entropy
                entropy = dist.entropy().mean()

                actor_loss = (
                        - torch.min(surr_loss, clipped_surr_loss).mean()
                        - entropy * self.entropy_weight
                )

                # critic_loss
                value = self.critic(state)
                critic_loss = (return_ - value).pow(2).mean()

                # train critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                # clip_grad_norm_(self.critic.parameters(), 10.0)  # gradient clipping
                self.critic_optimizer.step()

                # train actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # clip_grad_norm_(self.actor.parameters(), 10.0)  # gradient clipping
                self.actor_optimizer.step()

        self.memory.clear()

        return actor_loss.item(), critic_loss.item()

    def train(self, num_frames: int):
        """ Train the agent. """
        self.is_test = False

        actor_losses = []
        critic_losses = []
        scores = []  # episodic cumulated reward

        state = self.env.reset()
        # Expanding the shape is necessary for tensor.cat() in the update_model()
        state = np.expand_dims(state, axis=0)
        score = 0
        while self.total_step <= num_frames + 1:
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    state = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    score = 0

                if self.total_step % 1000 == 0:
                    # print("{}: {}".format(self.total_step, sum(scores) / len(scores)))
                    print("{}: {}".format(self.total_step, sum(scores[-100:]) / 100))

            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        # termination
        self.env.close()

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
