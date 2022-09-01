import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Dict, Tuple

from algorithms.discrete.replay_buffer import ReplayBuffer
from networks.discrete.DQN.network import Network


class DQNAgent:
    """
    DQN Agent interacting with environment.

    Attribute:
        env:
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer: optimizer for training dqn
        loss_criterion: loss function for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
            self,
            env,
            obs_dim: int,
            action_dim: int,
            lr: float,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
    ):
        """
        Initialization.
        :param env:
        :param obs_dim: observation dimension
        :param action_dim: action dimension
        :param lr (float): learning rate
        :param memory_size (int): length of memory
        :param batch_size (int): batch size for sampling
        :param target_update (int): period for target model's hard update
        :param epsilon_decay (float): step size to decrease epsilon
        :param max_epsilon (float): max value of epsilon
        :param min_epsilon (float): min value of epsilon
        :param gamma (float): discount factor
        """
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.obs_dim, memory_size, batch_size)
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        # device: cpu / gpu, gpu is preferred
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(self.obs_dim, self.action_dim).to(self.device)
        self.dqn_target = Network(self.obs_dim, self.action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)
        self.loss_criterion = nn.SmoothL1Loss()

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """ Select an action. """
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.from_numpy(state).float().to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """ Take an action and return the response of the env. """
        next_state, reward, done, truncated, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """ Update the model by gradient descent. """
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_frames: int):
        """ Train the agent. """
        self.is_test = False

        losses = []
        scores = []  # episodic cumulated reward
        update_counter = 0

        state = self.env.reset()
        score = 0
        for frame_idx in range(1, num_frames + 1):

            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            if frame_idx % 1000 == 0:
                print("{}: {}".format(frame_idx, sum(scores)/len(scores)))

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_counter += 1

                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )

                # if hard update is needed
                if update_counter % self.target_update == 0:
                    self._target_hard_update()

        self.env.close()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """ Return dqn loss. """
        device = self.device  # for shortening the following lines
        state = torch.from_numpy(samples["obs"]).float().to(device)
        next_state = torch.from_numpy(samples["next_obs"]).float().to(device)
        action = torch.from_numpy(samples["acts"].reshape(-1, 1)).long().to(device)
        reward = torch.from_numpy(samples["rews"].reshape(-1, 1)).float().to(device)
        done = torch.from_numpy(samples["done"].reshape(-1, 1)).float().to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = self.loss_criterion(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """ Apply hard update to the target model."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

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
