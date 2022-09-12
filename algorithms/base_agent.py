import torch

from abc import ABCMeta, abstractmethod


class BaseAgent(metaclass=ABCMeta):

    def __init__(self):
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # mode: train / test
        self.is_test = False

    @abstractmethod
    def select_action(self, state):
        """ Select an action. """
        pass

    @abstractmethod
    def step(self, action):
        """ Take an action and return the response of the env. """
        pass

    @abstractmethod
    def update_model(self):
        """ update the value/policy model """
        pass

    @abstractmethod
    def train(self):
        """ Train the agent. """
        pass

    @abstractmethod
    def test(self):
        """ Test the agent. """
        pass
