import numpy as np


class ActionNormalizer:
    """ Rescale and relocate a single action set. """

    def __init__(self, low, high):
        self.low, self.high = low, high

        self.scale_factor = (self.high - self.low) / 2
        self.reloc_factor = self.high - self.scale_factor

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """
        Change the range (-1, 1) to (low, high).
        """
        reversed_action = np.copy(action)

        reversed_action = reversed_action * self.scale_factor + self.reloc_factor
        reversed_action = np.clip(reversed_action, self.low, self.high)

        return reversed_action

    def norm_action(self, action: np.ndarray) -> np.ndarray:
        """
        Change the range (low, high) to (-1, 1).
        """
        normed_action = np.copy(action)

        normed_action = (normed_action - self.reloc_factor) / self.scale_factor
        normed_action = np.clip(normed_action, -1.0, 1.0)

        return normed_action
