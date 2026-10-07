import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridWorldEnv(gym.Env):
    """Image-based GridWorld environment for exploration experiments."""

    def __init__(self, grid_size=10, max_steps=100):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(grid_size, grid_size, 3),
            dtype=np.uint8,
        )

        self._action_to_direction = {
            0: np.array([-1, 0]),  # UP
            1: np.array([1, 0]),   # DOWN
            2: np.array([0, -1]),  # LEFT
            3: np.array([0, 1]),   # RIGHT
            4: np.array([0, 0]),   # STAY
        }

        self._explorer_position = None
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0

        self._explorer_position = self.np_random.integers(
            low=0,
            high=self.grid_size,
            size=2,
            dtype=int,
        )

        observation = self._get_observation()

        return observation, {}

    def step(self, action):
        direction = self._action_to_direction[action]

        new_position = self._explorer_position + direction

        self._explorer_position = np.clip(
            new_position,
            0,
            self.grid_size - 1,
        )

        self._step_count += 1

        observation = self._get_observation()

        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps

        return observation, reward, terminated, truncated, {}

    def _get_observation(self):
        observation = np.zeros(
            (self.grid_size, self.grid_size, 3),
            dtype=np.uint8,
        )

        row, col = self._explorer_position
        observation[row, col] = [255, 255, 255]

        return observation