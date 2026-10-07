import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridWorldEnv(gym.Env):
    """Image-based GridWorld environment for exploration experiments."""

    def __init__(
        self,
        grid_size=10,
        max_steps=100,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    ):
        super().__init__()

        if not 0.0 <= dynamics_stochasticity <= 1.0:
            raise ValueError(
                "dynamics_stochasticity must be between 0 and 1."
            )

        if not 0.0 <= expert_stochasticity <= 1.0:
            raise ValueError(
                "expert_stochasticity must be between 0 and 1."
            )

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.dynamics_stochasticity = dynamics_stochasticity
        self.expert_stochasticity = expert_stochasticity

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
        self._expert_position = None
        self._goal_position = None
        self._step_count = 0
        self._visit_counts = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0

        self._explorer_position = self._sample_position()

        self._expert_position = self._sample_position(
            excluded_positions=[self._explorer_position]
        )

        self._goal_position = self._sample_position(
            excluded_positions=[
                self._explorer_position,
                self._expert_position,
            ]
        )

        self._visit_counts = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=int,
        )

        self._record_explorer_visit()

        observation = self._get_observation()

        return observation, {}

    def step(self, action):
        executed_action = self._apply_dynamics_stochasticity(action)
        direction = self._action_to_direction[executed_action]

        new_position = self._explorer_position + direction

        self._explorer_position = np.clip(
            new_position,
            0,
            self.grid_size - 1,
        )

        self._record_explorer_visit()

        expert_action = self._get_expert_action()
        expert_executed_action = self._apply_dynamics_stochasticity(
            expert_action
        )
        expert_direction = self._action_to_direction[
            expert_executed_action
        ]

        new_expert_position = self._expert_position + expert_direction

        self._expert_position = np.clip(
            new_expert_position,
            0,
            self.grid_size - 1,
        )

        self._step_count += 1

        observation = self._get_observation()

        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "requested_action": action,
            "executed_action": executed_action,
            "expert_requested_action": expert_action,
            "expert_executed_action": expert_executed_action,
        }

        return observation, reward, terminated, truncated, info

    def get_state_coverage(self):
        visited_states = np.count_nonzero(self._visit_counts)
        total_states = self.grid_size * self.grid_size

        return visited_states / total_states

    def _record_explorer_visit(self):
        row, col = self._explorer_position
        self._visit_counts[row, col] += 1

    def _apply_dynamics_stochasticity(self, action):
        if self.np_random.random() >= self.dynamics_stochasticity:
            return action

        alternative_actions = [
            candidate_action
            for candidate_action in range(self.action_space.n)
            if candidate_action != action
        ]

        return int(self.np_random.choice(alternative_actions))

    def _get_expert_action(self):
        deterministic_action = self._get_deterministic_expert_action()

        if self.np_random.random() >= self.expert_stochasticity:
            return deterministic_action

        alternative_actions = [
            candidate_action
            for candidate_action in range(self.action_space.n)
            if candidate_action != deterministic_action
        ]

        return int(self.np_random.choice(alternative_actions))

    def _get_deterministic_expert_action(self):
        expert_row, expert_col = self._expert_position
        goal_row, goal_col = self._goal_position

        if expert_row > goal_row:
            return 0  # UP

        if expert_row < goal_row:
            return 1  # DOWN

        if expert_col > goal_col:
            return 2  # LEFT

        if expert_col < goal_col:
            return 3  # RIGHT

        return 4  # STAY

    def _sample_position(self, excluded_positions=None):
        if excluded_positions is None:
            excluded_positions = []

        while True:
            position = self.np_random.integers(
                low=0,
                high=self.grid_size,
                size=2,
                dtype=int,
            )

            if not any(
                np.array_equal(position, excluded)
                for excluded in excluded_positions
            ):
                return position

    def _get_observation(self):
        observation = np.zeros(
            (self.grid_size, self.grid_size, 3),
            dtype=np.uint8,
        )

        explorer_position = tuple(self._explorer_position)
        expert_position = tuple(self._expert_position)
        goal_position = tuple(self._goal_position)

        positions = {
            explorer_position,
            expert_position,
            goal_position,
        }

        for row, col in positions:
            has_explorer = (row, col) == explorer_position
            has_expert = (row, col) == expert_position
            has_goal = (row, col) == goal_position

            if has_explorer and has_expert and has_goal:
                color = [128, 128, 128]

            elif has_explorer and has_expert:
                color = [255, 0, 255]

            elif has_explorer and has_goal:
                color = [0, 255, 255]

            elif has_expert and has_goal:
                color = [255, 255, 0]

            elif has_explorer:
                color = [255, 255, 255]

            elif has_expert:
                color = [255, 0, 0]

            else:
                color = [0, 255, 0]

            observation[row, col] = color

        return observation