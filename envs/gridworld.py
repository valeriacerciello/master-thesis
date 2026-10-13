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

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

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

        # Fixed goal in the top-right corner.
        self._goal_position = np.array(
            [0, self.grid_size - 1],
            dtype=int,
        )

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
        self._step_count = 0
        self._visit_counts = None
        self._joint_visit_counts = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0

        # Both agents start at random, distinct positions.
        # Neither agent starts on the fixed goal.
        self._explorer_position = self._sample_position(
            excluded_positions=[self._goal_position]
        )

        self._expert_position = self._sample_position(
            excluded_positions=[
                self._goal_position,
                self._explorer_position,
            ]
        )

        self._visit_counts = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=int,
        )

        self._joint_visit_counts = {}

        self._record_explorer_visit()
        self._record_joint_visit()

        observation = self._get_observation()

        return observation, {}

    def step(self, action):
        # 1. Save the current joint state.
        current_explorer_position = self._explorer_position.copy()
        current_expert_position = self._expert_position.copy()

        # 2. Determine both requested actions.
        explorer_action = action
        expert_action = self._get_expert_action()

        # 3. Apply environment stochasticity independently.
        explorer_executed_action = self._apply_dynamics_stochasticity(
            explorer_action
        )

        expert_executed_action = self._apply_dynamics_stochasticity(
            expert_action
        )

        # 4. Calculate both intended next positions.
        explorer_direction = self._action_to_direction[
            explorer_executed_action
        ]

        expert_direction = self._action_to_direction[
            expert_executed_action
        ]

        intended_explorer_position = np.clip(
            current_explorer_position + explorer_direction,
            0,
            self.grid_size - 1,
        )

        intended_expert_position = np.clip(
            current_expert_position + expert_direction,
            0,
            self.grid_size - 1,
        )

        # 5. Check for collisions.
        same_cell_collision = np.array_equal(
            intended_explorer_position,
            intended_expert_position,
        )

        position_swap_collision = (
            np.array_equal(
                intended_explorer_position,
                current_expert_position,
            )
            and np.array_equal(
                intended_expert_position,
                current_explorer_position,
            )
        )

        collision = (
            same_cell_collision
            or position_swap_collision
        )

        # 6. Update both agents simultaneously.
        if collision:
            self._explorer_position = current_explorer_position
            self._expert_position = current_expert_position
        else:
            self._explorer_position = intended_explorer_position
            self._expert_position = intended_expert_position

        # 7. Record the actual visited positions and joint state.
        self._record_explorer_visit()
        self._record_joint_visit()

        self._step_count += 1

        observation = self._get_observation()

        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "requested_action": explorer_action,
            "executed_action": explorer_executed_action,
            "expert_requested_action": expert_action,
            "expert_executed_action": expert_executed_action,
            "collision": collision,
            "same_cell_collision": same_cell_collision,
            "position_swap_collision": position_swap_collision,
        }

        return observation, reward, terminated, truncated, info

    def get_state_coverage(self):
        visited_states = np.count_nonzero(self._visit_counts)
        total_states = self.grid_size * self.grid_size

        return visited_states / total_states

    def get_joint_state_coverage(self):
        visited_joint_states = len(self._joint_visit_counts)

        total_cells = self.grid_size * self.grid_size
        total_joint_states = total_cells * (total_cells - 1)

        return visited_joint_states / total_joint_states

    def _record_explorer_visit(self):
        row, col = self._explorer_position
        self._visit_counts[row, col] += 1

    def _record_joint_visit(self):
        joint_state = (
            int(self._explorer_position[0]),
            int(self._explorer_position[1]),
            int(self._expert_position[0]),
            int(self._expert_position[1]),
        )

        self._joint_visit_counts[joint_state] = (
            self._joint_visit_counts.get(joint_state, 0) + 1
        )

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