class CoverageTracker:
    """Track cumulative state coverage across multiple episodes."""

    def __init__(self, grid_size):
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

        self.grid_size = grid_size

        total_cells = grid_size * grid_size

        self.total_explorer_states = total_cells
        self.total_joint_states = total_cells * (total_cells - 1)

        self.reset()

    def reset(self):
        """Reset all cumulative statistics."""

        self.explorer_visit_counts = {}
        self.joint_visit_counts = {}

        self.total_steps = 0
        self.coverage_history = []

    def record_state(self, explorer_position, expert_position):
        """Record a visited state without incrementing the step counter."""

        explorer_state = tuple(int(x) for x in explorer_position)
        expert_state = tuple(int(x) for x in expert_position)

        if len(explorer_state) != 2 or len(expert_state) != 2:
            raise ValueError("Each agent position must contain two coordinates.")

        if explorer_state == expert_state:
            raise ValueError("Agents cannot occupy the same cell.")

        for row, col in (explorer_state, expert_state):
            if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
                raise ValueError("Agent position is outside the grid.")

        joint_state = explorer_state + expert_state

        self.explorer_visit_counts[explorer_state] = (
            self.explorer_visit_counts.get(explorer_state, 0) + 1
        )

        self.joint_visit_counts[joint_state] = (
            self.joint_visit_counts.get(joint_state, 0) + 1
        )

    def record_step(self, explorer_position, expert_position):
        """Record the state reached after one environment interaction."""

        self.record_state(explorer_position, expert_position)

        self.total_steps += 1

        self.coverage_history.append(
            {
                "step": self.total_steps,
                "explorer_coverage": self.get_explorer_coverage(),
                "joint_coverage": self.get_joint_coverage(),
            }
        )

    def get_explorer_coverage(self):
        """Return cumulative explorer-position coverage."""

        return (
            len(self.explorer_visit_counts)
            / self.total_explorer_states
        )

    def get_joint_coverage(self):
        """Return cumulative joint-state coverage."""

        return (
            len(self.joint_visit_counts)
            / self.total_joint_states
        )