import numpy as np
import pytest

from envs.gridworld import GridWorldEnv


def test_reset_creates_valid_distinct_positions():
    env = GridWorldEnv(grid_size=10)

    observation, info = env.reset(seed=42)

    assert observation.shape == (10, 10, 3)
    assert observation.dtype == np.uint8

    assert not np.array_equal(
        env._explorer_position,
        env._expert_position,
    )
    assert not np.array_equal(
        env._explorer_position,
        env._goal_position,
    )
    assert not np.array_equal(
        env._expert_position,
        env._goal_position,
    )


def test_goal_is_fixed_at_top_right_corner():
    env = GridWorldEnv(grid_size=10)

    expected_goal = np.array([0, 9])

    for seed in [1, 42, 100]:
        env.reset(seed=seed)

        assert np.array_equal(
            env._goal_position,
            expected_goal,
        )


def test_goal_position_adapts_to_grid_size():
    for grid_size in [2, 5, 10, 15]:
        env = GridWorldEnv(grid_size=grid_size)
        env.reset(seed=42)

        expected_goal = np.array([0, grid_size - 1])

        assert np.array_equal(
            env._goal_position,
            expected_goal,
        )


def test_agents_never_start_on_fixed_goal():
    env = GridWorldEnv(grid_size=10)

    for seed in range(20):
        env.reset(seed=seed)

        assert not np.array_equal(
            env._explorer_position,
            env._goal_position,
        )

        assert not np.array_equal(
            env._expert_position,
            env._goal_position,
        )


def test_grid_size_must_be_at_least_two():
    with pytest.raises(ValueError):
        GridWorldEnv(grid_size=1)


def test_deterministic_expert_reaches_goal():
    env = GridWorldEnv(
        grid_size=10,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )

    env.reset(seed=42)

    # Place the explorer away from the expert's path.
    env._explorer_position = np.array([9, 0])
    env._expert_position = np.array([5, 5])

    # The expert needs 5 UP moves and 4 RIGHT moves.
    for _ in range(9):
        _, _, _, _, info = env.step(4)

        assert not info["collision"]

    assert np.array_equal(
        env._expert_position,
        env._goal_position,
    )


def test_deterministic_dynamics_execute_requested_actions():
    env = GridWorldEnv(
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )

    env.reset(seed=42)

    for _ in range(10):
        _, _, _, _, info = env.step(3)

        assert info["executed_action"] == info["requested_action"]
        assert (
            info["expert_executed_action"]
            == info["expert_requested_action"]
        )


def test_fully_stochastic_dynamics_change_requested_actions():
    env = GridWorldEnv(
        dynamics_stochasticity=1.0,
        expert_stochasticity=0.0,
    )

    env.reset(seed=42)

    for _ in range(10):
        _, _, _, _, info = env.step(3)

        assert info["executed_action"] != info["requested_action"]
        assert (
            info["expert_executed_action"]
            != info["expert_requested_action"]
        )


def test_fully_stochastic_expert_changes_deterministic_action():
    env = GridWorldEnv(
        dynamics_stochasticity=0.0,
        expert_stochasticity=1.0,
    )

    env.reset(seed=42)

    for _ in range(10):
        deterministic_action = (
            env._get_deterministic_expert_action()
        )

        _, _, _, _, info = env.step(4)

        assert (
            info["expert_requested_action"]
            != deterministic_action
        )

        assert (
            info["expert_executed_action"]
            == info["expert_requested_action"]
        )


def test_stochastic_trajectory_is_reproducible():
    def run_trajectory(seed):
        env = GridWorldEnv(
            dynamics_stochasticity=0.3,
            expert_stochasticity=0.3,
        )

        env.reset(seed=seed)

        trajectory = []

        for _ in range(10):
            _, _, _, _, info = env.step(3)

            trajectory.append(
                (
                    tuple(env._explorer_position),
                    tuple(env._expert_position),
                    info["executed_action"],
                    info["expert_requested_action"],
                    info["expert_executed_action"],
                )
            )

        return trajectory

    trajectory_1 = run_trajectory(seed=42)
    trajectory_2 = run_trajectory(seed=42)

    assert trajectory_1 == trajectory_2


def test_visit_counts_include_initial_state_and_repeated_visits():
    env = GridWorldEnv(
        grid_size=5,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )

    env.reset(seed=42)

    assert env._visit_counts.sum() == 1

    start_position = tuple(env._explorer_position)

    for _ in range(3):
        env.step(4)

    assert env._visit_counts.sum() == 4
    assert env._visit_counts[start_position] == 4


def test_state_coverage_counts_unique_cells_only():
    env = GridWorldEnv(
        grid_size=5,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )

    env.reset(seed=42)

    assert env.get_state_coverage() == 1 / 25

    for _ in range(3):
        env.step(4)

    assert env.get_state_coverage() == 1 / 25

    env.step(3)

    assert env.get_state_coverage() == 2 / 25


def test_explorer_and_expert_overlap_color():
    env = GridWorldEnv(grid_size=5)
    env.reset(seed=42)

    env._explorer_position = np.array([1, 1])
    env._expert_position = np.array([1, 1])
    env._goal_position = np.array([3, 3])

    observation = env._get_observation()

    assert np.array_equal(
        observation[1, 1],
        [255, 0, 255],
    )


def test_explorer_and_goal_overlap_color():
    env = GridWorldEnv(grid_size=5)
    env.reset(seed=42)

    env._explorer_position = np.array([1, 1])
    env._expert_position = np.array([3, 3])
    env._goal_position = np.array([1, 1])

    observation = env._get_observation()

    assert np.array_equal(
        observation[1, 1],
        [0, 255, 255],
    )


def test_expert_and_goal_overlap_color():
    env = GridWorldEnv(grid_size=5)
    env.reset(seed=42)

    env._explorer_position = np.array([3, 3])
    env._expert_position = np.array([1, 1])
    env._goal_position = np.array([1, 1])

    observation = env._get_observation()

    assert np.array_equal(
        observation[1, 1],
        [255, 255, 0],
    )


def test_all_entities_overlap_color():
    env = GridWorldEnv(grid_size=5)
    env.reset(seed=42)

    env._explorer_position = np.array([1, 1])
    env._expert_position = np.array([1, 1])
    env._goal_position = np.array([1, 1])

    observation = env._get_observation()

    assert np.array_equal(
        observation[1, 1],
        [128, 128, 128],
    )


def test_same_cell_collision_blocks_both_agents():
    env = GridWorldEnv(
        grid_size=5,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )
    env.reset(seed=42)

    env._explorer_position = np.array([0, 3])
    env._expert_position = np.array([0, 4])

    _, _, _, _, info = env.step(3)

    assert np.array_equal(
        env._explorer_position,
        [0, 3],
    )
    assert np.array_equal(
        env._expert_position,
        [0, 4],
    )
    assert info["collision"]
    assert info["same_cell_collision"]
    assert not info["position_swap_collision"]


def test_position_swap_collision_blocks_both_agents():
    env = GridWorldEnv(
        grid_size=5,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )
    env.reset(seed=42)

    env._explorer_position = np.array([1, 2])
    env._expert_position = np.array([2, 2])

    _, _, _, _, info = env.step(1)

    assert np.array_equal(
        env._explorer_position,
        [1, 2],
    )
    assert np.array_equal(
        env._expert_position,
        [2, 2],
    )
    assert info["collision"]
    assert info["position_swap_collision"]
    assert not info["same_cell_collision"]


def test_non_colliding_actions_move_both_agents():
    env = GridWorldEnv(
        grid_size=5,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )
    env.reset(seed=42)

    env._explorer_position = np.array([3, 0])
    env._expert_position = np.array([3, 3])

    _, _, _, _, info = env.step(3)

    assert np.array_equal(
        env._explorer_position,
        [3, 1],
    )
    assert np.array_equal(
        env._expert_position,
        [2, 3],
    )
    assert not info["collision"]
    assert not info["same_cell_collision"]
    assert not info["position_swap_collision"]


def test_collision_records_visit_to_current_position():
    env = GridWorldEnv(
        grid_size=5,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )
    env.reset(seed=42)

    env._explorer_position = np.array([0, 3])
    env._expert_position = np.array([0, 4])

    initial_visits = env._visit_counts[0, 3]

    _, _, _, _, info = env.step(3)

    assert info["collision"]
    assert env._visit_counts[0, 3] == initial_visits + 1


def test_joint_coverage_records_initial_state():
    env = GridWorldEnv(grid_size=5)
    env.reset(seed=42)

    joint_state = (
        int(env._explorer_position[0]),
        int(env._explorer_position[1]),
        int(env._expert_position[0]),
        int(env._expert_position[1]),
    )

    assert env._joint_visit_counts == {joint_state: 1}

    # 25 possible cells, with 24 choices for the other agent.
    assert env.get_joint_state_coverage() == 1 / (25 * 24)


def test_joint_coverage_repeated_visits_do_not_increase_coverage():
    env = GridWorldEnv(
        grid_size=5,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )
    env.reset(seed=42)

    # Put the expert at its goal and the explorer elsewhere.
    env._explorer_position = np.array([4, 0])
    env._expert_position = np.array([0, 4])

    # Reset tracking to the manually constructed joint state.
    env._joint_visit_counts = {}
    env._record_joint_visit()

    joint_state = (4, 0, 0, 4)

    for _ in range(3):
        env.step(4)

    assert env._joint_visit_counts[joint_state] == 4
    assert env.get_joint_state_coverage() == 1 / (25 * 24)


def test_joint_coverage_distinguishes_expert_positions():
    env = GridWorldEnv(
        grid_size=5,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )
    env.reset(seed=42)

    # The explorer stays still while the expert moves toward the goal.
    env._explorer_position = np.array([4, 0])
    env._expert_position = np.array([2, 3])

    env._visit_counts = np.zeros((5, 5), dtype=int)
    env._joint_visit_counts = {}

    env._record_explorer_visit()
    env._record_joint_visit()

    initial_explorer_coverage = env.get_state_coverage()

    env.step(4)

    # Explorer has not moved, so explorer coverage is unchanged.
    assert env.get_state_coverage() == initial_explorer_coverage

    # Expert has moved, creating a new joint state.
    assert len(env._joint_visit_counts) == 2
    assert env.get_joint_state_coverage() == 2 / (25 * 24)


def test_joint_coverage_collision_counts_repeated_state():
    env = GridWorldEnv(
        grid_size=5,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )
    env.reset(seed=42)

    # Explorer attempts to enter the expert's goal cell.
    env._explorer_position = np.array([0, 3])
    env._expert_position = np.array([0, 4])

    env._joint_visit_counts = {}
    env._record_joint_visit()

    _, _, _, _, info = env.step(3)

    assert info["collision"]
    assert env._joint_visit_counts[(0, 3, 0, 4)] == 2
    assert len(env._joint_visit_counts) == 1
    assert env.get_joint_state_coverage() == 1 / (25 * 24)