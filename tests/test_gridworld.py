import numpy as np

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


def test_deterministic_expert_reaches_goal():
    env = GridWorldEnv(
        grid_size=10,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )

    env.reset(seed=42)

    for _ in range(20):
        env.step(4)

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