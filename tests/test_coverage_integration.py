import numpy as np

from envs.gridworld import GridWorldEnv
from evaluation.coverage import CoverageTracker


def test_cumulative_coverage_across_gridworld_episodes():
    env = GridWorldEnv(
        grid_size=5,
        max_steps=10,
        dynamics_stochasticity=0.0,
        expert_stochasticity=0.0,
    )

    tracker = CoverageTracker(grid_size=5)

    number_of_episodes = 2
    steps_per_episode = 10

    for episode in range(number_of_episodes):
        observation, info = env.reset(seed=42 + episode)

        # Record the initial state without counting an interaction.
        tracker.record_state(
            env._explorer_position,
            env._expert_position,
        )

        terminated = False
        truncated = False
        episode_steps = 0

        while not (terminated or truncated):
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            tracker.record_step(
                env._explorer_position,
                env._expert_position,
            )

            episode_steps += 1

        assert episode_steps == steps_per_episode

    # Both episodes contribute to the same cumulative tracker.
    assert tracker.total_steps == 20
    assert len(tracker.coverage_history) == 20

    # The tracker records one initial state per episode
    # and one resulting state per environment interaction.
    assert sum(tracker.explorer_visit_counts.values()) == 22
    assert sum(tracker.joint_visit_counts.values()) == 22

    # Coverage is based on unique states, not total visits.
    assert tracker.get_explorer_coverage() == (
        len(tracker.explorer_visit_counts) / 25
    )

    assert tracker.get_joint_coverage() == (
        len(tracker.joint_visit_counts) / (25 * 24)
    )

    # Coverage cannot decrease as more states are visited.
    explorer_coverage_history = [
        entry["explorer_coverage"]
        for entry in tracker.coverage_history
    ]

    joint_coverage_history = [
        entry["joint_coverage"]
        for entry in tracker.coverage_history
    ]

    assert np.all(np.diff(explorer_coverage_history) >= 0)
    assert np.all(np.diff(joint_coverage_history) >= 0)

    # Both coverage metrics must remain within [0, 1].
    assert 0.0 <= tracker.get_explorer_coverage() <= 1.0
    assert 0.0 <= tracker.get_joint_coverage() <= 1.0