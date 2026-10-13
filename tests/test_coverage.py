import pytest

from evaluation.coverage import CoverageTracker


def test_initial_state_does_not_increment_step_counter():
    tracker = CoverageTracker(grid_size=5)

    tracker.record_state(
        explorer_position=(1, 1),
        expert_position=(3, 3),
    )

    assert tracker.total_steps == 0
    assert tracker.coverage_history == []

    assert tracker.get_explorer_coverage() == 1 / 25
    assert tracker.get_joint_coverage() == 1 / (25 * 24)


def test_repeated_visits_do_not_increase_coverage():
    tracker = CoverageTracker(grid_size=5)

    tracker.record_state((1, 1), (3, 3))

    for _ in range(3):
        tracker.record_step((1, 1), (3, 3))

    assert tracker.total_steps == 3

    assert tracker.explorer_visit_counts[(1, 1)] == 4
    assert tracker.joint_visit_counts[(1, 1, 3, 3)] == 4

    assert tracker.get_explorer_coverage() == 1 / 25
    assert tracker.get_joint_coverage() == 1 / (25 * 24)


def test_coverage_accumulates_across_episodes():
    tracker = CoverageTracker(grid_size=5)

    # Episode 1
    tracker.record_state((1, 1), (3, 3))
    tracker.record_step((1, 2), (3, 3))
    tracker.record_step((1, 3), (3, 3))

    # Episode 2: the initial state has already been visited.
    tracker.record_state((1, 1), (3, 3))
    tracker.record_step((1, 2), (2, 3))
    tracker.record_step((2, 2), (2, 3))

    assert tracker.total_steps == 4

    # Four distinct explorer positions:
    # (1,1), (1,2), (1,3), (2,2)
    assert len(tracker.explorer_visit_counts) == 4

    # Five distinct joint states.
    assert len(tracker.joint_visit_counts) == 5

    assert tracker.get_explorer_coverage() == 4 / 25
    assert tracker.get_joint_coverage() == 5 / (25 * 24)


def test_coverage_history_tracks_environment_interactions():
    tracker = CoverageTracker(grid_size=5)

    tracker.record_state((1, 1), (3, 3))

    tracker.record_step((1, 2), (3, 3))
    tracker.record_step((1, 3), (3, 3))
    tracker.record_step((1, 3), (3, 3))

    assert len(tracker.coverage_history) == 3

    assert [entry["step"] for entry in tracker.coverage_history] == [
        1,
        2,
        3,
    ]

    assert [
        entry["explorer_coverage"]
        for entry in tracker.coverage_history
    ] == [
        2 / 25,
        3 / 25,
        3 / 25,
    ]

    assert [
        entry["joint_coverage"]
        for entry in tracker.coverage_history
    ] == [
        2 / (25 * 24),
        3 / (25 * 24),
        3 / (25 * 24),
    ]


def test_invalid_positions_are_rejected():
    tracker = CoverageTracker(grid_size=5)

    with pytest.raises(ValueError):
        tracker.record_state((1, 1), (1, 1))

    with pytest.raises(ValueError):
        tracker.record_state((-1, 0), (3, 3))

    with pytest.raises(ValueError):
        tracker.record_state((5, 0), (3, 3))

    with pytest.raises(ValueError):
        tracker.record_state((1,), (3, 3))


def test_reset_clears_cumulative_statistics():
    tracker = CoverageTracker(grid_size=5)

    tracker.record_state((1, 1), (3, 3))
    tracker.record_step((1, 2), (3, 3))

    tracker.reset()

    assert tracker.total_steps == 0
    assert tracker.explorer_visit_counts == {}
    assert tracker.joint_visit_counts == {}
    assert tracker.coverage_history == []

    assert tracker.get_explorer_coverage() == 0.0
    assert tracker.get_joint_coverage() == 0.0


def test_invalid_grid_size_is_rejected():
    with pytest.raises(ValueError):
        CoverageTracker(grid_size=1)