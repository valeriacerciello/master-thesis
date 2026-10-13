from experiments.random_baseline import run_random_baseline


def test_random_baseline_interaction_count(tmp_path, monkeypatch):
    """Check that the experiment respects its interaction budget."""

    monkeypatch.chdir(tmp_path)

    tracker = run_random_baseline(
        grid_size=5,
        max_steps=10,
        number_of_episodes=3,
        seed=42,
    )

    assert tracker.total_steps == 30
    assert len(tracker.coverage_history) == 30

    # Each episode contributes one initial state.
    assert sum(tracker.explorer_visit_counts.values()) == 33
    assert sum(tracker.joint_visit_counts.values()) == 33


def test_random_baseline_coverage_is_valid(tmp_path, monkeypatch):
    """Check that coverage measurements remain valid."""

    monkeypatch.chdir(tmp_path)

    tracker = run_random_baseline(
        grid_size=5,
        max_steps=10,
        number_of_episodes=3,
        seed=42,
    )

    assert 0.0 <= tracker.get_explorer_coverage() <= 1.0
    assert 0.0 <= tracker.get_joint_coverage() <= 1.0

    assert len(tracker.explorer_visit_counts) <= 25
    assert len(tracker.joint_visit_counts) <= 25 * 24


def test_random_baseline_is_reproducible(tmp_path, monkeypatch):
    """Check that identical seeds produce identical results."""

    monkeypatch.chdir(tmp_path)

    tracker_1 = run_random_baseline(
        grid_size=5,
        max_steps=10,
        number_of_episodes=3,
        seed=42,
    )

    tracker_2 = run_random_baseline(
        grid_size=5,
        max_steps=10,
        number_of_episodes=3,
        seed=42,
    )

    assert (
        tracker_1.explorer_visit_counts
        == tracker_2.explorer_visit_counts
    )

    assert (
        tracker_1.joint_visit_counts
        == tracker_2.joint_visit_counts
    )

    assert tracker_1.coverage_history == tracker_2.coverage_history


def test_random_baseline_saves_figures(tmp_path, monkeypatch):
    """Check that both visualization files are created."""

    monkeypatch.chdir(tmp_path)

    run_random_baseline(
        grid_size=5,
        max_steps=10,
        number_of_episodes=3,
        seed=42,
    )

    coverage_path = (
        tmp_path / "results" / "random_baseline_coverage.png"
    )

    heatmap_path = (
        tmp_path / "results" / "random_baseline_heatmap.png"
    )

    assert coverage_path.is_file()
    assert heatmap_path.is_file()

    assert coverage_path.stat().st_size > 0
    assert heatmap_path.stat().st_size > 0