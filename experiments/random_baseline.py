from pathlib import Path

from envs.gridworld import GridWorldEnv
from evaluation.coverage import CoverageTracker
from evaluation.plots import plot_coverage, plot_visitation_heatmap


def run_random_baseline(
    grid_size=5,
    max_steps=50,
    number_of_episodes=10,
    seed=42,
    save_plots=True,
    dynamics_stochasticity=0.0,
    expert_stochasticity=0.0,
):
    """Run random exploration and track cumulative coverage."""

    env = GridWorldEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        dynamics_stochasticity=dynamics_stochasticity,
        expert_stochasticity=expert_stochasticity,
    )

    tracker = CoverageTracker(grid_size=grid_size)

    # Use a dedicated random generator for the explorer's actions.
    env.action_space.seed(seed)

    for episode in range(number_of_episodes):
        observation, info = env.reset(seed=seed + episode)

        # Record the initial joint state.
        tracker.record_state(
            env._explorer_position,
            env._expert_position,
        )

        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Random exploration: every action has equal probability.
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            tracker.record_step(
                env._explorer_position,
                env._expert_position,
            )

    print("Random exploration baseline")
    print("---------------------------")
    print(f"Seed: {seed}")
    print(f"Grid size: {grid_size} × {grid_size}")
    print(f"Episodes: {number_of_episodes}")
    print(f"Total interactions: {tracker.total_steps}")
    print(f"Dynamics stochasticity: {dynamics_stochasticity}")
    print(f"Expert stochasticity: {expert_stochasticity}")

    print(
        f"Explorer-position coverage: "
        f"{tracker.get_explorer_coverage():.2%}"
    )

    print(
        f"Joint-state coverage: "
        f"{tracker.get_joint_coverage():.2%}"
    )

    if save_plots:
        # Create the results directory if necessary.
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Save the cumulative coverage curves.
        coverage_path = results_dir / "random_baseline_coverage.png"

        plot_coverage(
            tracker,
            save_path=coverage_path,
        )

        print(f"Coverage plot saved to: {coverage_path}")

        # Save the explorer-position visitation heatmap.
        heatmap_path = results_dir / "random_baseline_heatmap.png"

        plot_visitation_heatmap(
            tracker,
            save_path=heatmap_path,
        )

        print(f"Visitation heatmap saved to: {heatmap_path}")

    return tracker


if __name__ == "__main__":
    run_random_baseline()