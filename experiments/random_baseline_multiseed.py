from pathlib import Path

import numpy as np

from evaluation.plots import plot_multiseed_coverage
from experiments.random_baseline import run_random_baseline


def run_multiseed_baseline():
    """Run random exploration with multiple seeds and summarize coverage."""

    seeds = list(range(10))

    trackers = []

    for seed in seeds:
        print(f"\nRunning experiment with seed {seed}...\n")

        tracker = run_random_baseline(
            grid_size=5,
            max_steps=50,
            number_of_episodes=10,
            seed=seed,
            save_plots=False,
        )

        trackers.append(tracker)

    # Collect final coverage values from each experiment.
    explorer_coverages = np.array(
        [tracker.get_explorer_coverage() for tracker in trackers]
    )

    joint_coverages = np.array(
        [tracker.get_joint_coverage() for tracker in trackers]
    )

    # Calculate mean and standard deviation across seeds.
    explorer_mean = np.mean(explorer_coverages)
    explorer_std = np.std(explorer_coverages)

    joint_mean = np.mean(joint_coverages)
    joint_std = np.std(joint_coverages)

    print("\nMultiple-seed random exploration baseline")
    print("-----------------------------------------")
    print(f"Number of seeds: {len(seeds)}")
    print(f"Interactions per seed: {trackers[0].total_steps}")

    print(
        f"Explorer-position coverage: "
        f"{explorer_mean:.2%} ± {explorer_std:.2%}"
    )

    print(
        f"Joint-state coverage: "
        f"{joint_mean:.2%} ± {joint_std:.2%}"
    )

    # Create the results directory if necessary.
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Plot mean coverage and standard deviation across seeds.
    coverage_path = (
        results_dir / "random_baseline_multiseed_coverage.png"
    )

    plot_multiseed_coverage(
        trackers,
        save_path=coverage_path,
    )

    print(f"Multiple-seed coverage plot saved to: {coverage_path}")

    return trackers


if __name__ == "__main__":
    run_multiseed_baseline()