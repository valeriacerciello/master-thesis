import numpy as np

from experiments.random_baseline import run_random_baseline


def run_expert_stochasticity_experiment():
    """Compare random exploration under different expert stochasticity levels."""

    stochasticity_levels = [0.0, 0.2, 0.5]
    seeds = range(10)

    results = {}

    for stochasticity in stochasticity_levels:
        print("\n" + "=" * 50)
        print(f"Expert stochasticity: {stochasticity}")
        print("=" * 50)

        explorer_coverages = []
        joint_coverages = []

        for seed in seeds:
            tracker = run_random_baseline(
                grid_size=5,
                max_steps=50,
                number_of_episodes=10,
                seed=seed,
                save_plots=False,
                dynamics_stochasticity=0.0,
                expert_stochasticity=stochasticity,
            )

            explorer_coverages.append(
                tracker.get_explorer_coverage()
            )

            joint_coverages.append(
                tracker.get_joint_coverage()
            )

        # Calculate statistics across the 10 seeds.
        explorer_mean = np.mean(explorer_coverages)
        explorer_std = np.std(explorer_coverages)

        joint_mean = np.mean(joint_coverages)
        joint_std = np.std(joint_coverages)

        results[stochasticity] = {
            "explorer_mean": explorer_mean,
            "explorer_std": explorer_std,
            "joint_mean": joint_mean,
            "joint_std": joint_std,
        }

    # Display a summary after all experiments finish.
    print("\n" + "=" * 60)
    print("EXPERT STOCHASTICITY COMPARISON")
    print("=" * 60)

    print(f"{'Stochasticity':<15} {'Explorer coverage':<22} {'Joint coverage'}")
    print("-" * 60)

    for stochasticity, stats in results.items():
        explorer_summary = (
            f"{stats['explorer_mean']:.2%} ± "
            f"{stats['explorer_std']:.2%}"
        )

        joint_summary = (
            f"{stats['joint_mean']:.2%} ± "
            f"{stats['joint_std']:.2%}"
        )

        print(
            f"{stochasticity:<15.1f} "
            f"{explorer_summary:<22} "
            f"{joint_summary}"
        )

    return results


if __name__ == "__main__":
    run_expert_stochasticity_experiment()