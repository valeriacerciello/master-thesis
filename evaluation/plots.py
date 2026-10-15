import matplotlib.pyplot as plt
import numpy as np


def plot_coverage(tracker, save_path=None):
    """
    Plot cumulative coverage against environment interactions.

    Parameters
    ----------
    tracker : CoverageTracker
        Tracker containing coverage history.

    save_path : str or None
        If provided, save the figure to this path.
        Otherwise, display the figure.
    """

    if not tracker.coverage_history:
        raise ValueError("No coverage history available to plot.")

    steps = [
        entry["step"]
        for entry in tracker.coverage_history
    ]

    explorer_coverage = [
        entry["explorer_coverage"] * 100
        for entry in tracker.coverage_history
    ]

    joint_coverage = [
        entry["joint_coverage"] * 100
        for entry in tracker.coverage_history
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        steps,
        explorer_coverage,
        label="Explorer-position coverage",
        linewidth=2,
    )

    ax.plot(
        steps,
        joint_coverage,
        label="Joint-state coverage",
        linewidth=2,
    )

    ax.set_xlabel("Environment interactions")
    ax.set_ylabel("Cumulative coverage (%)")
    ax.set_title("Random Exploration — Cumulative Coverage")

    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_visitation_heatmap(tracker, save_path=None):
    """
    Plot cumulative explorer-position visitation counts.

    Parameters
    ----------
    tracker : CoverageTracker
        Tracker containing cumulative visitation counts.

    save_path : str or None
        If provided, save the figure to this path.
        Otherwise, display the figure.
    """

    grid_size = tracker.grid_size

    # Convert visitation counts into a 2D array.
    visit_counts = np.zeros(
        (grid_size, grid_size),
        dtype=int,
    )

    for (row, col), count in tracker.explorer_visit_counts.items():
        visit_counts[row, col] = count

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        visit_counts,
        cmap="viridis",
        interpolation="nearest",
    )

    # Display the number of visits inside each cell.
    for row in range(grid_size):
        for col in range(grid_size):
            ax.text(
                col,
                row,
                str(visit_counts[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if visit_counts[row, col] < visit_counts.max() / 2
                    else "black"
                ),
            )

    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Random Exploration — Explorer Visitation Heatmap")

    fig.colorbar(
        image,
        ax=ax,
        label="Number of visits",
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_multiseed_coverage(trackers, save_path=None):
    """
    Plot mean cumulative coverage and standard deviation across seeds.

    Parameters
    ----------
    trackers : list of CoverageTracker
        Coverage trackers from independent runs.

    save_path : str or None
        If provided, save the figure to this path.
        Otherwise, display the figure.
    """

    if not trackers:
        raise ValueError("At least one tracker is required.")

    # Extract coverage histories from all runs.
    explorer_histories = np.array(
        [
            [
                entry["explorer_coverage"]
                for entry in tracker.coverage_history
            ]
            for tracker in trackers
        ],
        dtype=float,
    )

    joint_histories = np.array(
        [
            [
                entry["joint_coverage"]
                for entry in tracker.coverage_history
            ]
            for tracker in trackers
        ],
        dtype=float,
    )

    if explorer_histories.ndim != 2 or explorer_histories.shape[1] == 0:
        raise ValueError("Trackers must contain non-empty coverage histories.")

    # All runs must use the same interaction budget.
    if any(
        len(tracker.coverage_history) != explorer_histories.shape[1]
        for tracker in trackers
    ):
        raise ValueError("All runs must have the same number of steps.")

    steps = np.array(
        [
            entry["step"]
            for entry in trackers[0].coverage_history
        ]
    )

    if any(
        [
            entry["step"]
            for entry in tracker.coverage_history
        ] != steps.tolist()
        for tracker in trackers
    ):
        raise ValueError("All runs must use the same step values.")

    # Convert fractions into percentages.
    explorer_histories *= 100
    joint_histories *= 100

    # Calculate mean and standard deviation at every interaction.
    explorer_mean = np.mean(explorer_histories, axis=0)
    explorer_std = np.std(explorer_histories, axis=0)

    joint_mean = np.mean(joint_histories, axis=0)
    joint_std = np.std(joint_histories, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Mean explorer-position coverage.
    explorer_line, = ax.plot(
        steps,
        explorer_mean,
        label="Explorer-position coverage",
        linewidth=2,
    )

    ax.fill_between(
        steps,
        np.maximum(explorer_mean - explorer_std, 0),
        np.minimum(explorer_mean + explorer_std, 100),
        color=explorer_line.get_color(),
        alpha=0.2,
    )

    # Mean joint-state coverage.
    joint_line, = ax.plot(
        steps,
        joint_mean,
        label="Joint-state coverage",
        linewidth=2,
    )

    ax.fill_between(
        steps,
        np.maximum(joint_mean - joint_std, 0),
        np.minimum(joint_mean + joint_std, 100),
        color=joint_line.get_color(),
        alpha=0.2,
    )

    ax.set_xlabel("Environment interactions")
    ax.set_ylabel("Cumulative coverage (%)")
    ax.set_title(
        f"Random Exploration — Mean Coverage Across {len(trackers)} Seeds"
    )

    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(title="Shaded regions: ±1 standard deviation")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()