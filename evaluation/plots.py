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