import numpy as np

from envs.gridworld import GridWorldEnv


env = GridWorldEnv(
    grid_size=5,
    max_steps=100,
    dynamics_stochasticity=0.0,
    expert_stochasticity=0.0,
)

env.reset(seed=42)


def check_pixel(name, position, expected_color):
    observation = env._get_observation()

    row, col = position
    actual_color = observation[row, col]

    print(
        f"{name}: "
        f"expected={expected_color}, "
        f"actual={actual_color}, "
        f"correct={np.array_equal(actual_color, expected_color)}"
    )


# Explorer + Expert
env._explorer_position = np.array([1, 1])
env._expert_position = np.array([1, 1])
env._goal_position = np.array([3, 3])

check_pixel(
    "Explorer + Expert",
    (1, 1),
    [255, 0, 255],
)


# Explorer + Goal
env._explorer_position = np.array([1, 1])
env._expert_position = np.array([3, 3])
env._goal_position = np.array([1, 1])

check_pixel(
    "Explorer + Goal",
    (1, 1),
    [0, 255, 255],
)


# Expert + Goal
env._explorer_position = np.array([3, 3])
env._expert_position = np.array([1, 1])
env._goal_position = np.array([1, 1])

check_pixel(
    "Expert + Goal",
    (1, 1),
    [255, 255, 0],
)


# Explorer + Expert + Goal
env._explorer_position = np.array([1, 1])
env._expert_position = np.array([1, 1])
env._goal_position = np.array([1, 1])

check_pixel(
    "Explorer + Expert + Goal",
    (1, 1),
    [128, 128, 128],
)