from envs.gridworld import GridWorldEnv


def run_trajectory(seed):
    env = GridWorldEnv(
        grid_size=10,
        max_steps=100,
        dynamics_stochasticity=0.3,
        expert_stochasticity=0.3,
    )

    env.reset(seed=seed)

    trajectory = []

    for _ in range(10):
        observation, reward, terminated, truncated, info = env.step(3)

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

print("Trajectory 1:")
for step in trajectory_1:
    print(step)

print("\nTrajectory 2:")
for step in trajectory_2:
    print(step)

print("\nTrajectories identical:", trajectory_1 == trajectory_2)