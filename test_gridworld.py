from envs.gridworld import GridWorldEnv


env = GridWorldEnv(grid_size=10, max_steps=100)

observation, info = env.reset(seed=42)

print("Explorer start:", env._explorer_position)
print("Expert start:", env._expert_position)
print("Goal:", env._goal_position)
print()

for step in range(8):
    expert_action = env._get_expert_action()

    # Keep the explorer stationary so that we can isolate
    # and inspect the expert's behaviour.
    observation, reward, terminated, truncated, info = env.step(4)

    print(
        f"Step {step + 1}: "
        f"expert action = {expert_action}, "
        f"expert position = {env._expert_position}"
    )