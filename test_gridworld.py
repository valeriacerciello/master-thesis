from envs.gridworld import GridWorldEnv


env = GridWorldEnv(grid_size=10, max_steps=100)

observation, info = env.reset(seed=42)

print("Observation shape:", observation.shape)
print("Observation dtype:", observation.dtype)
print("Initial explorer position:", env._explorer_position)

action = 3  # RIGHT

observation, reward, terminated, truncated, info = env.step(action)

print("Explorer position after RIGHT:", env._explorer_position)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)