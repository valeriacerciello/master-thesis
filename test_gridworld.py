import matplotlib.pyplot as plt

from envs.gridworld import GridWorldEnv


env = GridWorldEnv(
    grid_size=10,
    max_steps=100,
    dynamics_stochasticity=0.0,
    expert_stochasticity=0.0,
)

env.reset(seed=42)

# Random exploration for 100 steps.
for _ in range(100):
    action = env.action_space.sample()
    env.step(action)

print("Total visits:", env._visit_counts.sum())
print("Unique cells visited:", (env._visit_counts > 0).sum())
print("State coverage:", env.get_state_coverage())
print(
    "State coverage percentage:",
    env.get_state_coverage() * 100,
)

plt.imshow(env._visit_counts)
plt.colorbar(label="Number of visits")
plt.title("Explorer visitation heatmap")
plt.xlabel("Column")
plt.ylabel("Row")
plt.xticks(range(env.grid_size))
plt.yticks(range(env.grid_size))

plt.show()