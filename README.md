# Master's Thesis — Exploration for Deep Multi-Agent Imitation Learning

This repository contains the experimental work for my Master's thesis on
exploration in deep multi-agent imitation learning (MAIL).

## 1. Research Goal

The goal is to investigate whether unsupervised reinforcement learning
exploration methods can improve the exploration phase of deep MAIL.

The initial experiments will compare two exploration strategies based on
different principles:

- **ProtoRL:** state novelty and coverage.
- **Disagreement:** uncertainty about environment dynamics.

The main research question is not simply whether an exploration method
achieves broad state coverage, but whether it visits states that provide
useful information for recovering the expert policy.

In particular, the experiments aim to investigate:

1. How different exploration strategies behave in multi-agent environments.
2. Whether they achieve different forms of state coverage.
3. How their performance changes under stochastic expert policies and
   stochastic environment dynamics.
4. Whether better exploration coverage translates into better
   expert-policy recovery.
5. Whether exploration strategies should specifically target states
   informative about expert behaviour.

The objective is not to reproduce the experiments from the reference
MAIL implementation.

Instead, the reference implementation provides a starting point for
understanding the exploration problem and designing experiments that
isolate the effects of different exploration strategies.

---

## 2. Initial Experimental Setup

The first experiments use a deliberately simple image-based GridWorld.

Starting with a simple environment allows us to understand exploration
behaviour before introducing more complex environments and learning
algorithms.

### 2.1 Image-based observations

The environment provides RGB observations with shape:

`(grid_size, grid_size, 3)`

Each grid cell is represented by a pixel, and different entities are
represented by different colours.

The observation encodes:

- Explorer position.
- Expert position.
- Goal position.

This provides a simple image-based environment suitable for experimenting
with representation learning and deep reinforcement learning.

### 2.2 Two-agent environment

The environment contains two agents:

**Explorer**

The explorer is controlled by the exploration algorithm.

Its purpose is to visit states that may provide useful information about
the fixed expert policy.

**Expert**

The expert follows a fixed goal-directed policy.

In the current implementation, the expert is a simple heuristic agent,
not a policy obtained by solving a Markov game.

Both agents move in the same environment, and their actions jointly
determine the next state.

The explorer can influence the expert's movement through collisions.

However, the expert does not strategically adapt its policy to the
explorer.

This distinction is important because the current environment is
primarily intended for developing and testing exploration methods.

It does not yet support a complete game-theoretic evaluation of MAIL.

### 2.3 Goal position

The goal is fixed in the top-right corner:

`goal_position = (0, grid_size - 1)`

For a 5 × 5 grid:

`goal_position = (0, 4)`

For a 10 × 10 grid:

`goal_position = (0, 9)`

The goal remains unchanged across episodes.

At the beginning of each episode:

- The explorer starts at a randomly sampled position.
- The expert starts at a randomly sampled position.
- The two agents start in different cells.
- Neither agent starts on the goal.

A fixed goal reduces experimental variability and simplifies the
interpretation of exploration behaviour.

Randomized goals may be considered in later experiments.

### 2.4 Action space

The environment supports five discrete actions:

| Action | Movement |
|---|---|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |
| 4 | Stay |

Actions that would move an agent outside the grid are clipped to
the grid boundaries.

### 2.5 Simultaneous movement

Both agents move simultaneously.

At each environment interaction:

1. The explorer selects an action.
2. The expert selects an action according to its fixed policy.
3. Environment stochasticity is applied independently to both actions.
4. Both intended next positions are calculated.
5. Collisions are checked.
6. Both agents' positions are updated simultaneously.

The next joint state therefore depends on the actions of both agents.

### 2.6 Collision handling

Two types of collisions are implemented.

**Same-cell collision**

If both agents attempt to occupy the same cell, neither agent moves.

**Position-swap collision**

If the agents attempt to exchange positions in a single step,
neither agent moves.

When no collision occurs, both agents move to their intended positions.

Collision information is returned through the environment's `info`
dictionary:

- `collision`
- `same_cell_collision`
- `position_swap_collision`

Collisions introduce interactions between the explorer and the expert.

For example, the explorer may block the expert's preferred movement.

However, the current expert does not explicitly plan around obstacles
or choose alternative paths when blocked.

### 2.7 Expert policy

The deterministic expert follows a goal-directed heuristic.

Its action-selection rule is:

1. Move vertically toward the goal.
2. Once the correct row is reached, move horizontally toward the goal.
3. Stay when the goal is reached.

The requested action depends on the expert's position relative to
the goal.

It does not explicitly depend on the explorer's position.

Nevertheless, the explorer can influence the expert's actual trajectory
through collision handling.

The expert is therefore interactive at the level of environment
dynamics, but it is not a strategic best-response policy.

This is a limitation of the initial setup.

### 2.8 Sources of stochasticity

Two sources of stochasticity are independently configurable.

#### Environment stochasticity

The parameter:

`dynamics_stochasticity`

controls the probability that a requested action is replaced by a
different randomly selected action.

The replacement action is sampled uniformly from the other available
actions.

This mechanism is applied independently to the explorer and expert.

- `0.0`: requested actions are always executed.
- `1.0`: requested actions are always replaced by another action.

The resulting executed actions determine the intended movements and
possible collisions.

#### Expert stochasticity

The parameter:

`expert_stochasticity`

controls the probability that the expert deviates from its deterministic
goal-following action.

When a deviation occurs, another action is sampled uniformly from
the remaining actions.

- `0.0`: the expert always requests its deterministic action.
- `1.0`: the expert always requests a different action.

These definitions are implementation choices and may be revisited
depending on the experimental objectives.

### 2.9 Experimental configurations

The two sources of stochasticity allow four configurations:

| Configuration | Dynamics | Expert |
|---|---|---|
| 1 | Deterministic | Deterministic |
| 2 | Deterministic | Stochastic |
| 3 | Stochastic | Deterministic |
| 4 | Stochastic | Stochastic |

These configurations are intended to investigate how exploration
strategies behave under different forms of uncertainty.

They are particularly relevant for Disagreement, whose intrinsic
reward depends on uncertainty in predicted transitions.

For the initial random baseline, only the fully deterministic
configuration has been tested.

---

## 3. Coverage Evaluation

The evaluation infrastructure distinguishes between:

1. Explorer-position coverage.
2. Joint-state coverage.

Both metrics are available at the episode level and cumulatively
across multiple episodes.

### 3.1 Explorer-position coverage

Explorer-position coverage measures the fraction of grid cells
visited by the explorer.

For a grid containing N cells:

`C_explorer = unique_explorer_positions / N`

For a 5 × 5 grid:

`N = 25`

The episode-level metric is available through:

```python
env.get_state_coverage()
```

The corresponding visitation counts are stored in:

```python
env._visit_counts
```

### 3.2 Joint-state coverage

Joint-state coverage considers the positions of both agents.

A joint state is represented as:

`(explorer_row, explorer_col, expert_row, expert_col)`

For example:

`(1, 2, 3, 4)`

and:

`(1, 2, 4, 3)`

represent different joint states.

This distinction is important because the explorer may occupy the
same cell while encountering the expert in different positions.

#### Number of possible joint states

If the grid contains N cells, and both agents must occupy
different cells, the number of valid joint states is:

`N × (N - 1)`

For a 5 × 5 grid:

`25 × 24 = 600`

For a 10 × 10 grid:

`100 × 99 = 9900`

The joint-state coverage metric is:

`C_joint = unique_joint_states / (N × (N - 1))`

The episode-level metric is available through:

```python
env.get_joint_state_coverage()
```

The corresponding visitation counts are stored in:

```python
env._joint_visit_counts
```

The denominator counts all configurations with distinct agent
positions, not only those reachable under a particular expert
policy and interaction budget.

### 3.3 Episode-level coverage

The GridWorld records:

- The initial state when `reset()` is called.
- The resulting state after every environment interaction.

Repeated visits increase visitation counts but do not increase
the number of unique states.

When a collision prevents movement, the resulting state is
recorded again.

Episode-level visitation counts are reset whenever the
environment is reset.

---

## 4. Cumulative Coverage Tracking

Cumulative coverage is implemented in:

`evaluation/coverage.py`

The `CoverageTracker` class records coverage across multiple
episodes.

This component is independent of the GridWorld environment.

The separation allows the same evaluation procedure to be reused
for different exploration algorithms.

### 4.1 Why cumulative coverage?

Episode-level coverage measures the states visited during
one episode.

However, exploration algorithms typically interact with the
environment across many episodes.

We therefore want to measure the total number of distinct
states discovered throughout the exploration process.

For example:

| Episode | States visited |
|---|---|
| 1 | A, B, C |
| 2 | B, C, D |
| 3 | C, D, E |

The cumulative set of visited states is:

`{A, B, C, D, E}`

Therefore, cumulative coverage is based on the union of visited
states rather than the sum of episode-level coverage values.

### 4.2 CoverageTracker

Create a tracker using:

```python
from evaluation.coverage import CoverageTracker

tracker = CoverageTracker(grid_size=5)
```

The tracker maintains:

- `explorer_visit_counts`
- `joint_visit_counts`
- `total_steps`
- `coverage_history`

### 4.3 Recording initial states

At the beginning of each episode:

```python
observation, info = env.reset()

tracker.record_state(
    env._explorer_position,
    env._expert_position,
)
```

Recording the initial state updates visitation counts without
incrementing the environment interaction counter.

### 4.4 Recording environment interactions

After each environment step:

```python
observation, reward, terminated, truncated, info = env.step(action)

tracker.record_step(
    env._explorer_position,
    env._expert_position,
)
```

This method:

1. Records the resulting explorer position.
2. Records the resulting joint state.
3. Increments the environment interaction counter.
4. Saves the current cumulative coverage values.

### 4.5 Cumulative coverage metrics

Cumulative explorer-position coverage:

```python
tracker.get_explorer_coverage()
```

Cumulative joint-state coverage:

```python
tracker.get_joint_coverage()
```

Both values are between 0 and 1.

### 4.6 Coverage history

The tracker stores coverage after every environment interaction.

Each entry contains:

```python
{
    "step": 1,
    "explorer_coverage": 0.08,
    "joint_coverage": 0.0033,
}
```

These values are illustrative.

The history allows coverage to be plotted against the number of
environment interactions.

This will be useful for comparing the exploration efficiency of
different algorithms under identical interaction budgets.

### 4.7 Episode-level versus cumulative coverage

| Property | Episode-level | Cumulative |
|---|---|---|
| Component | GridWorld | CoverageTracker |
| Resets between episodes | Yes | No |
| Explorer-position coverage | Yes | Yes |
| Joint-state coverage | Yes | Yes |
| Tracks coverage over interactions | No | Yes |
| Main purpose | Analyze individual episodes | Compare exploration methods |

---

## 5. Random Exploration Baseline

The first exploration baseline is implemented in:

`experiments/random_baseline.py`

The explorer selects actions uniformly at random.

No intrinsic reward or learned exploration policy is used.

This baseline provides a simple reference against which ProtoRL
and Disagreement can eventually be compared.

### 5.1 Experimental configuration

The initial experiment uses:

| Parameter | Value |
|---|---|
| Grid size | 5 × 5 |
| Maximum steps per episode | 50 |
| Number of episodes | 10 |
| Total environment interactions | 500 |
| Random seed | 42 |
| Environment dynamics | Deterministic |
| Expert policy | Deterministic |
| Explorer policy | Uniform random |

The expert follows the goal-directed heuristic described earlier.

The explorer selects one of the five actions with equal probability.

### 5.2 Reproducibility

The experiment uses separate random generators for:

- Explorer actions.
- Environment transitions and initial positions.

The explorer's action space is seeded using the experiment seed.

Each episode resets the environment with a deterministic seed
derived from the experiment seed and episode index.

This makes the experiment reproducible when using the same
configuration and software environment.

### 5.3 Running the experiment

From the project root:

```bash
python -m experiments.random_baseline
```

The script:

1. Creates the GridWorld.
2. Creates a cumulative coverage tracker.
3. Runs the specified number of episodes.
4. Records initial states and environment interactions.
5. Prints the final coverage values.
6. Generates the coverage curves.
7. Generates the visitation heatmap.

### 5.4 Initial results

The first run produced:

| Metric | Result |
|---|---:|
| Total environment interactions | 500 |
| Explorer-position coverage | 100% |
| Joint-state coverage | 12% |
| Unique explorer positions | 25 |
| Unique joint states | 72 |

The explorer visited every cell in the grid.

However, it visited only 72 of the 600 possible joint states.

This illustrates that complete explorer-position coverage does not
imply complete joint-state coverage.

These results are preliminary and correspond to a single random seed.

They should not be interpreted as representative performance
estimates.

### 5.5 Interpretation

Explorer-position coverage reached 100% after approximately
190 environment interactions.

After this point, the explorer continued visiting previously
discovered cells.

Joint-state coverage continued increasing throughout the
experiment, reaching 12% after 500 interactions.

This suggests that the explorer continued discovering new
configurations involving the expert even after all individual
grid cells had been visited.

However, neither metric directly measures whether the collected
states are useful for expert-policy recovery.

The usefulness of the visited states for imitation learning
remains an open research question.

---

## 6. Visualization

Visualization functions are implemented in:

`evaluation/plots.py`

The current implementation provides:

1. Cumulative coverage curves.
2. Explorer-position visitation heatmaps.

Both figures are automatically generated by the random baseline
experiment.

### 6.1 Cumulative coverage curves

The coverage plot is saved to:

`results/random_baseline_coverage.png`

It displays:

- Explorer-position coverage.
- Joint-state coverage.

Both metrics are plotted against environment interactions.

This makes it possible to investigate how quickly different
exploration methods discover new states.

The plot uses percentages, but the two metrics have different
denominators.

Therefore, their values should be interpreted separately.

### 6.2 Explorer-position visitation heatmap

The visitation heatmap is saved to:

`results/random_baseline_heatmap.png`

It displays the cumulative number of visits to each grid cell.

Unlike coverage, which only distinguishes visited and unvisited
states, the heatmap shows how frequently each position is visited.

The initial experiment produced the following observations:

- All 25 cells were visited.
- The most frequently visited cell was `(0, 3)`, with 44 visits.
- The least frequently visited cell was `(4, 0)`, with 2 visits.
- The explorer appeared to visit the upper part of the grid
  more frequently than the lower part.

The visitation counts include:

- 500 resulting states from environment interactions.
- 10 initial states, one for each episode.

Therefore, the total number of recorded explorer-position
visits is:

`500 + 10 = 510`

The heatmap demonstrates that complete spatial coverage does
not imply uniform visitation.

However, the observed spatial distribution may be influenced
by randomness, collision dynamics, and the expert's movement.

Multiple seeds will be required before drawing general
conclusions about visitation patterns.

### 6.3 Relevance to MAIL

The current visualizations characterize different aspects
of exploration:

| Metric | Interpretation |
|---|---|
| Explorer-position coverage | Number of distinct explorer positions |
| Joint-state coverage | Number of distinct configurations involving both agents |
| Visitation heatmap | Spatial distribution of exploration effort |
| Coverage curves | Rate at which new states are discovered |

These metrics are useful diagnostics.

However, the main thesis objective is to determine whether
exploration produces data that improves expert-policy recovery.

Consequently, the experiments will eventually need metrics
that evaluate the usefulness of visited states for imitation
learning.

---

## 7. Planned Exploration Algorithms

### 7.1 ProtoRL

ProtoRL is the selected novelty-based exploration method.

It uses learned representations and prototypes to encourage
exploration of novel states.

Conceptually:

```text
Image observation
       ↓
Representation encoder
       ↓
Learned representation
       ↓
Prototype-based novelty
       ↓
Intrinsic reward
       ↓
Exploration policy
```

The reference ProtoRL implementation is provided in the
Unsupervised Reinforcement Learning Benchmark (URLB).

The original implementation uses a continuous-action RL
backbone.

Since the current GridWorld has discrete actions, the
ProtoRL-specific intrinsic reward and representation-learning
components will need to be adapted to a discrete-action
learning algorithm.

A shared discrete-action backbone may be useful for comparing
ProtoRL and Disagreement while isolating the effects of
their exploration objectives.

The exact adaptation has not yet been implemented.

### 7.2 Disagreement

Disagreement is the selected uncertainty-based exploration
method.

It uses an ensemble of forward-dynamics models to predict
the next representation of the environment.

The intrinsic reward is based on disagreement between
the models' predictions.

Conceptually:

```text
Image observation
       ↓
Representation encoder
       ↓
Current representation + action
       ↓
Ensemble of forward models
       ↓
Predicted next representations
       ↓
Prediction disagreement
       ↓
Intrinsic reward
       ↓
Exploration policy
```

The original URLB implementation uses a continuous-action
RL backbone.

As with ProtoRL, the intrinsic-reward mechanism will need
to be adapted to the discrete-action GridWorld.

An important experimental question is how Disagreement
behaves under stochastic environment dynamics.

High prediction disagreement may reflect uncertainty
caused by insufficient exploration, but stochastic
transitions may also affect model predictions.

The four environment configurations will help investigate
this distinction.

---

## 8. Current Project Structure

```text
master-thesis/
├── envs/
│   └── gridworld.py
│
├── evaluation/
│   ├── coverage.py
│   └── plots.py
│
├── experiments/
│   └── random_baseline.py
│
├── tests/
│   ├── test_gridworld.py
│   ├── test_coverage.py
│   ├── test_coverage_integration.py
│   └── test_random_baseline.py
│
├── results/
│   ├── random_baseline_coverage.png
│   └── random_baseline_heatmap.png
│
├── README.md
└── requirements.txt
```

The `results/` directory currently contains the figures
generated by the random baseline.

The organization separates:

- Environment dynamics.
- Evaluation metrics.
- Experiment execution.
- Automated tests.
- Experimental results.

This structure is intended to support the future integration
of multiple exploration algorithms.

---

## 9. Testing

The project uses `pytest` for automated testing.

Run the complete test suite from the project root:

```bash
python -m pytest -q
```

### 9.1 GridWorld tests

`tests/test_gridworld.py`

The tests cover:

- Valid initial positions.
- Fixed goal position.
- Invalid grid sizes.
- Deterministic expert behaviour.
- Deterministic and stochastic dynamics.
- Expert stochasticity.
- Reproducibility.
- Explorer-position visitation counts.
- Explorer-position coverage.
- Image encoding.
- Same-cell collisions.
- Position-swap collisions.
- Simultaneous movement.
- Joint-state visitation counts.
- Joint-state coverage.

### 9.2 CoverageTracker tests

`tests/test_coverage.py`

The tests cover:

- Initial-state recording.
- Environment interaction counting.
- Repeated visits.
- Cumulative coverage across episodes.
- Coverage history.
- Invalid positions.
- Resetting statistics.
- Invalid grid sizes.

### 9.3 Integration tests

`tests/test_coverage_integration.py`

The integration test verifies that the cumulative
coverage tracker works correctly across multiple
GridWorld episodes.

### 9.4 Random baseline tests

`tests/test_random_baseline.py`

The tests verify:

- Correct environment interaction counts.
- Correct visitation counts.
- Valid coverage values.
- Reproducibility with fixed seeds.
- Successful generation of both visualization files.

### 9.5 Current test results

At the current checkpoint:

**36 tests pass.**

This provides an initial level of confidence in the
environment, evaluation infrastructure, and random baseline.

---

## 10. Limitations and Open Questions

The current implementation is an initial experimental
framework rather than a complete MAIL evaluation pipeline.

Several limitations remain.

### 10.1 Simple expert policy

The current expert follows a goal-directed heuristic.

Its requested action does not depend on the full joint state.

Therefore, visiting more joint states does not necessarily
reveal more information about the expert policy.

A more suitable expert policy may be required to evaluate
expert-policy recovery.

### 10.2 No expert-policy recovery evaluation

The current experiments measure exploration coverage.

They do not yet collect expert demonstrations or train
an imitation-learning policy.

Therefore, no conclusions can currently be drawn about
whether an exploration method improves imitation performance.

### 10.3 No game-theoretic evaluation

The current GridWorld does not implement a complete
Markov game with strategically derived expert policies.

Consequently, metrics such as Nash gap are not yet
available.

### 10.4 Single-seed experiment

The random baseline has been evaluated using one seed.

Multiple independent runs will be necessary to estimate
average performance and variability.

### 10.5 Small environment

The current baseline uses a 5 × 5 grid.

This is useful for debugging and understanding exploration,
but it may be too simple to reveal meaningful differences
between exploration algorithms.

Larger grids and longer horizons may be required.

### 10.6 Exploration quality versus coverage

High coverage does not necessarily imply that exploration
produces useful information about the expert.

A central question is whether general novelty-based or
uncertainty-based exploration naturally visits states
that are informative for imitation learning.

This question will guide the design of the main experiments.

---

## 11. Next Steps

The next development stages are:

1. Review the current GridWorld design and evaluation metrics.
2. Confirm whether the initial heuristic expert is sufficient for early exploration experiments.
3. Define how expert demonstrations should be collected during exploration.
4. Identify metrics for evaluating expert-policy recovery.
5. Extend the random baseline to multiple seeds.
6. Compare the four combinations of expert and environment stochasticity.
7. Determine suitable interaction budgets and grid sizes.
8. Finalize a shared discrete-action RL backbone.
9. Adapt and implement ProtoRL.
10. Adapt and implement Disagreement.
11. Compare the exploration methods under controlled experimental conditions.
12. Extend the setup to support downstream MAIL evaluation.

The central research objective remains:

**Which exploration strategies collect the most useful information for recovering expert behaviour in deep multi-agent imitation learning?**