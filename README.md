# Master's Thesis — Exploration for Deep Multi-Agent Imitation Learning

This repository contains the experimental work for my Master's thesis on
exploration in deep multi-agent imitation learning (MAIL).

## Research Goal

The goal is to investigate whether unsupervised reinforcement learning
exploration methods can improve the exploration phase of deep MAIL.

In particular, the experiments will compare exploration strategies that use
different principles, starting with:

- **ProtoRL** — state coverage / novelty
- **Disagreement** — dynamics uncertainty

The main question is not only whether an exploration method achieves broad
coverage, but whether it visits states that are informative for recovering the
expert policy.

The objective is not to reproduce the experiments from the reference MAIL
implementation. Instead, the implementation provides a starting point for
understanding the exploration problem and designing experiments that isolate
the effects of different exploration strategies.

## Initial Experimental Setup

The first experiments use a deliberately simple GridWorld.

The purpose of starting with a simple environment is to make the behaviour of
the exploration algorithms interpretable. If an experiment fails in a complex
environment, it can be difficult to distinguish between a problem with the
exploration method, the neural-network architecture, or the implementation.

### Image-based observations

The GridWorld is represented through image observations rather than a
vector representation of the state.

This creates a simple setting in which function approximation can be used
while the underlying environment remains easy to understand.

The current implementation uses RGB observations with shape:

`(grid_size, grid_size, 3)`

Each entity is represented by a different colour.

### Two-agent environment

The GridWorld contains two agents:

1. An **exploring agent**, whose actions are supplied by the exploration
   algorithm.
2. An **expert agent**, which follows a fixed policy.

Both agents move in the same environment.

The current expert policy moves toward a goal using a deterministic
shortest-path heuristic, with optional stochastic deviations.

More precisely, the deterministic expert prioritizes vertical movement
toward the goal, followed by horizontal movement.

This is a preliminary expert policy, not a policy obtained by solving a
Markov game.

In the reference deep MAIL implementation, exploration can be performed
against a fixed expert policy, with both players' actions determining the
next joint state.

Our environment is intended to support the same general interaction pattern,
without necessarily reproducing the reference environment's exact dynamics,
rewards, or experimental configurations.

The current implementation includes interactions through joint transitions
and collision handling.

However, it does not yet define a complete Markov game with game-theoretic
expert policies. This will need to be addressed before evaluating downstream
multi-agent imitation learning performance.

### Fixed goal position

The initial GridWorld uses a fixed goal position in the top-right corner:

`goal_position = (0, grid_size - 1)`

For the default 10 × 10 grid, this corresponds to:

`goal_position = (0, 9)`

The goal remains unchanged across episodes.

At the beginning of each episode:

- the explorer starts at a randomly sampled position,
- the expert starts at a randomly sampled position,
- the two starting positions are distinct,
- neither agent starts on the goal.

A fixed goal reduces variability between episodes and makes exploration
strategies easier to compare.

It also simplifies the analysis of state visitation and expert behaviour.

This is an experimental design decision rather than a requirement of the
reference MAIL algorithm.

Randomized goals may be introduced later as an additional experimental
condition.

### Actions

The current environment has five discrete actions:

| Action | Movement |
|---|---|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |
| 4 | Stay |

Actions that would move an agent outside the grid are clipped to the
grid boundaries.

### Simultaneous movement and joint transitions

The environment uses simultaneous movement.

At each environment step:

1. The explorer selects an action.
2. The fixed expert policy selects an action based on the current state.
3. Environment stochasticity is applied independently to both actions.
4. The intended next positions of both agents are calculated.
5. Potential collisions are checked.
6. Both positions are updated simultaneously.

The next joint state therefore depends on the actions of both agents.

This introduces interactions between the agents while keeping the
environment simple enough to analyze.

### Collision handling

Two types of collisions are implemented.

**Same-cell collision**

If both agents attempt to occupy the same cell after executing their
actions, neither agent moves.

**Position-swap collision**

If the agents attempt to exchange positions in a single step, neither
agent moves.

When neither collision occurs, both agents move to their intended next
positions.

Consequently, the agents cannot move into the same cell or exchange
positions during normal environment transitions.

The environment reports collision information through the `info`
dictionary returned by `step()`:

- `collision`
- `same_cell_collision`
- `position_swap_collision`

This information can be useful when analyzing how exploration affects
the expert's movement.

For example, an explorer may prevent the expert from moving toward
its goal by occupying a strategically relevant position.

Collision handling is an initial design choice inspired by the reference
GridWorld environment.

Its effect on exploration and expert-policy recovery should be evaluated
before the main experiments.

### Expert policy and limitations

The current expert follows a fixed goal-directed policy.

In the deterministic case, the expert:

1. Moves vertically toward the goal until it reaches the correct row.
2. Moves horizontally toward the goal until it reaches the correct column.
3. Requests `STAY` once it reaches the goal.

The expert does not explicitly consider the explorer's position when
selecting an action.

However, its actual movement depends on the explorer because collisions
can prevent the requested action from being executed.

For example, if the explorer blocks the expert's preferred route, the
expert may repeatedly attempt the same action rather than choosing
an alternative path.

Therefore, the environment currently provides:

- **Interactive transition dynamics:** the explorer can influence the
  expert's trajectory.
- **A fixed expert policy:** the expert follows a predefined heuristic.
- **No strategic expert planning:** the expert does not calculate a
  best response or adapt its route to avoid the explorer.

This distinction is important for the thesis.

The environment is suitable as an initial setting for studying
exploration behaviour, but additional work is required before it can
support meaningful game-theoretic evaluation of MAIL.

### Two-phase experimental approach

The initial experiments will follow a two-phase approach.

**Phase 1 — Exploration baseline**

Use the current goal-following expert to develop, test, and compare
exploration strategies.

The purpose is to investigate how ProtoRL and Disagreement behave in a
simple environment with image observations, joint transitions, and
configurable stochasticity.

This phase focuses on exploration behaviour and diagnostic metrics rather
than equilibrium recovery.

**Phase 2 — MAIL evaluation**

Introduce an expert whose decisions depend meaningfully on the joint state.

The purpose is to investigate whether different exploration strategies
collect data that improves expert-policy recovery.

Additional game-theoretic structure, including suitable rewards and
expert policies, may be required to evaluate downstream MAIL performance.

This two-phase approach is a provisional experimental plan and should be
reviewed with the thesis supervisor before the main experiments.

### Sources of stochasticity

Two sources of stochasticity can be independently controlled:

1. **Environment dynamics**
2. **Expert policy**

This allows the exploration methods to be studied under different conditions:

| Environment dynamics | Expert policy |
|---|---|
| Deterministic | Deterministic |
| Deterministic | Stochastic |
| Stochastic | Deterministic |
| Stochastic | Stochastic |

#### Environment stochasticity

The parameter `dynamics_stochasticity` controls the probability that a
requested action is replaced by another randomly selected action.

The replacement is sampled uniformly from the other available actions.

This mechanism is applied independently to the explorer's and expert's
requested actions.

When `dynamics_stochasticity = 0`, requested actions are always executed.

When `dynamics_stochasticity = 1`, every requested action is replaced by
a different action.

The resulting executed actions are then used to calculate intended
movements and determine whether a collision occurs.

#### Expert stochasticity

The parameter `expert_stochasticity` controls the probability that the
expert deviates from its deterministic goal-following action.

When a deviation occurs, the expert selects uniformly from the other
available actions.

When `expert_stochasticity = 0`, the expert always requests its
deterministic goal-following action.

When `expert_stochasticity = 1`, the expert always requests a different
action.

These definitions are initial implementation choices. Alternative models
of stochasticity may be considered if they are more appropriate for the
experiments.

Separating the two sources of stochasticity is particularly important for
uncertainty-based exploration.

For example, dynamics disagreement may reflect uncertainty caused by
insufficient experience, but it may also be affected by inherently
stochastic transitions.

### What exploration should achieve

The exploring agent itself is not the final object of interest.

Its purpose is to visit states in which observing or querying the fixed
expert provides useful information for imitation learning.

Therefore, the experiments should distinguish between:

- **general state coverage**, and
- **coverage of states that are informative for expert-policy recovery**.

Simple state coverage and visitation heatmaps can be used as initial
diagnostic metrics, but they are not sufficient on their own to evaluate
exploration for MAIL.

The introduction of collisions makes this distinction particularly
relevant.

An exploration strategy might visit many different cells without
interacting meaningfully with the expert.

Another strategy might achieve lower overall coverage but repeatedly
encounter situations in which the expert's behaviour is affected by
the explorer.

Whether these interactions provide more useful information for
imitation learning remains an experimental question.

## Coverage Metrics

The implementation distinguishes between two types of coverage:

1. Explorer-position coverage.
2. Joint-state coverage.

Both metrics are available at two levels:

- **Episode-level coverage**, recorded by the GridWorld environment.
- **Cumulative coverage**, recorded by a separate evaluation component.

### Explorer-position coverage

The environment records how often the explorer visits each grid cell.

The corresponding coverage metric is:

`number of unique explorer cells visited / total number of grid cells`

For a grid containing N cells:

`C_explorer = unique_explorer_positions / N`

For the default 10 × 10 grid:

`N = 100`

This metric measures how broadly the explorer moves through the grid.

It does not account for the expert's position.

The episode-level metric is available through:

```python
env.get_state_coverage()
```

The underlying visitation counts are stored in:

```python
env._visit_counts
```

### Joint-state coverage

The environment also records visits to joint states containing the
positions of both agents.

A joint state is represented as:

`state = (explorer_row, explorer_col, expert_row, expert_col)`

For example:

`state = (2, 3, 5, 6)`

and:

`state = (2, 3, 7, 8)`

are different joint states, even though the explorer occupies the same
cell in both cases.

#### Number of possible joint states

For a grid containing N cells, each agent can occupy one of those cells.

Without collision restrictions, there would be:

`N²`

possible joint states.

However, the current collision rules prevent the agents from occupying
the same cell during normal environment transitions.

Therefore, the number of valid joint states is:

`N × (N - 1)`

For the default 10 × 10 grid:

`N = 100`

`total_joint_states = 100 × 99 = 9900`

This denominator counts all configurations with distinct agent positions.
It does not imply that every configuration will be visited by the current
expert policy during a finite experiment.

#### Joint-state coverage metric

Joint-state coverage is defined as:

`C_joint = unique_joint_states_visited / (N × (N - 1))`

The episode-level metric is available through:

```python
env.get_joint_state_coverage()
```

The underlying visitation counts are stored in:

```python
env._joint_visit_counts
```

The dictionary maps each joint state to the number of times it has
been visited.

For example:

```python
{
    (2, 3, 5, 6): 4,
    (2, 3, 4, 6): 2,
}
```

Both states contribute once to the coverage numerator, regardless of
how often they are visited.

### Episode-level coverage tracking

The GridWorld records:

1. The initial state when `reset()` is called.
2. The resulting state after every `step()`.

If a collision prevents both agents from moving, the current state
is recorded again.

Repeated visits increase visitation counts but do not increase coverage.

Both environment-level coverage trackers are reset at the beginning
of every episode.

Therefore, the environment's coverage methods measure coverage within
a single episode.

## Cumulative Coverage Evaluation

Cumulative coverage is implemented separately from the GridWorld in:

`evaluation/coverage.py`

The `CoverageTracker` class tracks visited explorer positions and joint
states across multiple episodes.

This separation is intentional.

The GridWorld is responsible for environment dynamics, while the
evaluation component is responsible for measuring exploration progress.

The same evaluation component can therefore be reused for different
exploration algorithms.

### Why cumulative coverage is needed

Episode-level coverage measures the states visited during one episode.

However, exploration algorithms are generally trained and evaluated
over many episodes.

We therefore want to measure how many distinct states an algorithm
discovers throughout the exploration process.

For example:

| Episode | Joint states visited |
|---|---|
| 1 | A, B, C |
| 2 | B, C, D |
| 3 | C, D, E |

Each episode visits three joint states.

However, the cumulative number of distinct joint states is five:

`{A, B, C, D, E}`

Cumulative coverage is calculated using the union of visited states,
not the sum or average of episode-level coverage.

### CoverageTracker

Create a tracker using:

```python
from evaluation.coverage import CoverageTracker

tracker = CoverageTracker(grid_size=10)
```

The tracker maintains:

- `explorer_visit_counts`: cumulative explorer-position visitation counts.
- `joint_visit_counts`: cumulative joint-state visitation counts.
- `total_steps`: number of recorded environment interactions.
- `coverage_history`: coverage measurements after each recorded interaction.

The tracker is independent of the exploration algorithm.

### Recording initial states

At the beginning of each episode, reset the environment and record
the initial state:

```python
observation, info = env.reset()

tracker.record_state(
    env._explorer_position,
    env._expert_position,
)
```

The `record_state()` method updates visitation counts without increasing
the environment interaction counter.

This is intentional because resetting the environment is not counted
as an environment interaction.

The tracker is not reset between episodes.

### Recording environment interactions

After executing an environment step:

```python
observation, reward, terminated, truncated, info = env.step(action)

tracker.record_step(
    env._explorer_position,
    env._expert_position,
)
```

The `record_step()` method:

1. Records the resulting explorer position.
2. Records the resulting joint state.
3. Increments the environment interaction counter.
4. Stores the updated coverage measurements.

This produces a coverage history indexed by the number of environment
interactions.

### Accessing cumulative coverage

Cumulative explorer-position coverage:

```python
tracker.get_explorer_coverage()
```

Cumulative joint-state coverage:

```python
tracker.get_joint_coverage()
```

Both metrics return values between 0 and 1.

For example, a value of `0.25` corresponds to 25% coverage.

### Coverage history

After each recorded environment interaction, the tracker stores an entry
containing:

- `step`
- `explorer_coverage`
- `joint_coverage`

For example:

```python
[
    {
        "step": 1,
        "explorer_coverage": 0.02,
        "joint_coverage": 0.0002,
    },
    {
        "step": 2,
        "explorer_coverage": 0.03,
        "joint_coverage": 0.0003,
    },
]
```

These values are illustrative.

The actual coverage depends on the grid size and the visited states.

This history will allow us to plot coverage against environment
interactions.

### Comparing exploration efficiency

Tracking coverage over environment interactions will allow us to compare
not only how much of the state space an algorithm explores, but also
how quickly it discovers new states.

For example, ProtoRL and Disagreement may eventually reach similar
coverage while requiring different numbers of interactions.

Coverage curves can help identify these differences.

To make the comparisons meaningful, experiments should use:

- the same environment configuration,
- the same interaction budget,
- comparable random seeds,
- the same coverage definitions,
- the same evaluation procedure.

The evaluation should eventually be repeated across multiple seeds
to account for variability.

### Limitations of cumulative coverage

Cumulative coverage measures the diversity of visited states.

It does not directly measure whether those states are informative for
imitation learning.

This is especially important in the current GridWorld because the
heuristic expert's requested action depends primarily on its own
position rather than the complete joint state.

Therefore, visiting many different joint states may not provide
proportionally more information about the expert policy.

The main thesis experiments will need additional evaluation metrics
that assess whether the collected exploration data improves
expert-policy recovery.

Potential future measurements include:

- expert-policy prediction accuracy,
- imitation performance,
- coverage of states relevant to expert behaviour,
- game-theoretic evaluation when suitable expert policies are available.

These metrics have not yet been implemented.

## Exploration Methods

### ProtoRL

ProtoRL is used as the first state-coverage / novelty-based exploration
method.

The reference implementation is the ProtoRL implementation provided in
the URLB repository.

ProtoRL does not measure novelty directly from raw GridWorld coordinates.

Instead, observations are mapped into a learned representation space
using an encoder.

ProtoRL learns prototypes in this representation space and uses a
novelty-based intrinsic reward to encourage exploration.

Conceptually:

pixel observation
→ learned representation
→ prototype-based novelty estimate
→ intrinsic reward
→ exploration policy

The original URLB implementation uses DDPG/DrQ-v2 as the underlying RL
algorithm.

This part is designed for continuous-control environments and therefore
cannot be transferred directly to the discrete-action GridWorld.

The ProtoRL-specific representation learning and intrinsic-reward mechanism
should be kept conceptually separate from the RL backbone.

A discrete-action RL method will therefore be required to optimize the
ProtoRL intrinsic reward in the GridWorld.

The environment needs to provide transitions containing at least:

- current image observation,
- explorer action,
- next image observation.

The expert policy is not required to compute the ProtoRL intrinsic reward.

Expert information will instead be relevant when evaluating whether the
states visited during exploration are useful for imitation learning.

### Disagreement

Disagreement is used as the uncertainty-based exploration method.

The reference implementation is the Disagreement implementation provided
in the URLB repository.

Disagreement learns an ensemble of forward-dynamics models.

Each model receives the encoded current observation together with the
action and predicts the encoded next observation.

Conceptually:

pixel observation
→ learned representation
→ representation + action
→ ensemble of next-representation predictions
→ variance across predictions
→ intrinsic reward
→ exploration policy

The intrinsic reward is high when the ensemble models disagree about
the predicted next representation.

This provides a measure of uncertainty about the environment dynamics.

The ensemble models are trained using transitions containing:

- current image observation,
- explorer action,
- next image observation.

As with ProtoRL, the original URLB implementation uses DDPG as the
underlying RL algorithm.

The Disagreement intrinsic-reward mechanism should therefore be separated
from the continuous-control RL backbone and combined with a
discrete-action RL method for the GridWorld.

An important experimental question is how Disagreement behaves when the
environment dynamics themselves are stochastic.

In that case, prediction uncertainty may reflect inherent unpredictability
rather than uncertainty that is useful for recovering the expert policy.

The independently configurable dynamics stochasticity in the GridWorld
will allow this to be investigated.

## Current Status

### GridWorld environment

The initial GridWorld infrastructure supports:

- image observations,
- two moving agents,
- five discrete actions,
- a fixed goal location,
- random initial agent positions,
- simultaneous movement,
- joint transitions,
- same-cell collision handling,
- position-swap collision handling,
- collision diagnostics,
- independently configurable environment stochasticity,
- independently configurable expert stochasticity,
- reproducible random seeds,
- episode-level explorer-position visitation counts,
- episode-level explorer-position coverage,
- episode-level joint-state visitation counts,
- episode-level joint-state coverage.

The environment currently provides zero reward to the exploring agent.

This is suitable as an initial interface for reward-free exploration,
where intrinsic rewards will be supplied by the exploration algorithm.

It does not yet provide the game-theoretic rewards required for evaluating
equilibrium recovery.

The expert is currently a goal-following heuristic rather than a
game-theoretically derived expert policy.

### Evaluation infrastructure

The evaluation component currently supports:

- cumulative explorer-position visitation counts,
- cumulative joint-state visitation counts,
- cumulative coverage across episodes,
- environment interaction counting,
- coverage history indexed by interactions,
- resetting cumulative statistics,
- validation of recorded positions.

The coverage tracker has been tested independently and through
integration tests with the GridWorld.

The integration test runs two episodes using random explorer actions,
records the initial state of each episode, and tracks all resulting states.

It verifies that:

- the environment resets between episodes,
- the cumulative tracker preserves previously visited states,
- interaction counts remain correct,
- coverage does not decrease,
- coverage values remain within valid bounds.

This provides the initial evaluation infrastructure needed to compare
exploration methods using consistent coverage metrics.

### Project structure

The current relevant files are:

```text
master-thesis/
├── envs/
│   └── gridworld.py
├── evaluation/
│   └── coverage.py
├── tests/
│   ├── test_gridworld.py
│   ├── test_coverage.py
│   └── test_coverage_integration.py
├── README.md
└── requirements.txt
```

## Design Decisions to Validate

Several details of the current GridWorld are implementation choices rather
than requirements of the experimental setup.

These should be validated before the main experiments:

- grid size,
- episode horizon,
- action set,
- image encoding,
- expert policy,
- expert behaviour at the goal,
- definition of environment stochasticity,
- definition of expert stochasticity,
- treatment of collisions,
- whether the expert should react strategically to the explorer,
- definition of expert-relevant coverage,
- selection of experimental interaction budgets,
- selection of evaluation seeds,
- collection of expert demonstrations,
- integration with the deep MAIL pipeline,
- evaluation of recovered policies.

The following components have been implemented and tested:

- fixed goal position,
- simultaneous movement,
- same-cell collision handling,
- position-swap collision handling,
- explorer-position coverage,
- joint-state coverage,
- cumulative coverage tracking,
- coverage history over environment interactions.

These decisions may be revisited based on the requirements of the
exploration experiments.

## Testing

The project currently includes three test files:

### `tests/test_gridworld.py`

Tests the GridWorld environment, including:

- valid and distinct initial positions,
- fixed goal location,
- goal placement for different grid sizes,
- initial positions away from the goal,
- invalid grid sizes,
- deterministic expert behaviour without obstruction,
- deterministic environment dynamics,
- stochastic environment dynamics,
- stochastic expert behaviour,
- reproducibility,
- visitation counts,
- explorer-position coverage,
- image encoding when entities overlap,
- same-cell collisions,
- position-swap collisions,
- simultaneous movement without collisions,
- visitation counting after collisions,
- initial joint-state tracking,
- repeated joint-state visits,
- distinct joint states with the same explorer position,
- joint-state visitation counting after collisions.

### `tests/test_coverage.py`

Tests the cumulative coverage tracker, including:

- recording initial states without counting interactions,
- repeated visits,
- cumulative coverage across episodes,
- coverage history,
- invalid positions,
- resetting cumulative statistics,
- invalid grid sizes.

### `tests/test_coverage_integration.py`

Tests the connection between the GridWorld environment and the
cumulative coverage tracker.

The integration test verifies that cumulative coverage is correctly
recorded across multiple environment episodes.

### Running the tests

Run the complete test suite using:

```bash
python -m pytest -q
```

At the current checkpoint, all **32 tests pass**.

## Next Steps

The next development steps are:

1. Implement a reproducible experiment runner that connects the
   GridWorld, exploration policy, and coverage tracker.
2. Establish a random exploration baseline.
3. Generate explorer-position visitation heatmaps and cumulative
   coverage curves.
4. Define an initial set of environment configurations covering the
   four combinations of deterministic/stochastic expert policies and
   environment dynamics.
5. Decide how to collect expert demonstrations from visited states.
6. Identify evaluation metrics that measure whether exploration
   improves expert-policy recovery.
7. Finalize the interfaces required by ProtoRL and Disagreement.
8. Implement and compare ProtoRL and Disagreement.
9. Extend the environment and expert policies for downstream
   MAIL evaluation.

The central research objective remains to understand which exploration
strategies collect the most useful information for recovering expert
behaviour, rather than simply maximizing state coverage.