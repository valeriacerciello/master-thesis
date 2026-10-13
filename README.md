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

The metric is available through:

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

This is important because the expert's position changes the situation
encountered by the exploring agent.

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

#### Joint-state coverage metric

Joint-state coverage is defined as:

`C_joint = unique_joint_states_visited / (N × (N - 1))`

The metric is available through:

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

This indicates that the first joint state was visited four times and
the second joint state twice.

Both states contribute once to the coverage numerator.

#### When visits are recorded

The environment records:

1. The initial joint state when `reset()` is called.
2. The resulting joint state after every `step()`.

If a collision prevents both agents from moving, the current joint
state is recorded again.

Repeated visits increase the visitation count but do not increase
coverage.

#### Episode-level tracking

Both coverage trackers are reset at the beginning of every episode.

Therefore, the current methods measure coverage within a single episode.

For the main experiments, cumulative coverage across multiple episodes
will also be needed.

This can be implemented in the experimental evaluation pipeline by
aggregating visited states across episodes.

### Interpreting coverage for MAIL

Explorer-position coverage and joint-state coverage provide complementary
information.

For example, two exploration methods may visit the same number of
explorer positions but different numbers of joint states.

This would indicate that the methods encounter different levels of
diversity in the configurations involving both agents.

However, higher joint-state coverage does not necessarily imply that
the collected data is more informative for imitation learning.

The final evaluation should investigate whether exploration improves
the recovery of expert behaviour.

This will require additional metrics beyond coverage, potentially
including expert-policy prediction accuracy, imitation performance,
and game-theoretic evaluation when the appropriate environment and
expert policies are available.

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
- explorer-position visitation counts,
- explorer-position coverage measurement,
- joint-state visitation counts,
- joint-state coverage measurement,
- episode-level coverage tracking,
- automated tests for the core environment behaviour.

The environment currently provides zero reward to the exploring agent.

This is suitable as an initial interface for reward-free exploration,
where intrinsic rewards will be supplied by the exploration algorithm.

It does not yet provide the game-theoretic rewards required for evaluating
equilibrium recovery.

The expert is currently a goal-following heuristic rather than a
game-theoretically derived expert policy.

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
- cumulative coverage across episodes,
- collection of expert demonstrations,
- integration with the deep MAIL pipeline,
- evaluation of recovered policies.

The following design decisions have been implemented and tested:

- fixed goal position,
- simultaneous movement,
- same-cell collision handling,
- position-swap collision handling,
- explorer-position coverage,
- joint-state coverage.

These decisions may be revisited based on the requirements of the
exploration experiments.

## Testing

The environment has automated tests covering:

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

Run the tests using:

```bash
python -m pytest -q
```

At the current checkpoint, all 24 tests pass.

## Next Steps

Before implementing the exploration algorithms:

1. Define how to aggregate coverage across multiple episodes.
2. Determine which additional exploration metrics are needed, particularly
   metrics related to expert-policy recovery.
3. Establish how exploration data will be used to query the expert and
   construct imitation-learning datasets.
4. Finalize the interfaces required by ProtoRL and Disagreement.
5. Implement a simple exploration baseline for comparison.
6. Implement and compare ProtoRL and Disagreement.
7. Extend the environment and expert policies for downstream MAIL evaluation.