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

### State coverage

The current implementation records how often the explorer visits each
grid cell.

The corresponding coverage metric is:

`number of unique explorer cells visited / total number of grid cells`

This measures **explorer-position coverage**.

It does not measure coverage of the full joint state.

For a two-player environment with a fixed goal, the joint state can be
represented as:

`state = (player_1_position, player_2_position)`

Two observations with the explorer in the same cell but the expert in
different cells correspond to different joint states.

Consequently, explorer-position coverage is useful as a preliminary
diagnostic but is not sufficient to characterize exploration in a
multi-agent environment.

Joint-state coverage and expert-relevant coverage should be considered
when defining the experimental metrics.

Collision frequency may also be useful as a diagnostic, although it
does not directly measure how informative the collected data is.

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
- visitation heatmaps,
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
- definition of joint-state coverage,
- definition of expert-relevant coverage,
- collection of expert demonstrations,
- integration with the deep MAIL pipeline,
- evaluation of recovered policies.

The following design decisions have been implemented and tested:

- fixed goal position,
- simultaneous movement,
- same-cell collision handling,
- position-swap collision handling.

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
- visitation counting after collisions.

Run the tests using:

```bash
python -m pytest -q
```

At the current checkpoint, all 20 tests pass.

## Next Steps

Before implementing the exploration algorithms:

1. Continue examining the reference deep MAIL environment and exploration
   interface.
2. Evaluate whether the current goal-following expert is appropriate for
   studying informative exploration.
3. Determine whether additional game-theoretic structure is required.
4. Define joint-state coverage and other experimental metrics.
5. Establish how exploration data will be used for expert-policy recovery.
6. Finalize the interfaces required by ProtoRL and Disagreement.
7. Implement and compare the exploration methods.