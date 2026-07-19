# Constrained form-finding

Plain [form-finding](form_finding.md) hands you *an* equilibrium: pick force densities, solve, done.
But a design brief rarely asks for just any equilibrium.
It asks for the shape whose arch segments stay between three-quarters of a meter and a meter, or the one that carries its load with the least material, or the roof that touches down exactly on four given points.

Constrained form-finding is how you ask for *that* shape.
You describe the equilibrium you want as a loss to minimize and a set of bounds to respect, and `constrained_fdm` searches the space of force densities (and, if you like, supports and loads) for the equilibrium that fits best.

```python
from jax_fdm.equilibrium import constrained_fdm

optimized = constrained_fdm(datastructure, optimizer, loss, parameters, constraints)
```

Under the hood `constrained_fdm` runs the same force density solve as `fdm`, over and over, inside an optimization loop: each iteration nudges the parameters, re-solves for equilibrium, and scores the result against your loss.
Because the whole solve is differentiable, the optimizer gets exact gradients for free.

The rest of this guide is a tour of the five ingredients you assemble, from the variables the optimizer moves to the algorithm that moves them.

## The five ingredients

### 1. Parameters: what the optimizer may change

A **parameter** marks one quantity as a design variable, something the optimizer is allowed to tune, optionally between bounds.
By far the most common is the force density of an edge:

```python
from jax_fdm.optimization import EdgeForceDensityParameter


parameters = [EdgeForceDensityParameter(edge, -20.0, -0.1) for edge in network.edges()]
```

Each parameter takes an element key and a lower and upper bound (`None` on either side leaves it unbounded).
Here every edge force density is free to move within `[-20.0, -0.1]`, keeping the structure in compression.

You are not limited to force densities.
Support positions (`NodeSupportXParameter`, `NodeSupportZParameter`, …) and applied loads (`NodeLoadXParameter`, …) can be design variables too, and group variants (`EdgeGroupForceDensityParameter`, …) tie many elements to a single shared value.

!!! note "Parameters are optional"

    If you pass `parameters=None`, the optimizer falls back to a sensible default: every edge force density becomes a free variable, unbounded.
    So the quickest constrained run needs no explicit parameters at all, and you reach for them when you want bounds or want to expose supports and loads.

### 2. Goals: the targets you aim for

A **goal** is a soft target on an equilibrium quantity: reach this length, hover over this point, minimize this force.
Soft means the optimizer trades goals off against one another rather than satisfying any single one exactly.

Goals are the subject of their own guide, [goals](goals.md), which covers what a goal is and how keys become indices.
Here it is enough to know that JAX FDM ships a large bank of them and that you create one goal per element you want to steer:

```python
from jax_fdm.goals import NetworkLoadPathGoal


goals = [NetworkLoadPathGoal()]
```

### 3. The loss: how goals become a single number

An optimizer minimizes one scalar.
A **loss** is what turns your list of goals into that scalar, by measuring how far each goal's prediction sits from its target and summing the misfit.

You wrap your goals in an **error term** and hand the term to `Loss`:

```python
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError


loss = Loss(SquaredError(goals=goals))
```

`SquaredError` is the workhorse: the weighted sum of squared prediction-target gaps.
JAX FDM offers a family of error terms for when the squared gap is not what you want:

| Error term | Measures |
| --- | --- |
| `SquaredError` | the summed squared gap to target |
| `MeanSquaredError` | the same, averaged over goals |
| `RootMeanSquaredError` | the root of the mean squared gap, back in the units of the quantity |
| `AbsoluteError`, `MeanAbsoluteError` | the absolute gap, less swayed by outliers |
| `PredictionError`, `MeanPredictionError` | the predicted quantity itself, no target, for minimizing something like load path toward zero |
| `LogMaxError` | a soft one-sided barrier that penalizes only overshoot past the target |

A loss can hold several error terms at once, each with its own `alpha` weight scaling its contribution, so you can, say, chase a target shape while gently minimizing load path.
You can also add a **regularizer** such as `L2Regularizer(alpha)`, which penalizes the force densities themselves to keep the solution well-behaved.

!!! tip "Prediction errors need no target"

    Most error terms compare a prediction against a goal's target.
    `PredictionError` and its mean cousin skip the target and penalize the raw predicted quantity, which is exactly what you want when the aim is to *minimize* a quantity (total load path, total area) rather than drive it to a set value.

### 4. Constraints: the lines you will not cross

Where a goal is a soft preference, a **constraint** is a hard bound: the optimizer must keep the constrained quantity between its limits, full stop.

```python
from jax_fdm.constraints import EdgeLengthConstraint


constraints = [EdgeLengthConstraint(edge, 0.75, 1.0) for edge in network.edges()]
```

Constraints are covered in [constraints](constraints.md).
Two things matter when composing a problem: they are honored only by the optimizers that support them (`SLSQP` and `IPOPT`), and they are optional, leave them out for an unconstrained minimization.

### 5. The optimizer: who does the searching

The **optimizer** is the algorithm that walks the parameter space toward a minimum.
JAX FDM wraps a spread of them, sorted by what they can do:

- **Gradient-based, unconstrained** — `LBFGSB`, `BFGS`, `GradientDescent`, and the second-order Newton and trust-region family (`NewtonCG`, `TruncatedNewton`, `TrustRegionNewton`, …). Fast, and the usual first choice when you have no hard constraints.
- **Gradient-based, constrained** — `SLSQP`, `IPOPT`, `TrustRegionConstrained`. These are the ones that honor the constraints from ingredient 4.
- **Gradient-free** — `DifferentialEvolution`, `DualAnnealing`, `Powell`, `NelderMead`. Slower, but they need no gradients and can escape local minima, useful for rough or non-smooth objectives.

```python
from jax_fdm.optimization import SLSQP


optimizer = SLSQP()
```

## Putting it together

Here is a complete constrained form-finding run, the same one featured in the [arch optimization](../examples/arch.md) example.
The brief: find the arch that minimizes total load path while keeping every segment between 0.75 and 1 meter.

```python
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.optimization import SLSQP
from jax_fdm.constraints import EdgeLengthConstraint
from jax_fdm.goals import NetworkLoadPathGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import PredictionError


length = 5.0
num_segments = 10
segment_length = length / num_segments

xs = [-length / 2.0 + i * segment_length for i in range(num_segments + 1)]
nodes = [[x, 0.0, 0.0] for x in xs]
edges = [(i, i + 1) for i in range(num_segments)]

network = FDNetwork.from_nodes_and_edges(nodes, edges)

network.edges_forcedensities(q=-1.0)
network.nodes_supports(keys=[node for node in network.nodes() if network.is_leaf(node)])
network.nodes_loads([0.0, 0.0, -0.3])

loss = Loss(PredictionError(goals=[NetworkLoadPathGoal()]))
constraints = [EdgeLengthConstraint(edge, 0.75, 1.0) for edge in network.edges()]

optimized = constrained_fdm(
    network,
    optimizer=SLSQP(),
    loss=loss,
    constraints=constraints,
)
```

Notice there are no explicit `parameters` here: with `parameters=None` the optimizer treats every edge force density as a free variable, which is exactly what this problem wants.
The `PredictionError` on a `NetworkLoadPathGoal` means "make the load path as small as you can" (no target), while the `EdgeLengthConstraint` list keeps the segments in range.

### Reading the result

`constrained_fdm`, like `fdm`, returns a *copy* of the datastructure in equilibrium, leaving the original untouched.
The optimizer's solution is baked into that copy: every edge reports its optimized force density `q`, along with the resulting `length` and `force`, and every node reports its equilibrium `xyz`, residual, and load.
You read the outcome straight off `optimized` with the usual datastructure accessors, or hand it to the [`Viewer`](../api/jax_fdm.visualization.viewers.md) to see it.

!!! note "Constraints run on the dense solver"

    The sparse solver does not support constraints yet.
    When you pass constraints with the default `sparse=True`, `constrained_fdm` switches the solve to dense and prints a short notice.
    Unconstrained problems keep the sparse solver and its scaling.

## Where to next

- New to the objects `constrained_fdm` juggles under the hood? Start with [form-finding](form_finding.md).
- Need a target or a bound the library does not ship? Write a [custom goal](custom_goals.md) or [custom constraint](custom_constraints.md).
